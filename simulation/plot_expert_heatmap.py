#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


PLACEMENT_RE = re.compile(
    r"expert placement gpu placement_owner=(?P<owner>\d+) "
    r"kind=(?P<kind>\w+) gpu=(?P<gpu>\d+:\d+) experts=\[(?P<experts>[^\]]*)\]"
)


def load_config(config_path):
    config = {}
    if not config_path.exists():
        return config
    with config_path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts:
                config[parts[0]] = parts[1:]
    return config


def config_int(config, key, default):
    values = config.get(key, [])
    if not values:
        return default
    try:
        return int(float(values[0]))
    except ValueError:
        return default


def config_bool(config, key, default=False):
    values = config.get(key, [])
    if not values:
        return default
    return values[0] not in ("0", "false", "False", "no", "No")


def resolve_sim_path(path_value, config_path):
    path = Path(path_value)
    if path.is_absolute():
        return path
    script_root = Path(__file__).resolve().parent
    candidates = [
        script_root / path,
        config_path.parent / path,
        Path.cwd() / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return script_root / path


def parse_uint_csv_line(line):
    values = []
    for token in line.strip().split(","):
        token = token.strip()
        if not token:
            continue
        try:
            values.append(int(float(token)))
        except ValueError:
            values.append(0)
    return values


def load_trace_device_experts(device_path, layer_id):
    try:
        data = json.loads(device_path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    layers = data.get("layer_list", []) if isinstance(data, dict) else data
    if not isinstance(layers, list):
        return []
    target = None
    for layer in layers:
        if isinstance(layer, dict) and int(layer.get("layer_id", -1)) == layer_id:
            target = layer
            break
    if target is None and 0 <= layer_id < len(layers):
        target = layers[layer_id]
    if not isinstance(target, dict):
        return []
    devices = target.get("device_list", [])
    if not isinstance(devices, list):
        return []
    device_experts = []
    for device in devices:
        experts = device.get("device_expert", []) if isinstance(device, dict) else []
        if isinstance(experts, list):
            device_experts.append([int(expert) for expert in experts])
    return device_experts


def load_moe_layer_count(device_path):
    try:
        data = json.loads(device_path.read_text())
    except (OSError, json.JSONDecodeError):
        return 0
    if isinstance(data, dict):
        try:
            count = int(data.get("moe_layer_count", 0) or 0)
        except (TypeError, ValueError):
            count = 0
        if count > 0:
            return count
        layers = data.get("layer_list", [])
        if isinstance(layers, list):
            return len(layers)
    if isinstance(data, list):
        return len(data)
    return 0


def load_decode_counts(decode_path, layer_id):
    try:
        with decode_path.open() as f:
            line = ""
            for _ in range(layer_id + 1):
                line = f.readline()
                if not line:
                    break
    except OSError:
        return []
    if not line:
        return []
    return parse_uint_csv_line(line)


def load_trace_layer_targets(config_path, device_path, decode_dir, layer_id, expert_num):
    device_experts = load_trace_device_experts(device_path, layer_id)
    if not device_experts:
        return []

    targets = [0 for _ in range(expert_num)]
    for device_id, experts in enumerate(device_experts):
        if not experts:
            continue
        counts = load_decode_counts(decode_dir / f"decode_{device_id}.csv", layer_id)
        for expert, count in zip(experts, counts):
            if expert >= len(targets):
                targets.extend([0] * (expert + 1 - len(targets)))
            targets[expert] += count
    return targets


def scale_targets(targets, divisor):
    if divisor <= 1:
        return targets
    return [0 if count == 0 else (count + divisor - 1) // divisor for count in targets]


def uniform_preserve_total(targets, expert_num):
    total = sum(targets)
    count = max(expert_num, len(targets), 1)
    out = [total // count for _ in range(count)]
    for expert in range(total % count):
        out[expert] += 1
    return out


def load_trace_freq(config, config_path):
    if not config_bool(config, "TRACE_DISPATCH_ENABLE"):
        return []
    if not config_bool(config, "TRACE_ACCESS_TARGETS_AS_EXPERT_FREQ", True):
        return []
    device_values = config.get("TRACE_DEVICE_FILE", [])
    decode_values = config.get("TRACE_DECODE_DIR", [])
    if not device_values or not decode_values:
        return []

    layer_id = config_int(config, "TRACE_LAYER_ID", 0)
    expert_num = max(1, config_int(config, "EXPERT_NUM", 1))
    divisor = max(1, config_int(config, "TRACE_ACCESS_DIVISOR", 1))
    device_path = resolve_sim_path(device_values[0], config_path)
    decode_dir = resolve_sim_path(decode_values[0], config_path)

    if "TRACE_MOE_LAYER_COUNT" in config:
        layer_count = config_int(config, "TRACE_MOE_LAYER_COUNT", 0)
        if layer_count <= 0:
            layer_count = load_moe_layer_count(device_path)
        if layer_count <= 0:
            layer_count = layer_id + 1
        targets = [0 for _ in range(expert_num)]
        for current_layer in range(layer_count):
            layer_targets = load_trace_layer_targets(
                config_path, device_path, decode_dir, current_layer, expert_num)
            if not layer_targets:
                continue
            layer_targets = scale_targets(layer_targets, divisor)
            if config_bool(config, "TRACE_UNIFORM_ACCESS_TARGETS"):
                layer_targets = uniform_preserve_total(layer_targets, expert_num)
            if len(layer_targets) > len(targets):
                targets.extend([0] * (len(layer_targets) - len(targets)))
            for expert, count in enumerate(layer_targets):
                targets[expert] += count
        return [float(value) for value in targets]

    targets = load_trace_layer_targets(config_path, device_path, decode_dir, layer_id, expert_num)
    if not targets:
        return []

    if divisor > 1:
        targets = scale_targets(targets, divisor)

    if config_bool(config, "TRACE_UNIFORM_ACCESS_TARGETS"):
        total = sum(targets)
        count = max(expert_num, len(targets))
        target_count = config_int(config, "TRACE_UNIFORM_ACCESS_TARGET_COUNT", 0)
        if target_count <= 0 and total > 0:
            target_count = (total + count - 1) // count
        targets = [target_count for _ in range(count)]
    return [float(value) for value in targets]


def load_expert_freq(config_path):
    config = load_config(config_path)
    trace_freq = load_trace_freq(config, config_path)
    if trace_freq:
        return trace_freq
    freq = []
    for value in config.get("EXPERT_ACCESS_FREQ", []):
        try:
            freq.append(float(value))
        except ValueError:
            freq.append(1.0)
    return freq


def parse_log(log_path, kind_filter, snapshot):
    gpu_order = []
    expert_max = -1
    counts = {}
    selected_owner = None
    for line in log_path.open(errors="ignore"):
        m = PLACEMENT_RE.search(line)
        if not m:
            continue
        kind = m.group("kind")
        if kind_filter != "all" and kind != kind_filter:
            continue
        owner = m.group("owner")
        if snapshot == "first":
            if selected_owner is None:
                selected_owner = owner
            elif owner != selected_owner:
                continue
        elif snapshot == "latest" and owner != selected_owner:
            selected_owner = owner
            gpu_order = []
            expert_max = -1
            counts = {}
        gpu = m.group("gpu")
        if gpu not in counts:
            counts[gpu] = {}
            gpu_order.append(gpu)
        experts_raw = m.group("experts").strip()
        if not experts_raw:
            continue
        for token in experts_raw.split(","):
            token = token.strip()
            if not token:
                continue
            expert = int(token)
            expert_max = max(expert_max, expert)
            counts[gpu][expert] = counts[gpu].get(expert, 0.0) + 1.0
    return gpu_order, counts, expert_max + 1


def build_matrix(gpu_order, counts, expert_count, freq):
    import numpy as np

    count_matrix = np.zeros((len(gpu_order), expert_count), dtype=float)
    weighted_matrix = np.zeros_like(count_matrix)
    for row, gpu in enumerate(gpu_order):
        for expert, count in counts.get(gpu, {}).items():
            weight = freq[expert] if expert < len(freq) else 1.0
            count_matrix[row, expert] = count
            weighted_matrix[row, expert] = count * weight
    return count_matrix, weighted_matrix


def display_matrix(matrix, scale):
    import numpy as np

    if scale == "log1p":
        return np.log1p(matrix), "log1p(raw)"
    if scale == "sqrt":
        return np.sqrt(matrix), "sqrt(raw)"
    return matrix, "raw"


def main():
    parser = argparse.ArgumentParser(description="Plot expert placement/access heatmaps from mnCCL logs.")
    parser.add_argument("--log", default="mncc.log", help="mnCCL log path.")
    parser.add_argument(
        "--config",
        default="scratch/pipeline_policy.conf",
        help="Policy config path used to read EXPERT_ACCESS_FREQ.",
    )
    parser.add_argument("--output", default="mncc_expert_heatmap.png", help="Output PNG path.")
    parser.add_argument(
        "--kind",
        choices=["all", "prefill", "decode", "generic"],
        default="all",
        help="Filter placement kind.",
    )
    parser.add_argument(
        "--clip-percentile",
        type=float,
        default=99.0,
        help="Color upper bound percentile. Use 100 for full linear scale.",
    )
    parser.add_argument(
        "--snapshot",
        choices=["latest", "first", "all"],
        default="all",
        help="Which placement records to plot. 'all' accumulates every task placement.",
    )
    parser.add_argument(
        "--view",
        choices=["both", "placement", "weighted"],
        default="both",
        help="Plot both panels, only replica placement, or only weighted access heat.",
    )
    parser.add_argument(
        "--access-scale",
        choices=["linear", "log1p", "sqrt"],
        default="log1p",
        help="Display transform for the frequency-weighted access heat panel.",
    )
    args = parser.parse_args()

    log_path = Path(args.log)
    config_path = Path(args.config)
    freq = load_expert_freq(config_path)
    gpu_order, counts, expert_count = parse_log(log_path, args.kind, args.snapshot)
    expert_count = max(expert_count, len(freq))
    if not gpu_order or expert_count == 0:
        raise SystemExit(f"no expert placement records found in {log_path}")

    import matplotlib.pyplot as plt

    count_matrix, weighted_matrix = build_matrix(gpu_order, counts, expert_count, freq)
    weighted_display, weighted_scale_label = display_matrix(weighted_matrix, args.access_scale)

    images = []
    if args.view in ("both", "placement"):
        images.append((count_matrix, "Expert Replica Count per GPU", "replicas"))
    if args.view in ("both", "weighted"):
        images.append(
            (
                weighted_display,
                f"Frequency-Weighted Expert Access Heat ({weighted_scale_label})",
                weighted_scale_label,
            )
        )

    fig_height = max(4.5, len(gpu_order) * 0.16) * len(images)
    fig, axes = plt.subplots(
        len(images),
        1,
        figsize=(14, fig_height),
        sharex=True,
        squeeze=False,
    )
    axes_flat = list(axes[:, 0])
    for ax, (matrix, title, colorbar_label) in zip(axes_flat, images):
        vmax = None
        positive = matrix[matrix > 0]
        if positive.size and args.clip_percentile < 100.0:
            import numpy as np

            vmax = float(np.percentile(positive, args.clip_percentile))
            if vmax <= 0.0:
                vmax = None
        im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="viridis", vmax=vmax)
        ax.set_title(title)
        ax.set_ylabel("GPU")
        tick_step = max(1, len(gpu_order) // 32)
        ticks = list(range(0, len(gpu_order), tick_step))
        ax.set_yticks(ticks)
        ax.set_yticklabels([gpu_order[i] for i in ticks], fontsize=7)
        cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
        cbar.set_label(colorbar_label)

    axes_flat[-1].set_xlabel("Expert ID")
    axes_flat[-1].set_xticks(range(expert_count))
    axes_flat[-1].set_xticklabels([str(i) for i in range(expert_count)], rotation=90, fontsize=7)
    fig.suptitle(
        f"Expert Placement Heatmap kind={args.kind} snapshot={args.snapshot} view={args.view}"
    )
    fig.tight_layout()
    fig.savefig(args.output, dpi=160)


if __name__ == "__main__":
    main()
