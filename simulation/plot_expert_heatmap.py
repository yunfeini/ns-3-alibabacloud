#!/usr/bin/env python3
import argparse
import re
from pathlib import Path


PLACEMENT_RE = re.compile(
    r"expert placement gpu placement_owner=(?P<owner>\d+) "
    r"kind=(?P<kind>\w+) gpu=(?P<gpu>\d+:\d+) experts=\[(?P<experts>[^\]]*)\]"
)


def load_expert_freq(config_path):
    freq = []
    if not config_path.exists():
        return freq
    with config_path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts and parts[0] == "EXPERT_ACCESS_FREQ":
                for value in parts[1:]:
                    try:
                        freq.append(float(value))
                    except ValueError:
                        freq.append(1.0)
    return freq


def parse_log(log_path, kind_filter):
    gpu_order = []
    expert_max = -1
    counts = {}
    for line in log_path.open(errors="ignore"):
        m = PLACEMENT_RE.search(line)
        if not m:
            continue
        kind = m.group("kind")
        if kind_filter != "all" and kind != kind_filter:
            continue
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
    args = parser.parse_args()

    log_path = Path(args.log)
    config_path = Path(args.config)
    freq = load_expert_freq(config_path)
    gpu_order, counts, expert_count = parse_log(log_path, args.kind)
    if not gpu_order or expert_count == 0:
        raise SystemExit(f"no expert placement records found in {log_path}")

    import matplotlib.pyplot as plt

    count_matrix, weighted_matrix = build_matrix(gpu_order, counts, expert_count, freq)

    fig, axes = plt.subplots(2, 1, figsize=(14, max(7, len(gpu_order) * 0.16)), sharex=True)
    images = [
        (axes[0], count_matrix, "Expert Replica Count per GPU"),
        (axes[1], weighted_matrix, "Frequency-Weighted Expert Access Heat"),
    ]
    for ax, matrix, title in images:
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
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)

    axes[-1].set_xlabel("Expert ID")
    axes[-1].set_xticks(range(expert_count))
    axes[-1].set_xticklabels([str(i) for i in range(expert_count)], rotation=90, fontsize=7)
    fig.suptitle(f"Expert Placement Heatmap kind={args.kind}")
    fig.tight_layout()
    fig.savefig(args.output, dpi=160)


if __name__ == "__main__":
    main()
