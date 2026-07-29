#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_topology_ema_compare")

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "results" / "hw_topology_trace_full_topology_static_prior_score_v2_100x"
DEFAULT_TOPOLOGIES = ["topo14", "topo22", "topo41"]
BASELINE_METHOD = "baseline_trace"
FULL_METHOD = "full"
DEFAULT_ALGO_METHOD = "topology_static_prior"
DEFAULT_METHODS = [BASELINE_METHOD, FULL_METHOD, DEFAULT_ALGO_METHOD]
WEIGHT_NAMES = [
    "replica",
    "external",
    "l1_var",
    "l1_max",
    "gpu_heat_var",
    "access_latency",
]
EMA_INITIAL_BY_METHOD = {
    "multiplicative": [6.0, 8.0, 1.0, 1.5, 0.5, 7.0],
    "multiplicative_ma5": [6.0, 8.0, 1.0, 1.5, 0.5, 7.0],
    "partition_decay": [6.0, 8.0, 1.0, 1.5, 0.5, 7.0],
    "topology_static_prior": [6.0, 8.0, 1.0, 1.5, 0.5, 7.0],
    "ema": [6.0, 8.0, 1.0, 1.5, 0.5, 7.0],
    "adaptive_grad_ema_score_v2_priority_fast": [6.0, 8.0, 1.0, 1.5, 0.5, 7.0],
    "adaptive_grad_ema": [4.0, 4.0, 4.0, 4.0, 4.0, 4.0],
    "adaptive_grad_ema_bad_init": [1.0, 1.0, 7.0, 10.0, 3.0, 2.0],
    "adaptive_grad_ema_uniform_init": [4.0, 4.0, 4.0, 4.0, 4.0, 4.0],
    "adaptive_grad_ema_uniform_fast": [4.0, 4.0, 4.0, 4.0, 4.0, 4.0],
}


def row_float(row: dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def ema_initial_for_method(method: str) -> list[float]:
    return EMA_INITIAL_BY_METHOD.get(method, [4.0] * len(WEIGHT_NAMES))


def normalize_topology_if_needed(source: Path, out_dir: Path, topology: str) -> Path:
    tokens = source.read_text().split()
    if len(tokens) < 7:
        raise ValueError(f"invalid topology header: {source}")

    node_num = int(tokens[0])
    gpus_per_server = tokens[1]
    nvswitch_num = int(tokens[2])
    switch_num = int(tokens[3])
    link_num = int(tokens[4])
    gpu_type = tokens[5]
    ngpus_per_node = tokens[6]
    idx = 7
    nvswitch_ids = [int(value) for value in tokens[idx:idx + nvswitch_num]]
    idx += nvswitch_num
    switch_ids = [int(value) for value in tokens[idx:idx + switch_num]]
    idx += switch_num
    link_tokens = tokens[idx:]
    if len(link_tokens) % 5 != 0:
        raise ValueError(f"topology link token count is not divisible by 5: {source}")

    records: list[tuple[int, int, str, str, str]] = []
    max_id = max(nvswitch_ids + switch_ids + [0])
    for pos in range(0, len(link_tokens), 5):
        src = int(link_tokens[pos])
        dst = int(link_tokens[pos + 1])
        rate = link_tokens[pos + 2]
        delay = link_tokens[pos + 3]
        err = link_tokens[pos + 4]
        max_id = max(max_id, src, dst)
        records.append((src, dst, rate, delay, err))

    if max_id < node_num and len(records) == link_num:
        return source
    if nvswitch_num != 0:
        raise ValueError(
            f"topology repair currently supports nvswitch_num=0 only: {source}"
        )

    host_count = node_num - switch_num - nvswitch_num
    repaired_node_num = max(node_num, max_id + 1)
    repaired_switch_num = repaired_node_num - host_count
    if repaired_switch_num <= 0:
        raise ValueError(f"cannot infer repaired switch count for {source}")

    repaired = out_dir / "network_configs" / f"{topology}_repaired_topology.txt"
    repaired.parent.mkdir(parents=True, exist_ok=True)
    with repaired.open("w") as f:
        f.write(
            f"{repaired_node_num} {gpus_per_server} {nvswitch_num} "
            f"{repaired_switch_num} {len(records)} {gpu_type} {ngpus_per_node}\n"
        )
        f.write(" ".join(str(host_count + i) for i in range(repaired_switch_num)))
        f.write(" \n")
        for src, dst, rate, delay, err in records:
            f.write(f"{src} {dst} {rate} {delay} {err}\n")
    print(
        "[repair] {topology}: node_num {old_nodes}->{new_nodes}, "
        "switch_num {old_switches}->{new_switches}, link_num {old_links}->{new_links}, "
        "source={source}, repaired={repaired}".format(
            topology=topology,
            old_nodes=node_num,
            new_nodes=repaired_node_num,
            old_switches=switch_num,
            new_switches=repaired_switch_num,
            old_links=link_num,
            new_links=len(records),
            source=source,
            repaired=repaired,
        ),
        flush=True,
    )
    return repaired


def make_network_config(base_config: Path, out_path: Path, topology_file: Path) -> None:
    lines = base_config.read_text().splitlines()
    new_lines: list[str] = []
    replaced = False
    for line in lines:
        if line.strip().startswith("TOPOLOGY_FILE "):
            new_lines.append(f"TOPOLOGY_FILE {topology_file}")
            replaced = True
        else:
            new_lines.append(line)
    if not replaced:
        raise ValueError(f"TOPOLOGY_FILE line not found in {base_config}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(new_lines) + "\n")


def run_topology(args: argparse.Namespace, topology: str, network_config: Path) -> None:
    out_dir = args.output / topology
    cmd = [
        sys.executable,
        str(ROOT / "run_dispatch_adaptive_placement_visual.py"),
        "--methods",
        *args.methods,
        "--trace-access-divisor",
        str(args.trace_access_divisor),
        "--dispatch-need",
        str(args.dispatch_need),
        "--expert-num",
        str(args.expert_num),
        "--expert-per-gpu",
        str(args.expert_per_gpu),
        "--network-config",
        str(network_config),
        "--output",
        str(out_dir),
        "--keep-going",
    ]
    if not args.with_placement_detail:
        cmd.append("--no-placement-detail")
    if args.skip_run:
        cmd.append("--skip-run")
    print(f"[topology] {topology}: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def aggregate_summaries(out_dir: Path, topologies: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for topology in topologies:
        for row in read_csv(out_dir / topology / "summary.csv"):
            copied: dict[str, Any] = {"topology": topology}
            copied.update(row)
            rows.append(copied)
    write_csv(out_dir / "summary_all.csv", rows)
    return rows


def aggregate_weights(
    out_dir: Path,
    topologies: list[str],
    ema_method: str,
    ema_initial: list[float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    trajectory_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    for topology in topologies:
        rows = read_csv(out_dir / topology / "adaptive_weights.csv")
        for row in rows:
            copied: dict[str, Any] = {"topology": topology}
            copied.update(row)
            trajectory_rows.append(copied)
        ema_rows = [
            row for row in rows
            if row.get("method") == ema_method
        ]
        if not ema_rows:
            continue
        last = ema_rows[-1]
        final: dict[str, Any] = {
            "topology": topology,
            "method": ema_method,
            "updates": len(ema_rows),
        }
        for i, name in enumerate(WEIGHT_NAMES):
            initial = ema_initial[i]
            value = row_float(last, f"weight_{name}")
            final[f"initial_{name}"] = initial
            final[f"final_{name}"] = value
            final[f"delta_{name}"] = value - initial
        final_rows.append(final)
    write_csv(out_dir / "adaptive_weights_all.csv", trajectory_rows)
    write_csv(out_dir / "adaptive_final_weights.csv", final_rows)
    return trajectory_rows, final_rows


def plot_tpot(
    out_dir: Path,
    rows: list[dict[str, Any]],
    topologies: list[str],
    methods: list[str],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    by_key = {(row.get("topology"), row.get("method")): row for row in rows}
    x = np.arange(len(topologies))
    width = min(0.24, 0.78 / max(1, len(methods)))
    offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2.0) * width
    colors = {
        BASELINE_METHOD: "#8a8a8a",
        FULL_METHOD: "#2ca25f",
        DEFAULT_ALGO_METHOD: "#4e79a7",
        "partition_decay": "#4e79a7",
        "ema": "#4e79a7",
    }
    baseline_ns = [
        row_float(by_key.get((topology, BASELINE_METHOD), {}), "average_TPOT_ns")
        for topology in topologies
    ]

    fig, ax = plt.subplots(figsize=(10.4, 5.4))
    for idx, method in enumerate(methods):
        values_ns = [
            row_float(by_key.get((topology, method), {}), "average_TPOT_ns")
            for topology in topologies
        ]
        bars = ax.bar(
            x + offsets[idx],
            [value / 1000.0 for value in values_ns],
            width,
            label=method,
            color=colors.get(method, "#f28e2b"),
        )
        for topo_idx, bar in enumerate(bars):
            value = values_ns[topo_idx]
            base = baseline_ns[topo_idx]
            if method == BASELINE_METHOD:
                label = "baseline"
            elif base > 0.0 and value > 0.0:
                label = f"{(value / base - 1.0) * 100.0:+.2f}%"
            else:
                label = "n/a"
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                label,
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(topologies)
    ax.set_ylabel("avg TPOT (us)")
    ax.set_title("baseline_trace vs full/algorithm placement score across topologies")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "tpot_baseline_trace_full_new_by_topology.png", dpi=170)
    fig.savefig(fig_dir / "tpot_fixed_vs_ema_by_topology.png", dpi=170)
    plt.close(fig)


def plot_final_weights(
    out_dir: Path,
    final_rows: list[dict[str, Any]],
    ema_initial: list[float],
) -> None:
    if not final_rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    topologies = [str(row["topology"]) for row in final_rows]
    x = np.arange(len(WEIGHT_NAMES))
    width = min(0.22, 0.72 / max(1, len(topologies)))

    fig, ax = plt.subplots(figsize=(11, 5.6))
    offsets = (np.arange(len(topologies)) - (len(topologies) - 1) / 2.0) * width
    for idx, row in enumerate(final_rows):
        values = [row_float(row, f"final_{name}") for name in WEIGHT_NAMES]
        ax.bar(x + offsets[idx], values, width, label=str(row["topology"]))
    ax.plot(x, ema_initial, color="#333333", marker="o", linewidth=1.4, label="configured initial")
    ax.set_xticks(x)
    ax.set_xticklabels(WEIGHT_NAMES, rotation=20, ha="right")
    ax.set_ylabel("weight")
    ax.set_title("Algorithm final weights by topology")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(fig_dir / "ema_final_weights_by_topology.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5.4))
    for row in final_rows:
        deltas = [row_float(row, f"delta_{name}") for name in WEIGHT_NAMES]
        ax.plot(WEIGHT_NAMES, deltas, marker="o", linewidth=1.8, label=str(row["topology"]))
    ax.axhline(0.0, color="#444444", linewidth=1.0)
    ax.set_xticks(range(len(WEIGHT_NAMES)))
    ax.set_xticklabels(WEIGHT_NAMES, rotation=20, ha="right")
    ax.set_ylabel("final - initial")
    ax.set_title("Topology-specific weight shifts")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "ema_weight_shift_by_topology.png", dpi=170)
    plt.close(fig)


def plot_weight_convergence(out_dir: Path, rows: list[dict[str, Any]], ema_method: str) -> None:
    if not rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("method") != ema_method:
            continue
        grouped.setdefault(str(row.get("topology", "")), []).append(row)
    if not grouped:
        return

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for topology, topo_rows in sorted(grouped.items()):
        x = [int(float(row.get("index", i) or i)) for i, row in enumerate(topo_rows)]
        final = [
            row_float(topo_rows[-1], f"weight_{name}")
            for name in WEIGHT_NAMES
        ]
        final_norm = max(sum(value * value for value in final) ** 0.5, 1e-9)
        dist: list[float] = []
        for row in topo_rows:
            weights = [row_float(row, f"weight_{name}") for name in WEIGHT_NAMES]
            sq = sum((weights[i] - final[i]) * (weights[i] - final[i]) for i in range(len(WEIGHT_NAMES)))
            dist.append((sq ** 0.5) / final_norm)
        ax.plot(x, dist, marker="o", linewidth=1.8, label=topology)
    ax.set_xlabel("adaptive update index")
    ax.set_ylabel("||w_t - w_final|| / ||w_final||")
    ax.set_title("EMA convergence distance by topology")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "ema_convergence_by_topology.png", dpi=170)
    plt.close(fig)


def write_readme(
    out_dir: Path,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    final_rows: list[dict[str, Any]],
    network_configs: dict[str, Path],
) -> None:
    by_key = {(row.get("topology"), row.get("method")): row for row in rows}
    algo_initial = ema_initial_for_method(args.algo_method)
    baseline_method = BASELINE_METHOD if BASELINE_METHOD in args.methods else args.methods[0]
    full_method = FULL_METHOD if FULL_METHOD in args.methods else (
        args.methods[1] if len(args.methods) > 1 else ""
    )
    algo_method = args.algo_method
    summary_lines = []
    for topology in args.topologies:
        baseline = by_key.get((topology, baseline_method), {})
        full_row = by_key.get((topology, full_method), {})
        adaptive = by_key.get((topology, algo_method), {})
        baseline_tpot = row_float(baseline, "average_TPOT_ns")
        full_tpot = row_float(full_row, "average_TPOT_ns")
        adaptive_tpot = row_float(adaptive, "average_TPOT_ns")
        full_vs_baseline = "n/a"
        adaptive_vs_baseline = "n/a"
        adaptive_vs_full = "n/a"
        if baseline_tpot > 0.0 and full_tpot > 0.0:
            full_vs_baseline = f"{(full_tpot / baseline_tpot - 1.0) * 100.0:+.2f}%"
        if baseline_tpot > 0.0 and adaptive_tpot > 0.0:
            adaptive_vs_baseline = f"{(adaptive_tpot / baseline_tpot - 1.0) * 100.0:+.2f}%"
        if full_tpot > 0.0 and adaptive_tpot > 0.0:
            adaptive_vs_full = f"{(adaptive_tpot / full_tpot - 1.0) * 100.0:+.2f}%"
        summary_lines.append(
            "| {topology} | {baseline_status} | {baseline_tpot:.3f} | {full_status} | {full_tpot:.3f} | {full_vs_baseline} | {adaptive_status} | {adaptive_tpot:.3f} | {adaptive_vs_baseline} | {adaptive_vs_full} | {updates} |".format(
                topology=topology,
                baseline_status=baseline.get("status", ""),
                baseline_tpot=baseline_tpot / 1000.0,
                full_status=full_row.get("status", ""),
                full_tpot=full_tpot / 1000.0,
                full_vs_baseline=full_vs_baseline,
                adaptive_status=adaptive.get("status", ""),
                adaptive_tpot=adaptive_tpot / 1000.0,
                adaptive_vs_baseline=adaptive_vs_baseline,
                adaptive_vs_full=adaptive_vs_full,
                updates=next(
                    (row.get("updates", 0) for row in final_rows if row.get("topology") == topology),
                    0,
                ),
            )
        )

    weight_lines = []
    for row in final_rows:
        values = ", ".join(f"{row_float(row, f'final_{name}'):.4f}" for name in WEIGHT_NAMES)
        deltas = ", ".join(f"{row_float(row, f'delta_{name}'):+.4f}" for name in WEIGHT_NAMES)
        weight_lines.append(
            f"| {row.get('topology', '')} | {row.get('updates', 0)} | [{values}] | [{deltas}] |"
        )

    config_lines = [
        f"- `{topology}`: `{path}`"
        for topology, path in network_configs.items()
    ]

    text = f"""# Topology Placement Score Comparison

Generated by `simulation/run_dispatch_topology_ema_compare.py`.

This experiment compares:

- `{baseline_method}`: trace-device placement baseline with default routing/local queue.
- `{full_method}`: score-v2 full policy, domain-spread placement + probe routing + PLD-SRPT.
- `{algo_method}`: algorithm placement score with configured base weights `[{",".join(f"{value:g}" for value in algo_initial)}]`, plus probe routing + PLD-SRPT.

The `full` and `{algo_method}` methods share the same routing and local queue policy; only the expert-placement score differs.

## Workload

- topologies: `{", ".join(args.topologies)}`
- methods: `{", ".join(args.methods)}`
- trace access divisor: `{args.trace_access_divisor}`
- algorithm method: `{algo_method}`
- placement detail: `{args.with_placement_detail}`
- base network config: `{args.base_network_config}`

Generated network configs:

{chr(10).join(config_lines)}

## Summary

| topology | baseline status | baseline TPOT us | full status | full TPOT us | full vs baseline | algorithm status | algorithm TPOT us | algorithm vs baseline | algorithm vs full | adaptive updates |
|---|---|---:|---|---:|---:|---|---:|---:|---:|---:|
{chr(10).join(summary_lines)}

## Adaptive Weights

Weight order: `{", ".join(WEIGHT_NAMES)}`.

| topology | updates | final weights | final - initial |
|---|---:|---|---|
{chr(10).join(weight_lines)}

## Artifacts

- `summary_all.csv`: cross-topology parsed metrics.
- `adaptive_weights_all.csv`: adaptive weight trajectory with topology column when the selected algorithm logs one.
- `adaptive_final_weights.csv`: final adaptive weight vector per topology when available.
- `figures/tpot_baseline_trace_full_new_by_topology.png`: baseline_trace vs full/new placement score TPOT.
- `figures/tpot_fixed_vs_ema_by_topology.png`: compatibility copy of the TPOT comparison figure.
- `figures/ema_final_weights_by_topology.png`: final algorithm weights when available.
- `figures/ema_weight_shift_by_topology.png`: topology-specific movement from the configured base weights when available.
- `figures/ema_convergence_by_topology.png`: normalized distance to final weights when adaptive updates are available.
- `<topology>/`: raw per-topology experiment outputs from `run_dispatch_adaptive_placement_visual.py`.
"""
    (out_dir / "README.md").write_text(text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline_trace, full score-v2, and a new placement score across selected HW topologies."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT, help="Result directory.")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help=(
            "Child methods to run. Default is "
            f"{' '.join(DEFAULT_METHODS)}."
        ),
    )
    parser.add_argument(
        "--topologies",
        nargs="+",
        default=DEFAULT_TOPOLOGIES,
        help="Topology basenames under simulation/examples/HW, without .txt.",
    )
    parser.add_argument(
        "--base-network-config",
        type=Path,
        default=ROOT / "examples" / "HW" / "test.sh",
        help="Base network config template.",
    )
    parser.add_argument(
        "--trace-access-divisor",
        type=int,
        default=100,
        help="Pass-through trace access divisor for each topology experiment.",
    )
    parser.add_argument(
        "--dispatch-need",
        type=int,
        default=256,
        help="Pass-through DISPATCH_NEED for each topology experiment.",
    )
    parser.add_argument(
        "--expert-num",
        type=int,
        default=256,
        help="Pass-through EXPERT_NUM for each topology experiment.",
    )
    parser.add_argument(
        "--expert-per-gpu",
        type=int,
        default=9,
        help="Pass-through EXPERT_PER_GPU for each topology experiment.",
    )
    parser.add_argument(
        "--algo-method",
        default=DEFAULT_ALGO_METHOD,
        choices=sorted(EMA_INITIAL_BY_METHOD),
        help="Third placement method to compare against full.",
    )
    parser.add_argument(
        "--ema-method",
        dest="algo_method",
        choices=sorted(EMA_INITIAL_BY_METHOD),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--skip-run", action="store_true", help="Only re-aggregate existing runs.")
    parser.add_argument(
        "--with-placement-detail",
        action="store_true",
        help="Keep per-GPU expert placement detail and heatmap generation for child runs.",
    )
    args = parser.parse_args()
    if args.methods is None:
        args.methods = [BASELINE_METHOD, FULL_METHOD, args.algo_method]
    return args


def main() -> None:
    args = parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)

    network_configs: dict[str, Path] = {}
    for topology in args.topologies:
        topo_path = ROOT / "examples" / "HW" / f"{topology}.txt"
        if not topo_path.exists():
            raise FileNotFoundError(f"topology file not found: {topo_path}")
        topology_file = normalize_topology_if_needed(topo_path, args.output, topology)
        cfg_path = args.output / "network_configs" / f"{topology}.conf"
        make_network_config(args.base_network_config.resolve(), cfg_path, topology_file.resolve())
        network_configs[topology] = cfg_path

    for topology in args.topologies:
        run_topology(args, topology, network_configs[topology])

    rows = aggregate_summaries(args.output, args.topologies)
    ema_initial = ema_initial_for_method(args.algo_method)
    trajectory_rows, final_rows = aggregate_weights(
        args.output,
        args.topologies,
        args.algo_method,
        ema_initial,
    )
    plot_tpot(args.output, rows, args.topologies, args.methods)
    plot_final_weights(args.output, final_rows, ema_initial)
    plot_weight_convergence(args.output, trajectory_rows, args.algo_method)
    write_readme(args.output, args, rows, final_rows, network_configs)

    print(f"summary: {args.output / 'summary_all.csv'}")
    print(f"weights: {args.output / 'adaptive_final_weights.csv'}")
    print(f"figures: {args.output / 'figures'}")
    print(f"readme: {args.output / 'README.md'}")


if __name__ == "__main__":
    main()
