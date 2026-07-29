#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


LOADS = ["light", "medium", "heavy"]
METHODS = [
    "baseline",
    "probe_balanced",
    "probe_balanced_no_route",
]
METHOD_LABELS = {
    "baseline": "baseline",
    "probe_balanced": "probe balanced",
    "probe_balanced_no_route": "probe balanced no route",
}
COLORS = {
    "baseline": "#4c78a8",
    "probe_balanced": "#b5cf6b",
    "probe_balanced_no_route": "#eeca3b",
}


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def read_summary(path):
    rows = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows[(row["load"], row["method"])] = row
    return rows


def available_methods(rows):
    return [method for method in METHODS if any(key[1] == method for key in rows)]


def available_loads(rows):
    return [load for load in LOADS if any(key[0] == load for key in rows)]


def read_timeseries(path):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return rows


def add_bar_labels(ax, bars, fmt="{:.1f}", rotation=0):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            fmt.format(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=rotation,
        )


def plot_latency(rows, out_dir):
    import matplotlib.pyplot as plt
    import numpy as np

    methods = available_methods(rows)
    loads = available_loads(rows)
    x = np.arange(len(loads))
    width = min(0.28, 0.8 / max(1, len(methods)))
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    center = (len(methods) - 1) / 2.0
    for idx, method in enumerate(methods):
        offset = (idx - center) * width
        ttft_ms = [to_float(rows[(load, method)]["average_TTFT_ns"]) / 1e6 for load in loads]
        tpot_us = [to_float(rows[(load, method)]["average_TPOT_ns"]) / 1e3 for load in loads]
        bars = axes[0].bar(x + offset, ttft_ms, width, label=METHOD_LABELS[method], color=COLORS[method])
        add_bar_labels(axes[0], bars)
        bars = axes[1].bar(x + offset, tpot_us, width, label=METHOD_LABELS[method], color=COLORS[method])
        add_bar_labels(axes[1], bars)

    axes[0].set_ylabel("TTFT (ms)")
    axes[0].set_title("End-to-end TTFT")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[1].set_ylabel("TPOT (us/token)")
    axes[1].set_title("Decode TPOT")
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(loads)
    axes[0].legend(ncols=3, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "latency_comparison.png", dpi=170)


def plot_relative(rows, out_dir):
    import matplotlib.pyplot as plt
    import numpy as np

    methods = [method for method in available_methods(rows) if method != "baseline"]
    if not methods or "baseline" not in available_methods(rows):
        return
    loads = available_loads(rows)
    x = np.arange(len(loads))
    width = min(0.32, 0.8 / max(1, len(methods)))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    center = (len(methods) - 1) / 2.0
    for idx, method in enumerate(methods):
        values = []
        for load in loads:
            base = to_float(rows[(load, "baseline")]["average_TTFT_ns"])
            value = to_float(rows[(load, method)]["average_TTFT_ns"])
            values.append((value / base - 1.0) * 100.0 if base > 0 else 0.0)
        bars = ax.bar(x + (idx - center) * width, values, width, label=METHOD_LABELS[method], color=COLORS[method])
        add_bar_labels(ax, bars, fmt="{:+.1f}")
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(loads)
    ax.set_ylabel("TTFT change vs baseline (%)")
    ax.set_title("Relative TTFT")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "relative_ttft_vs_baseline.png", dpi=170)


def plot_queue_fragmentation(rows, out_dir):
    import matplotlib.pyplot as plt
    import numpy as np

    methods = available_methods(rows)
    loads = available_loads(rows)
    metrics = [
        ("queued_local_flows_avg", "Avg queued local flows"),
        ("queued_local_flows_p95", "P95 queued local flows"),
        ("cn_fragmentation_avg", "CN fragmentation"),
        ("gpu_mem_utilization_avg", "GPU memory utilization"),
    ]
    x = np.arange(len(loads))
    width = min(0.28, 0.8 / max(1, len(methods)))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, (field, title) in zip(axes.ravel(), metrics):
        center = (len(methods) - 1) / 2.0
        for idx, method in enumerate(methods):
            values = [to_float(rows[(load, method)].get(field)) for load in loads]
            ax.bar(x + (idx - center) * width, values, width, label=METHOD_LABELS[method], color=COLORS[method])
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(loads)
        ax.grid(True, axis="y", alpha=0.3)
    axes[0][0].legend(ncols=3, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "queue_fragmentation_comparison.png", dpi=170)


def plot_placement_quality(rows, out_dir):
    import matplotlib.pyplot as plt
    import numpy as np

    methods = [
        method for method in ["probe_balanced", "probe_balanced_no_route"]
        if any(key[1] == method for key in rows)
    ]
    if not methods:
        return
    loads = available_loads(rows)
    x = np.arange(len(loads))
    width = min(0.32, 0.8 / max(1, len(methods)))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, prefix in zip(axes, ["prefill", "decode"]):
        center = (len(methods) - 1) / 2.0
        for idx, method in enumerate(methods):
            values = [to_float(rows[(load, method)].get(f"{prefix}_l1_max_load_avg")) for load in loads]
            bars = ax.bar(x + (idx - center) * width, values, width, label=METHOD_LABELS[method], color=COLORS[method])
            add_bar_labels(ax, bars)
        ax.set_title(f"{prefix} L1 max access load")
        ax.set_xticks(x)
        ax.set_xticklabels(loads)
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("weighted access load")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "placement_l1_load_comparison.png", dpi=170)


def plot_heavy_timeseries(base_dir, out_dir):
    import matplotlib.pyplot as plt

    methods = [method for method in METHODS if (base_dir / "heavy" / method / "mncc_cluster_timeseries.csv").exists()]
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fields = [
        ("queued_local_flows", "Queued local flows"),
        ("gpu_mem_utilization", "GPU memory utilization"),
        ("cn_fragmentation", "CN fragmentation"),
    ]
    for method in methods:
        csv_path = base_dir / "heavy" / method / "mncc_cluster_timeseries.csv"
        data = read_timeseries(csv_path)
        t_ms = [to_float(row["time_ns"]) / 1e6 for row in data]
        for ax, (field, title) in zip(axes, fields):
            ax.plot(t_ms, [to_float(row.get(field)) for row in data], label=METHOD_LABELS[method], color=COLORS[method], linewidth=1.4)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("simulation time (ms)")
    axes[0].legend(ncols=3, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "heavy_timeseries.png", dpi=170)


def main():
    parser = argparse.ArgumentParser(description="Plot spaced-long domain placement comparison results.")
    parser.add_argument("--input-dir", default="simulation/results/domain_placement_spaced_long_compare")
    parser.add_argument("--output-dir", default="simulation/results/domain_placement_spaced_long_figures")
    args = parser.parse_args()

    base_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_summary(base_dir / "summary.csv")

    plot_latency(rows, out_dir)
    plot_relative(rows, out_dir)
    plot_queue_fragmentation(rows, out_dir)
    plot_placement_quality(rows, out_dir)
    plot_heavy_timeseries(base_dir, out_dir)
    print(f"wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
