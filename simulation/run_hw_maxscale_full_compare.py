#!/usr/bin/env python3
"""Run the largest stable HW full-comparison experiment and plot results.

The default profile is intentionally the largest configuration that completed
cleanly in the current simulator session:

  1024-host topology, PD total 128, EP 64, 128 inference tasks.

The script keeps every run's mncc.log, flow-finish CSV, cluster time-series CSV,
network config, and policy config under the output directory. It also generates
expert placement heatmaps and cross-method time-series figures.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUNNER = ROOT / "run_domain_placement_full_load_compare.py"
HEATMAP = ROOT / "plot_expert_heatmap.py"

DEFAULT_METHODS = ["baseline", "probe_balanced", "probe_cost", "full"]
ALL_METHODS = [
    "baseline",
    "spread_base",
    "hot_surplus",
    "topology_diverse",
    "probe_balanced",
    "probe_cost",
    "full",
    "probe_balanced_no_route",
]

PROFILES = {
    "max-stable": {
        "num_tasks": 64,
        "pd_total": 1024,
        "expert_num":64,
        "expert_per_gpu": 1,
        "prefill_min": 32,
        "prefill_max": 128,
        "decode_min": 4,
        "decode_max": 32,
        "submit_interval": "0.001",
        "stop_time": "0.05",
        "monitor_interval_ns": 50000,
        "print_submissions": "0",
        "print_expert_placement_detail": "1",
    },
    "stress-512": {
        "num_tasks": 512,
        "pd_total": 128,
        "expert_num": 64,
        "expert_per_gpu": 1,
        "prefill_min": 8,
        "prefill_max": 16,
        "decode_min": 2,
        "decode_max": 4,
        "submit_interval": "0.00002",
        "stop_time": "0.12",
        "monitor_interval_ns": 50000,
        "print_submissions": "0",
        "print_expert_placement_detail": "0",
    },
    "smoke": {
        "num_tasks": 1,
        "pd_total": 16,
        "expert_num": 8,
        "expert_per_gpu": 2,
        "prefill_min": 8,
        "prefill_max": 8,
        "decode_min": 1,
        "decode_max": 1,
        "submit_interval": "0.001",
        "stop_time": "0.01",
        "monitor_interval_ns": 50000,
        "print_submissions": "0",
        "print_expert_placement_detail": "1",
    },
}


def to_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def read_summary(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def run_command(cmd: list[str], *, cwd: Path = ROOT) -> None:
    print("[cmd] " + " ".join(str(part) for part in cmd), flush=True)
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = env.get("MPLCONFIGDIR") or "/tmp/mplconfig_hw_maxscale"
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def resolved_methods(raw: list[str] | None) -> list[str]:
    if not raw:
        return list(DEFAULT_METHODS)
    if len(raw) == 1 and raw[0] == "all":
        return list(ALL_METHODS)
    return raw


def profile_with_overrides(args: argparse.Namespace) -> dict[str, object]:
    cfg = dict(PROFILES[args.profile])
    for key in [
        "num_tasks",
        "pd_total",
        "expert_num",
        "expert_per_gpu",
        "prefill_min",
        "prefill_max",
        "decode_min",
        "decode_max",
        "submit_interval",
        "stop_time",
        "monitor_interval_ns",
        "print_submissions",
        "print_expert_placement_detail",
    ]:
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
    return cfg


def run_experiment(args: argparse.Namespace, cfg: dict[str, object], methods: list[str]) -> None:
    cmd = [
        sys.executable,
        str(RUNNER),
        "--loads",
        *args.loads,
        "--methods",
        *methods,
        "--num-tasks",
        str(cfg["num_tasks"]),
        "--prefill-min",
        str(cfg["prefill_min"]),
        "--prefill-max",
        str(cfg["prefill_max"]),
        "--decode-min",
        str(cfg["decode_min"]),
        "--decode-max",
        str(cfg["decode_max"]),
        "--submit-interval",
        str(cfg["submit_interval"]),
        "--stop-time",
        str(cfg["stop_time"]),
        "--monitor-interval-ns",
        str(cfg["monitor_interval_ns"]),
        "--pd-ratio",
        args.pd_ratio,
        "--pd-total",
        str(cfg["pd_total"]),
        "--expert-num",
        str(cfg["expert_num"]),
        "--expert-per-gpu",
        str(cfg["expert_per_gpu"]),
        "--expert-access-prior",
        str(args.expert_access_prior),
        "--print-submissions",
        str(cfg["print_submissions"]),
        "--print-expert-placement-detail",
        str(cfg["print_expert_placement_detail"]),
        "--output",
        str(args.output),
    ]
    if args.monitor:
        cmd.append("--monitor")
    run_command(cmd)


def plot_summary(rows: list[dict[str, str]], out_dir: Path) -> None:
    if not rows:
        return

    import matplotlib.pyplot as plt
    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)
    loads = []
    methods = []
    for row in rows:
        if row["load"] not in loads:
            loads.append(row["load"])
        if row["method"] not in methods:
            methods.append(row["method"])

    colors = {
        "baseline": "#64748b",
        "probe_balanced": "#3b82f6",
        "probe_cost": "#10b981",
        "full": "#f97316",
        "spread_base": "#8b5cf6",
        "hot_surplus": "#eab308",
        "topology_diverse": "#ef4444",
        "probe_balanced_no_route": "#14b8a6",
    }

    by_key = {(row["load"], row["method"]): row for row in rows}

    def values(field: str, scale: float, load: str) -> list[float]:
        return [to_float(by_key.get((load, method), {}).get(field)) / scale for method in methods]

    for load in loads:
        x = np.arange(len(methods))
        width = 0.38
        fig, ax = plt.subplots(figsize=(max(9, len(methods) * 1.2), 4.8))
        ax.bar(x - width / 2, values("average_TTFT_ns", 1e3, load), width,
               label="TTFT (us)", color="#3b82f6")
        ax.bar(x + width / 2, values("average_TPOT_ns", 1e3, load), width,
               label="TPOT (us)", color="#f97316")
        ax.set_title(f"{load}: latency comparison")
        ax.set_ylabel("Latency (us)")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=18, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"{load}_latency_bar.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(max(9, len(methods) * 1.2), 4.8))
        ax.bar(x - width / 2, values("queued_local_flows_avg", 1.0, load), width,
               label="avg queued flows", color="#14b8a6")
        ax.bar(x + width / 2, values("queued_local_flows_p95", 1.0, load), width,
               label="p95 queued flows", color="#ef4444")
        ax.set_title(f"{load}: local queue pressure")
        ax.set_ylabel("Queued local flows")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=18, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"{load}_queue_bar.png", dpi=180)
        plt.close(fig)

        fig, ax1 = plt.subplots(figsize=(max(9, len(methods) * 1.2), 4.8))
        ax2 = ax1.twinx()
        ax1.bar(x - width / 2, values("gpu_mem_utilization_avg", 1.0, load), width,
                label="avg GPU mem util", color="#8b5cf6")
        ax2.bar(x + width / 2, values("expert_fragmentation_avg", 1.0, load), width,
                label="avg expert fragmentation", color="#f59e0b")
        ax1.set_title(f"{load}: utilization and fragmentation")
        ax1.set_ylabel("GPU memory utilization")
        ax2.set_ylabel("Expert fragmentation")
        ax1.set_xticks(x)
        ax1.set_xticklabels(methods, rotation=18, ha="right")
        ax1.grid(axis="y", alpha=0.25)
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / f"{load}_util_fragmentation_bar.png", dpi=180)
        plt.close(fig)

    for field, title, ylabel in [
        ("queued_local_flows", "Queued Local Flows", "queued flows"),
        ("gpu_mem_utilization", "GPU Memory Utilization", "utilization"),
        ("expert_fragmentation", "Expert Fragmentation", "fragmentation"),
        ("active_tasks", "Active Tasks", "tasks"),
        ("active_local_gpus", "Active Local GPUs", "GPUs"),
    ]:
        for load in loads:
            fig, ax = plt.subplots(figsize=(11, 5))
            for method in methods:
                csv_path = out_dir.parent / load / method / "mncc_cluster_timeseries.csv"
                if not csv_path.exists():
                    continue
                with csv_path.open(newline="") as f:
                    data = list(csv.DictReader(f))
                if not data:
                    continue
                t_ms = [to_float(row.get("time_ns")) / 1e6 for row in data]
                y = [to_float(row.get(field)) for row in data]
                ax.plot(t_ms, y, label=method, color=colors.get(method), linewidth=1.5)
            ax.set_title(f"{load}: {title}")
            ax.set_xlabel("simulation time (ms)")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25)
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_dir / f"{load}_timeseries_{field}.png", dpi=180)
            plt.close(fig)


def plot_heatmaps(args: argparse.Namespace, methods: list[str], kinds: list[str]) -> None:
    fig_dir = args.output / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for load in args.loads:
        for method in methods:
            run_dir = args.output / load / method
            log_path = run_dir / "mncc.log"
            cfg_path = run_dir / "policy.conf"
            if not log_path.exists() or not cfg_path.exists():
                continue
            text = log_path.read_text(errors="ignore")
            if "Scheduler: expert placement gpu" not in text:
                print(f"[heatmap-skip] no per-GPU placement records in {log_path}")
                continue
            for kind in kinds:
                out = fig_dir / f"{load}_expert_heatmap_{method}_{kind}.png"
                cmd = [
                    sys.executable,
                    str(HEATMAP),
                    "--log",
                    str(log_path),
                    "--config",
                    str(cfg_path),
                    "--kind",
                    kind,
                    "--clip-percentile",
                    str(args.heatmap_clip_percentile),
                    "--output",
                    str(out),
                ]
                try:
                    run_command(cmd)
                except subprocess.CalledProcessError as exc:
                    print(f"[heatmap-failed] {load}/{method}/{kind}: {exc}")


def write_readme(args: argparse.Namespace,
                 cfg: dict[str, object],
                 methods: list[str],
                 rows: list[dict[str, str]]) -> None:
    readme = args.output / "README.md"
    lines = [
        "# HW Max-Scale Full Comparison",
        "",
        "This directory was generated by `simulation/run_hw_maxscale_full_compare.py`.",
        "",
        "## Profile",
        "",
        f"- profile: `{args.profile}`",
        f"- loads: `{', '.join(args.loads)}`",
        f"- methods: `{', '.join(methods)}`",
        f"- pd ratio: `{args.pd_ratio}`",
        f"- pd total: `{cfg['pd_total']}`",
        f"- tasks: `{cfg['num_tasks']}`",
        f"- expert_num: `{cfg['expert_num']}`",
        f"- expert_per_gpu: `{cfg['expert_per_gpu']}`",
        f"- prefill length: `{cfg['prefill_min']}-{cfg['prefill_max']}`",
        f"- decode length: `{cfg['decode_min']}-{cfg['decode_max']}`",
        f"- stop time: `{cfg['stop_time']}`",
        f"- print expert placement detail: `{cfg['print_expert_placement_detail']}`",
        "",
        "The default `max-stable` profile is the largest complete configuration",
        "that finished cleanly in the current simulator session while still",
        "preserving per-GPU expert placement logs for heatmap generation.",
        "",
        "## Result Files",
        "",
        "- `summary.csv`: per-method metrics",
        "- `<load>/<method>/mncc.log`: full mnCCL log",
        "- `<load>/<method>/mncc_cluster_timeseries.csv`: cluster monitor time-series",
        "- `<load>/<method>/mncc_flow_finish.csv`: per-flow finish records",
        "- `<load>/<method>/policy.conf`: generated policy config",
        "- `figures/`: latency, queue, utilization, time-series, and heatmap PNGs",
        "",
    ]
    if rows:
        fields = [
            "load",
            "method",
            "completed_tasks",
            "average_TTFT_ns",
            "average_TPOT_ns",
            "queued_local_flows_avg",
            "queued_local_flows_p95",
            "wall_time_s",
        ]
        lines += ["## Summary", "", "| " + " | ".join(fields) + " |",
                  "| " + " | ".join(["---"] * len(fields)) + " |"]
        for row in rows:
            values = []
            for field in fields:
                value = row.get(field, "")
                if field == "average_TTFT_ns" and value != "":
                    value = f"{to_float(value) / 1e3:.3f} us"
                elif field == "average_TPOT_ns" and value != "":
                    value = f"{to_float(value) / 1e3:.3f} us"
                values.append(str(value))
            lines.append("| " + " | ".join(values) + " |")
        lines.append("")

        baseline_by_load = {
            row["load"]: row for row in rows if row.get("method") == "baseline"
        }
        lines += ["## Relative To Baseline", "",
                  "| load | method | TTFT reduction | TPOT reduction | avg queue reduction |",
                  "| --- | --- | ---: | ---: | ---: |"]
        for row in rows:
            base = baseline_by_load.get(row["load"])
            if not base or row.get("method") == "baseline":
                continue
            base_ttft = to_float(base.get("average_TTFT_ns"))
            base_tpot = to_float(base.get("average_TPOT_ns"))
            base_queue = to_float(base.get("queued_local_flows_avg"))
            ttft = to_float(row.get("average_TTFT_ns"))
            tpot = to_float(row.get("average_TPOT_ns"))
            queue = to_float(row.get("queued_local_flows_avg"))
            lines.append(
                "| {load} | {method} | {ttft:.2f}% | {tpot:.2f}% | {queue:.2f}% |".format(
                    load=row["load"],
                    method=row["method"],
                    ttft=(base_ttft - ttft) / base_ttft * 100 if base_ttft else 0.0,
                    tpot=(base_tpot - tpot) / base_tpot * 100 if base_tpot else 0.0,
                    queue=(base_queue - queue) / base_queue * 100 if base_queue else 0.0,
                )
            )
        lines.append("")

    readme.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run HW max-scale full comparison and generate heatmaps/time-series figures."
    )
    parser.add_argument("--profile", choices=sorted(PROFILES), default="max-stable")
    parser.add_argument("--loads", nargs="+", default=["heavy"],
                        choices=["light", "medium", "heavy"])
    parser.add_argument("--methods", nargs="+",
                        help="Methods to run. Use 'all' for all available variants.")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results" / "hw_maxscale_full_compare")
    parser.add_argument("--skip-run", action="store_true",
                        help="Only parse/plot existing outputs.")
    parser.add_argument("--no-heatmaps", action="store_true",
                        help="Skip expert heatmap generation.")
    parser.add_argument("--heatmap-kinds", nargs="+",
                        default=["prefill", "decode", "all"],
                        choices=["prefill", "decode", "all", "generic"])
    parser.add_argument("--heatmap-clip-percentile", type=float, default=99.0)
    parser.add_argument("--monitor", action="store_true",
                        help="Enable detailed network monitors in generated network configs.")
    parser.add_argument("--pd-ratio", default="3:1")
    parser.add_argument("--expert-access-prior", default="1.0")

    for key in [
        "submit_interval",
        "stop_time",
        "print_submissions",
        "print_expert_placement_detail",
    ]:
        parser.add_argument(f"--{key.replace('_', '-')}")
    for key in [
        "num_tasks",
        "pd_total",
        "expert_num",
        "expert_per_gpu",
        "prefill_min",
        "prefill_max",
        "decode_min",
        "decode_max",
        "monitor_interval_ns",
    ]:
        parser.add_argument(f"--{key.replace('_', '-')}", type=int)

    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = os.environ.get("MPLCONFIGDIR") or "/tmp/mplconfig_hw_maxscale"

    methods = resolved_methods(args.methods)
    cfg = profile_with_overrides(args)

    if not args.skip_run:
        run_experiment(args, cfg, methods)

    rows = read_summary(args.output / "summary.csv")
    fig_dir = args.output / "figures"
    plot_summary(rows, fig_dir)
    if not args.no_heatmaps:
        plot_heatmaps(args, methods, args.heatmap_kinds)
    write_readme(args, cfg, methods, rows)

    print(f"[done] results: {args.output}")
    print(f"[done] figures: {fig_dir}")


if __name__ == "__main__":
    main()
