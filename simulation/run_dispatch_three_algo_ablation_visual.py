#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import run_dispatch_full_compare_visual as full


ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "results" / "hw_three_algo_ablation_100x"


PLACEMENT_ROUTE_BEST = {
    "EXPERT_PLACEMENT_POLICY": "probe_balanced_domain_spread",
    "SAME_RANK_ROUTE_POLICY": "probe_rtt_delta",
    "PLACEMENT_ACCESS_MODE": "posterior",
    "ASYNC_ROUTE_PROBE_ENABLE": "1",
    "ASYNC_ROUTE_LIVE_ROUTE": "1",
    "TRACE_USE_DEVICE_PLACEMENT": "0",
    **full.COMMON_PROBE_WEIGHTS,
}

PLACEMENT_ROUTE_SCORE_V2 = {
    "EXPERT_PLACEMENT_POLICY": "probe_balanced_domain_spread",
    "SAME_RANK_ROUTE_POLICY": "probe_rtt_delta",
    "PLACEMENT_ACCESS_MODE": "posterior",
    "ASYNC_ROUTE_PROBE_ENABLE": "1",
    "ASYNC_ROUTE_LIVE_ROUTE": "1",
    "TRACE_USE_DEVICE_PLACEMENT": "0",
    **full.SCORE_V2_PRIORITY_WEIGHTS,
}


PLD_SRPT_BEST = {
    "LOCAL_FLOW_SCHEDULE_POLICY": "pld_srpt",
    "LOCAL_FLOW_PREEMPTIVE_CHUNK_BYTES": "262144",
    "LOCAL_FLOW_PLD_SRPT_V_NS": "100000",
    "LOCAL_FLOW_PLD_SRPT_T0_NS": "100000",
    "LOCAL_FLOW_PLD_SRPT_D0_NS": "0",
    "LOCAL_FLOW_PLD_SRPT_KAPPA": "2.0",
    "LOCAL_FLOW_PLD_SRPT_BETA": "0.05",
    "LOCAL_FLOW_PLD_SRPT_PREEMPT_OVERHEAD_NS": "0",
}


def merged(*settings: dict[str, str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for setting in settings:
        out.update(setting)
    return out


THREE_ALGO_METHODS: dict[str, dict[str, str]] = {
    "baseline_uniform": dict(full.METHODS["baseline_uniform"]),
    "baseline_trace": dict(full.METHODS["baseline_trace"]),
    "all_three": merged(PLACEMENT_ROUTE_BEST, PLD_SRPT_BEST),
    "all_three_score_v2": merged(PLACEMENT_ROUTE_SCORE_V2, PLD_SRPT_BEST),
    "no_expert_placement": merged(
        PLACEMENT_ROUTE_BEST,
        PLD_SRPT_BEST,
        {
            "EXPERT_PLACEMENT_POLICY": "default",
        },
    ),
    "no_expert_routing": merged(
        PLACEMENT_ROUTE_BEST,
        PLD_SRPT_BEST,
        {
            "SAME_RANK_ROUTE_POLICY": "default",
            "ASYNC_ROUTE_PROBE_ENABLE": "0",
            "ASYNC_ROUTE_LIVE_ROUTE": "0",
        },
    ),
    "no_local_pld_srpt": merged(
        PLACEMENT_ROUTE_BEST,
        {
            "LOCAL_FLOW_SCHEDULE_POLICY": "default",
            "LOCAL_FLOW_PREEMPTIVE_CHUNK_BYTES": "0",
        },
    ),
    "no_expert_placement_score_v2": merged(
        PLACEMENT_ROUTE_SCORE_V2,
        PLD_SRPT_BEST,
        {
            "EXPERT_PLACEMENT_POLICY": "default",
        },
    ),
    "no_expert_routing_score_v2": merged(
        PLACEMENT_ROUTE_SCORE_V2,
        PLD_SRPT_BEST,
        {
            "SAME_RANK_ROUTE_POLICY": "default",
            "ASYNC_ROUTE_PROBE_ENABLE": "0",
            "ASYNC_ROUTE_LIVE_ROUTE": "0",
        },
    ),
    "no_local_pld_srpt_score_v2": merged(
        PLACEMENT_ROUTE_SCORE_V2,
        {
            "LOCAL_FLOW_SCHEDULE_POLICY": "default",
            "LOCAL_FLOW_PREEMPTIVE_CHUNK_BYTES": "0",
        },
    ),
}


DEFAULT_METHODS = [
    "baseline_uniform",
    "baseline_trace",
    "all_three",
    "no_expert_placement",
    "no_expert_routing",
    "no_local_pld_srpt",
]


METHOD_NOTES = {
    "baseline_uniform": "Trace device placement + default routing + uniform trace targets.",
    "baseline_trace": "Trace device placement + default routing + original trace targets.",
    "all_three": "All three proposed algorithms enabled: optimized expert placement, async probe routing, and PLD-SRPT local scheduling.",
    "all_three_score_v2": "All three proposed algorithms enabled with score-v2 priority placement weights.",
    "no_expert_placement": "Ablates algorithm 1; uses default expert placement while keeping optimized routing and PLD-SRPT.",
    "no_expert_routing": "Ablates algorithm 2; uses default local-first/random routing while keeping optimized placement and PLD-SRPT.",
    "no_local_pld_srpt": "Ablates algorithm 3; uses default local queue scheduling while keeping optimized placement and routing.",
    "no_expert_placement_score_v2": "Score-v2 ablation of algorithm 1; uses default expert placement while keeping optimized routing and PLD-SRPT.",
    "no_expert_routing_score_v2": "Score-v2 ablation of algorithm 2; uses default local-first/random routing while keeping optimized placement and PLD-SRPT.",
    "no_local_pld_srpt_score_v2": "Score-v2 ablation of algorithm 3; uses default local queue scheduling while keeping optimized placement and routing.",
}


def parse_policy(path: Path) -> dict[str, str]:
    policy: dict[str, str] = {}
    if not path.exists():
        return policy
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts:
                policy[parts[0]] = " ".join(parts[1:])
    return policy


def policy_for_method(args: Any, out_dir: Path, method: str, settings: dict[str, str]) -> dict[str, str]:
    from_file = parse_policy(out_dir / method / "policy.conf")
    if from_file:
        return from_file
    return full.settings_for_method(args, method, settings)


def policy_get(policy: dict[str, str], key: str, fallback: Any = "") -> str:
    value = policy.get(key)
    if value is None or value == "":
        return str(fallback)
    return value


def row_float(row: dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def fmt_us(ns_value: Any) -> str:
    try:
        return f"{float(ns_value) / 1000.0:.3f}"
    except (TypeError, ValueError):
        return "0.000"


def reference_method(rows: list[dict[str, Any]]) -> str:
    available = {str(row.get("method", "")) for row in rows}
    for candidate in ("all_three_score_v2", "all_three"):
        if candidate in available:
            return candidate
    return ""


def reference_tpot_ns(rows: list[dict[str, Any]]) -> float:
    reference = reference_method(rows)
    for row in rows:
        if row.get("method") == reference:
            value = row_float(row, "average_TPOT_ns")
            if value > 0.0:
                return value
    positives = [row_float(row, "average_TPOT_ns") for row in rows]
    positives = [value for value in positives if value > 0.0]
    return min(positives) if positives else 0.0


def plot_three_algo_ablation(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    labels = [str(row["method"]) for row in rows]
    tpot_us = [row_float(row, "average_TPOT_ns") / 1000.0 for row in rows]
    base_ns = reference_tpot_ns(rows)
    reference = reference_method(rows)
    colors = ["#2ca25f" if label == reference else "#f28e2b" for label in labels]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(9, len(labels) * 1.9), 5.2))
    bars = ax.bar(x, tpot_us, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("avg TPOT (us)")
    ax.set_title("Three-Algorithm Ablation")
    ax.grid(True, axis="y", alpha=0.25)
    for bar, row in zip(bars, rows):
        tpot_ns = row_float(row, "average_TPOT_ns")
        if base_ns > 0.0 and tpot_ns > 0.0:
            delta = (tpot_ns / base_ns - 1.0) * 100.0
            label = "ref" if row.get("method") == reference else f"{delta:+.1f}%"
        else:
            label = "n/a"
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            label,
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(fig_dir / "three_algo_ablation_tpot.png", dpi=170)
    plt.close(fig)


def selected_methods(args: Any) -> dict[str, dict[str, str]]:
    if args.methods == ["all"]:
        return THREE_ALGO_METHODS
    unknown = [name for name in args.methods if name not in THREE_ALGO_METHODS]
    if unknown:
        valid = ", ".join(THREE_ALGO_METHODS)
        raise SystemExit(f"unknown methods: {', '.join(unknown)}; valid: {valid}")
    return {name: THREE_ALGO_METHODS[name] for name in args.methods}


def parse_args() -> Any:
    full.METHODS = THREE_ALGO_METHODS
    full.DEFAULT_COMPARE_METHODS = DEFAULT_METHODS
    full.DEFAULT_OUT = DEFAULT_OUT
    full.PARSER_DESCRIPTION = (
        "Run ablation over the three proposed algorithms: expert placement, expert routing, and local PLD-SRPT scheduling."
    )
    args = full.parse_args()
    if (
        "--trace-access-divisor" not in sys.argv
        and "--trace-access-100x" not in sys.argv
    ):
        args.trace_access_divisor = 100
    return args


def write_readme(
    out_dir: Path,
    args: Any,
    rows: list[dict[str, Any]],
    methods: dict[str, dict[str, str]],
) -> None:
    first_policy = {}
    if rows:
        first_policy = parse_policy(out_dir / str(rows[0]["method"]) / "policy.conf")
    trace_divisor = policy_get(
        first_policy,
        "TRACE_ACCESS_DIVISOR",
        100 if args.trace_access_100x else max(1, args.trace_access_divisor),
    )
    base_ns = reference_tpot_ns(rows)
    reference = reference_method(rows)
    reference_label = reference or "lowest positive TPOT"

    method_lines = []
    for method, settings in methods.items():
        policy = policy_for_method(args, out_dir, method, settings)
        method_lines.append(
            "| {method} | {note} | {placement} | {route} | {local} | {chunk} | {access} | {async_probe} | {live_route} |".format(
                method=method,
                note=METHOD_NOTES.get(method, ""),
                placement=policy_get(policy, "EXPERT_PLACEMENT_POLICY", "default"),
                route=policy_get(policy, "SAME_RANK_ROUTE_POLICY", "default"),
                local=policy_get(policy, "LOCAL_FLOW_SCHEDULE_POLICY", "default"),
                chunk=policy_get(policy, "LOCAL_FLOW_PREEMPTIVE_CHUNK_BYTES", "0"),
                access=policy_get(policy, "PLACEMENT_ACCESS_MODE", "initial"),
                async_probe=policy_get(policy, "ASYNC_ROUTE_PROBE_ENABLE", "0"),
                live_route=policy_get(policy, "ASYNC_ROUTE_LIVE_ROUTE", "0"),
            )
        )

    summary_lines = []
    for row in rows:
        tpot_ns = row_float(row, "average_TPOT_ns")
        if base_ns > 0.0 and tpot_ns > 0.0:
            delta_value = (tpot_ns / base_ns - 1.0) * 100.0
            delta = "ref" if row.get("method") == reference else f"{delta_value:+.2f}%"
        else:
            delta = "n/a"
        summary_lines.append(
            "| {method} | {status} | {completed} | {latency_us} | {tpot_us} | {delta} | {trace_done} | {remaining} | {qavg} | {qp95} |".format(
                method=row.get("method", ""),
                status=row.get("status", ""),
                completed=row.get("completed_tasks", 0),
                latency_us=fmt_us(row.get("average_latency_ns", 0)),
                tpot_us=fmt_us(row.get("average_TPOT_ns", 0)),
                delta=delta,
                trace_done=row.get("trace_completed", 0),
                remaining=row.get("trace_remaining_accesses", 0),
                qavg=row.get("queued_local_flows_avg", 0),
                qp95=row.get("queued_local_flows_p95", 0),
            )
        )

    text = f"""# Three-Algorithm Dispatch Ablation

Generated by `simulation/run_dispatch_three_algo_ablation_visual.py`.

Reference method: `{reference_label}`.

## Workload

- network config: `{args.network_config}`
- trace dispatch: `{args.trace_dispatch}`
- trace layer id: `{args.trace_layer_id}`
- trace access divisor: `{trace_divisor}`
- dispatch need: `{policy_get(first_policy, "DISPATCH_NEED", args.dispatch_need)}`
- expert num: `{policy_get(first_policy, "EXPERT_NUM", args.expert_num)}`
- expert per GPU: `{policy_get(first_policy, "EXPERT_PER_GPU", args.expert_per_gpu)}`
- top-k: `{policy_get(first_policy, "DISPATCH_TOPK", args.topk)}`
- payload preset: `{args.payload_preset}`
- dispatch N: `{policy_get(first_policy, "DISPATCH_N", args.dispatch_n)}`
- route probe max inflight: `{policy_get(first_policy, "ROUTE_PROBE_MAX_INFLIGHT", args.route_probe_max_inflight)}`
- async route probe interval ns: `{policy_get(first_policy, "ASYNC_ROUTE_PROBE_INTERVAL_NS", args.async_route_probe_interval_ns)}`
- async route probe budget: `{policy_get(first_policy, "ASYNC_ROUTE_PROBE_BUDGET", args.async_route_probe_budget)}`
- async route probe refresh ns: `{policy_get(first_policy, "ASYNC_ROUTE_PROBE_REFRESH_NS", args.async_route_probe_refresh_ns)}`
- async route probe top-k: `{policy_get(first_policy, "ASYNC_ROUTE_PROBE_TOPK", args.async_route_probe_topk)}`

## Ablation Matrix

| method | ablated component | expert placement | route policy | local queue policy | chunk bytes | access mode | async probe | live route |
|---|---|---|---|---|---:|---|---:|---:|
{chr(10).join(method_lines)}

## Result Summary

| method | status | completed tasks | avg latency us | avg TPOT us | TPOT delta vs {reference_label} | trace completed | remaining trace accesses | avg queued flows | p95 queued flows |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(summary_lines)}

## Artifacts

- `summary.csv`: parsed metrics.
- `best_case.txt`: lowest parsed TPOT case.
- `<method>/policy.conf`: exact config for each run.
- `<method>/mncc.log`: mnCCL log.
- `<method>/mncc_flow_finish.csv`: per-flow completion records.
- `figures/three_algo_ablation_tpot.png`: TPOT delta relative to `{reference_label}`.
- `figures/latency_tpot_bar.png`: latency and TPOT comparison.
- `figures/queue_pressure_bar.png`: local queue pressure comparison.
- `figures/expert_heatmap_<method>.png`: placement/access heatmap.
- `figures/expert_placement_latest_<method>.png`: latest placement heatmap.
"""
    (out_dir / "README.md").write_text(text)


def main() -> None:
    args = parse_args()
    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = selected_methods(args)

    if args.skip_run:
        rows = full.parse_existing(out_dir, list(methods))
    else:
        rows = [
            full.run_method(args, out_dir, method, settings)
            for method, settings in methods.items()
        ]

    if not rows:
        raise SystemExit("no rows parsed; nothing to plot")

    full.write_summary(out_dir, rows)
    ordered_methods = [str(row["method"]) for row in rows]
    full.remove_obsolete_visualizations(out_dir)
    full.plot_metric_bars(out_dir, rows)
    plot_three_algo_ablation(out_dir, rows)
    if args.print_placement_detail:
        full.plot_heatmaps(
            out_dir,
            ordered_methods,
            args.heatmap_clip_percentile,
            args.placement_heatmap_snapshot,
            args.access_heat_scale,
        )
    write_readme(out_dir, args, rows, methods)
    print(f"summary: {out_dir / 'summary.csv'}")
    print(f"figures: {out_dir / 'figures'}")
    print(f"readme: {out_dir / 'README.md'}")


if __name__ == "__main__":
    main()
