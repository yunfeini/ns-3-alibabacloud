#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import run_dispatch_full_compare_visual as full
import plot_adaptive_weights as weight_plot


ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "results" / "hw_adaptive_placement_hop12_ema_small"


ROUTE_AND_QUEUE_BEST = {
    "SAME_RANK_ROUTE_POLICY": "probe_rtt_delta",
    "ASYNC_ROUTE_PROBE_ENABLE": "1",
    "ASYNC_ROUTE_LIVE_ROUTE": "1",
    "TRACE_USE_DEVICE_PLACEMENT": "0",
    "LOCAL_FLOW_SCHEDULE_POLICY": "pld_srpt",
    "LOCAL_FLOW_PREEMPTIVE_CHUNK_BYTES": "262144",
    "LOCAL_FLOW_PLD_SRPT_V_NS": "100000",
    "LOCAL_FLOW_PLD_SRPT_T0_NS": "100000",
    "LOCAL_FLOW_PLD_SRPT_D0_NS": "0",
    "LOCAL_FLOW_PLD_SRPT_KAPPA": "2.0",
    "LOCAL_FLOW_PLD_SRPT_BETA": "0.05",
    "LOCAL_FLOW_PLD_SRPT_PREEMPT_OVERHEAD_NS": "0",
}


FIXED_INCREMENTAL = {
    "EXPERT_PLACEMENT_POLICY": "probe_balanced_incremental",
    "PLACEMENT_ACCESS_MODE": "posterior",
    "PLACEMENT_ADAPTIVE_MODE": "fixed",
    "PLACEMENT_REPLICA_BALANCE_WEIGHT": "4.0",
    "PLACEMENT_EXTERNAL_WEIGHT": "4.0",
    "PLACEMENT_L1_VARIANCE_WEIGHT": "4.0",
    "PLACEMENT_L1_MAX_WEIGHT": "4.0",
    "PLACEMENT_GPU_VARIANCE_WEIGHT": "4.0",
    "PLACEMENT_PROBE_WEIGHT": "4.0",
    "PLACEMENT_PROBE_DELTA_WEIGHT": "0.5",
    "PLACEMENT_PROBE_RTT_WEIGHT": "0.10",
}


FULL_SCORE_V2 = {
    "EXPERT_PLACEMENT_POLICY": "probe_balanced_domain_spread",
    "PLACEMENT_ACCESS_MODE": "posterior",
    "PLACEMENT_ADAPTIVE_MODE": "fixed",
}


MULTIPLICATIVE_SCORE = {
    "EXPERT_PLACEMENT_POLICY": "probe_balanced_multiplicative",
    "PLACEMENT_ACCESS_MODE": "posterior",
    "PLACEMENT_ADAPTIVE_MODE": "fixed",
}

PARTITION_DECAY_SCORE = {
    "EXPERT_PLACEMENT_POLICY": "probe_balanced_partition_decay",
    "PLACEMENT_ACCESS_MODE": "posterior",
    "PLACEMENT_ADAPTIVE_MODE": "fixed",
}


TOPOLOGY_STATIC_PRIOR_SCORE = {
    "EXPERT_PLACEMENT_POLICY": "probe_balanced_topology_static_prior",
    "PLACEMENT_ACCESS_MODE": "posterior",
    "PLACEMENT_ADAPTIVE_MODE": "fixed",
}


MOVING_AVERAGE_INVERSE_5 = {
    "PLACEMENT_ADAPTIVE_MODE": "moving_average_inverse_5",
    "PLACEMENT_ADAPTIVE_MIN_WEIGHT": "0.0",
    "PLACEMENT_ADAPTIVE_MAX_WEIGHT": "64.0",
}


ROUNDROBIN_PLACEMENT_BASELINE = {
    "EXPERT_PLACEMENT_POLICY": "default",
    "PLACEMENT_ACCESS_MODE": "posterior",
    "PLACEMENT_ADAPTIVE_MODE": "fixed",
}


SPSA_COMMON = {
    "PLACEMENT_ADAPTIVE_LEARNING_RATE": "0.05",
    "PLACEMENT_ADAPTIVE_SIGMA": "0.10",
    "PLACEMENT_ADAPTIVE_MIN_WEIGHT": "0.0",
    "PLACEMENT_ADAPTIVE_MAX_WEIGHT": "16.0",
    "PLACEMENT_ADAPTIVE_SEED": "20260714",
}

GRADIENT_EMA = {
    "PLACEMENT_ADAPTIVE_MODE": "gradient_ema",
    "PLACEMENT_ADAPTIVE_LEARNING_RATE": "0.08",
    "PLACEMENT_ADAPTIVE_OBS_ALPHA": "0.20",
    "PLACEMENT_ADAPTIVE_MOMENTUM": "0.80",
    "PLACEMENT_ADAPTIVE_WEIGHT_SUM": "24.0",
    "PLACEMENT_ADAPTIVE_GRADIENT_CLIP": "0.35",
    "PLACEMENT_ADAPTIVE_DYNAMIC_LR_GAIN": "3.0",
    "PLACEMENT_ADAPTIVE_MAX_LR_MULTIPLIER": "3.0",
    "PLACEMENT_ADAPTIVE_LR_UPDATE_THRESHOLD": "0.0",
    "PLACEMENT_ADAPTIVE_MIN_WEIGHT": "0.10",
    "PLACEMENT_ADAPTIVE_MAX_WEIGHT": "12.0",
}

GRADIENT_EMA_BAD_INIT = {
    "PLACEMENT_REPLICA_BALANCE_WEIGHT": "1.0",
    "PLACEMENT_EXTERNAL_WEIGHT": "1.0",
    "PLACEMENT_L1_VARIANCE_WEIGHT": "7.0",
    "PLACEMENT_L1_MAX_WEIGHT": "10.0",
    "PLACEMENT_GPU_VARIANCE_WEIGHT": "3.0",
    "PLACEMENT_PROBE_WEIGHT": "2.0",
}

GRADIENT_EMA_UNIFORM_INIT = {
    "PLACEMENT_REPLICA_BALANCE_WEIGHT": "4.0",
    "PLACEMENT_EXTERNAL_WEIGHT": "4.0",
    "PLACEMENT_L1_VARIANCE_WEIGHT": "4.0",
    "PLACEMENT_L1_MAX_WEIGHT": "4.0",
    "PLACEMENT_GPU_VARIANCE_WEIGHT": "4.0",
    "PLACEMENT_PROBE_WEIGHT": "4.0",
}

SCORE_V2_PRIORITY_WEIGHTS = {
    "PLACEMENT_REPLICA_BALANCE_WEIGHT": "6.0",
    "PLACEMENT_EXTERNAL_WEIGHT": "8.0",
    "PLACEMENT_L1_VARIANCE_WEIGHT": "1.0",
    "PLACEMENT_L1_MAX_WEIGHT": "1.5",
    "PLACEMENT_GPU_VARIANCE_WEIGHT": "0.5",
    "PLACEMENT_PROBE_WEIGHT": "7.0",
}

SCORE_V2_EXTERNAL_STRONG_WEIGHTS = {
    "PLACEMENT_REPLICA_BALANCE_WEIGHT": "6.0",
    "PLACEMENT_EXTERNAL_WEIGHT": "14.0",
    "PLACEMENT_L1_VARIANCE_WEIGHT": "0.5",
    "PLACEMENT_L1_MAX_WEIGHT": "1.0",
    "PLACEMENT_GPU_VARIANCE_WEIGHT": "0.25",
    "PLACEMENT_PROBE_WEIGHT": "2.25",
}

GRADIENT_EMA_FAST_UPDATE = {
    "PLACEMENT_ADAPTIVE_LEARNING_RATE": "0.16",
    "PLACEMENT_ADAPTIVE_OBS_ALPHA": "0.50",
    "PLACEMENT_ADAPTIVE_MOMENTUM": "0.60",
    "PLACEMENT_ADAPTIVE_DYNAMIC_LR_GAIN": "4.0",
    "PLACEMENT_ADAPTIVE_MAX_LR_MULTIPLIER": "4.0",
    "PLACEMENT_ADAPTIVE_LR_UPDATE_THRESHOLD": "0.18",
    "PLACEMENT_ADAPTIVE_LR_FREEZE_MIN_OBSERVATIONS": "5",
}

GRADIENT_EMA_FAST_NOFREEZE = {
    "PLACEMENT_ADAPTIVE_LR_UPDATE_THRESHOLD": "0.0",
    "PLACEMENT_ADAPTIVE_LR_FREEZE_MIN_OBSERVATIONS": "1000000",
}

GRADIENT_EMA_FAST_LATE_FREEZE = {
    "PLACEMENT_ADAPTIVE_LR_UPDATE_THRESHOLD": "0.17",
    "PLACEMENT_ADAPTIVE_LR_FREEZE_MIN_OBSERVATIONS": "10",
}


def merged(*settings: dict[str, str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for setting in settings:
        out.update(setting)
    return out


METHODS: dict[str, dict[str, str]] = {
    "baseline_trace": dict(full.METHODS["baseline_trace"]),
    "full": merged(
        ROUTE_AND_QUEUE_BEST,
        FULL_SCORE_V2,
        SCORE_V2_PRIORITY_WEIGHTS,
    ),
    "multiplicative": merged(
        ROUTE_AND_QUEUE_BEST,
        MULTIPLICATIVE_SCORE,
        SCORE_V2_PRIORITY_WEIGHTS,
    ),
    "multiplicative_ma5": merged(
        ROUTE_AND_QUEUE_BEST,
        MULTIPLICATIVE_SCORE,
        SCORE_V2_PRIORITY_WEIGHTS,
        MOVING_AVERAGE_INVERSE_5,
    ),
    "partition_decay": merged(
        ROUTE_AND_QUEUE_BEST,
        PARTITION_DECAY_SCORE,
        SCORE_V2_PRIORITY_WEIGHTS,
    ),
    "topology_static_prior": merged(
        ROUTE_AND_QUEUE_BEST,
        TOPOLOGY_STATIC_PRIOR_SCORE,
        SCORE_V2_PRIORITY_WEIGHTS,
    ),
    "ema": merged(
        ROUTE_AND_QUEUE_BEST,
        FIXED_INCREMENTAL,
        SCORE_V2_PRIORITY_WEIGHTS,
        GRADIENT_EMA,
        GRADIENT_EMA_FAST_UPDATE,
    ),
    "baseline_roundrobin": merged(
        ROUTE_AND_QUEUE_BEST,
        ROUNDROBIN_PLACEMENT_BASELINE,
    ),
    "fixed_incremental": merged(ROUTE_AND_QUEUE_BEST, FIXED_INCREMENTAL),
    "fixed_score_v2_priority": merged(
        ROUTE_AND_QUEUE_BEST,
        FIXED_INCREMENTAL,
        SCORE_V2_PRIORITY_WEIGHTS,
    ),
    "fixed_score_v2_external_strong": merged(
        ROUTE_AND_QUEUE_BEST,
        FIXED_INCREMENTAL,
        SCORE_V2_EXTERNAL_STRONG_WEIGHTS,
    ),
    "adaptive_grad_ema": merged(
        ROUTE_AND_QUEUE_BEST,
        FIXED_INCREMENTAL,
        GRADIENT_EMA,
    ),
    "adaptive_grad_ema_bad_init": merged(
        ROUTE_AND_QUEUE_BEST,
        FIXED_INCREMENTAL,
        GRADIENT_EMA,
        GRADIENT_EMA_BAD_INIT,
    ),
    "adaptive_grad_ema_uniform_init": merged(
        ROUTE_AND_QUEUE_BEST,
        FIXED_INCREMENTAL,
        GRADIENT_EMA,
        GRADIENT_EMA_UNIFORM_INIT,
    ),
    "adaptive_grad_ema_uniform_fast": merged(
        ROUTE_AND_QUEUE_BEST,
        FIXED_INCREMENTAL,
        GRADIENT_EMA,
        GRADIENT_EMA_UNIFORM_INIT,
        GRADIENT_EMA_FAST_UPDATE,
    ),
    "adaptive_grad_ema_score_v2_priority_fast": merged(
        ROUTE_AND_QUEUE_BEST,
        FIXED_INCREMENTAL,
        SCORE_V2_PRIORITY_WEIGHTS,
        GRADIENT_EMA,
        GRADIENT_EMA_FAST_UPDATE,
    ),
    "adaptive_grad_ema_uniform_fast_nofreeze": merged(
        ROUTE_AND_QUEUE_BEST,
        FIXED_INCREMENTAL,
        GRADIENT_EMA,
        GRADIENT_EMA_UNIFORM_INIT,
        GRADIENT_EMA_FAST_UPDATE,
        GRADIENT_EMA_FAST_NOFREEZE,
    ),
    "adaptive_grad_ema_uniform_fast_late_freeze": merged(
        ROUTE_AND_QUEUE_BEST,
        FIXED_INCREMENTAL,
        GRADIENT_EMA,
        GRADIENT_EMA_UNIFORM_INIT,
        GRADIENT_EMA_FAST_UPDATE,
        GRADIENT_EMA_FAST_LATE_FREEZE,
    ),
    "adaptive_spsa": merged(
        ROUTE_AND_QUEUE_BEST,
        FIXED_INCREMENTAL,
        SPSA_COMMON,
        {"PLACEMENT_ADAPTIVE_MODE": "spsa"},
    ),
}


DEFAULT_METHODS = [
    "baseline_roundrobin",
    "fixed_incremental",
    "adaptive_grad_ema_uniform_init",
]


METHOD_NOTES = {
    "baseline_trace": "Trace placement + default routing + default local queue scheduling.",
    "full": "Score-v2 full policy: domain-spread placement, probe routing, and PLD-SRPT.",
    "multiplicative": "Incremental placement using multiplicative score penalties, probe routing, and PLD-SRPT.",
    "multiplicative_ma5": "Multiplicative placement with each metric weight set to the inverse of its first-five-placement moving average.",
    "partition_decay": "Two-stage partition greedy placement with per-expert temporary heat decay n/(n+1) after each replica.",
    "topology_static_prior": "One-shot topology-derived placement weights with lower-bound normalized saturated score penalties.",
    "ema": "Fast EMA adaptive incremental placement initialized from score-v2 priority weights, with probe routing and PLD-SRPT.",
    "baseline_roundrobin": "Round-robin/default expert placement with the same probe routing and PLD-SRPT local queue.",
    "fixed_incremental": "Fixed hyperparameters with incremental expert placement, probe routing, and PLD-SRPT.",
    "fixed_score_v2_priority": "Score-v2 fixed weights: replica/external first, latency next, L1/GPU heat balance as guardrails.",
    "fixed_score_v2_external_strong": "Score-v2 fixed weights with external coverage treated as the dominant objective.",
    "adaptive_grad_ema": "Incremental expert placement with derivative/EMA dynamic weight updates.",
    "adaptive_grad_ema_bad_init": "Same EMA update, initialized far from fixed weights to test convergence.",
    "adaptive_grad_ema_uniform_init": "Same EMA update, initialized with a uniform six-metric vector [4,4,4,4,4,4].",
    "adaptive_grad_ema_uniform_fast": "Uniform six-metric EMA with faster updates and a low-learning-rate momentum freeze threshold.",
    "adaptive_grad_ema_score_v2_priority_fast": "Fast EMA initialized from score-v2 priority weights.",
    "adaptive_grad_ema_uniform_fast_nofreeze": "Uniform fast EMA with adaptive initial placement skipped from observations and freeze disabled.",
    "adaptive_grad_ema_uniform_fast_late_freeze": "Uniform fast EMA with adaptive initial placement skipped from observations and freeze delayed to 10 observations at lr threshold 0.17.",
    "adaptive_spsa": "Incremental expert placement with online paired SPSA weight updates.",
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
    for candidate in ("full", "fixed_score_v2_priority", "fixed_incremental"):
        if candidate in available:
            return candidate
    positives = [
        (row_float(row, "average_TPOT_ns"), str(row.get("method", "")))
        for row in rows
        if row_float(row, "average_TPOT_ns") > 0.0
    ]
    if positives:
        positives.sort(key=lambda item: item[0])
        return positives[0][1]
    return ""


def reference_tpot_ns(rows: list[dict[str, Any]]) -> float:
    method = reference_method(rows)
    for row in rows:
        if row.get("method") == method:
            value = row_float(row, "average_TPOT_ns")
            if value > 0.0:
                return value
    positives = [row_float(row, "average_TPOT_ns") for row in rows]
    positives = [value for value in positives if value > 0.0]
    return min(positives) if positives else 0.0


def fixed_tpot_ns(rows: list[dict[str, Any]]) -> float:
    return reference_tpot_ns(rows)


def plot_adaptive_delta(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    labels = [str(row["method"]) for row in rows]
    tpot_us = [row_float(row, "average_TPOT_ns") / 1000.0 for row in rows]
    base_ns = fixed_tpot_ns(rows)
    reference = reference_method(rows)
    colors = ["#2ca25f" if label == reference else
              "#4e79a7" if label in ("adaptive_grad_ema", "ema", "multiplicative", "multiplicative_ma5", "partition_decay", "topology_static_prior") else
              "#f28e2b" for label in labels]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(9, len(labels) * 1.8), 5.2))
    bars = ax.bar(x, tpot_us, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("avg TPOT (us)")
    ax.set_title("Adaptive Placement Weight Comparison")
    ax.grid(True, axis="y", alpha=0.25)
    for bar, row in zip(bars, rows):
        tpot_ns = row_float(row, "average_TPOT_ns")
        if base_ns > 0.0 and tpot_ns > 0.0:
            delta = (tpot_ns / base_ns - 1.0) * 100.0
            label = reference if row.get("method") == reference else f"{delta:+.1f}%"
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
    fig.savefig(fig_dir / "adaptive_placement_tpot.png", dpi=170)
    plt.close(fig)


def selected_methods(args: Any) -> dict[str, dict[str, str]]:
    if args.methods == ["all"]:
        return METHODS
    unknown = [name for name in args.methods if name not in METHODS]
    if unknown:
        valid = ", ".join(METHODS)
        raise SystemExit(f"unknown methods: {', '.join(unknown)}; valid: {valid}")
    return {name: METHODS[name] for name in args.methods}


def parse_args() -> Any:
    full.METHODS = METHODS
    full.DEFAULT_COMPARE_METHODS = DEFAULT_METHODS
    full.DEFAULT_OUT = DEFAULT_OUT
    full.PARSER_DESCRIPTION = (
        "Run fixed vs adaptive incremental expert-placement hyperparameter comparisons."
    )
    args = full.parse_args()
    if (
        "--trace-access-divisor" not in sys.argv
        and "--trace-access-100x" not in sys.argv
    ):
        args.trace_access_divisor = 500
    return args


def write_readme(out_dir: Path,
                 args: Any,
                 rows: list[dict[str, Any]],
                 methods: dict[str, dict[str, str]]) -> None:
    first_policy = {}
    if rows:
        first_policy = parse_policy(out_dir / str(rows[0]["method"]) / "policy.conf")
    trace_divisor = policy_get(
        first_policy,
        "TRACE_ACCESS_DIVISOR",
        100 if args.trace_access_100x else max(1, args.trace_access_divisor),
    )
    base_ns = fixed_tpot_ns(rows)
    reference = reference_method(rows) or "reference"

    method_lines = []
    for method, settings in methods.items():
        policy = parse_policy(out_dir / method / "policy.conf")
        if not policy:
            policy = full.settings_for_method(args, method, settings)
        method_lines.append(
            "| {method} | {note} | {placement} | {adaptive} | {lr} | {lr_threshold} | {freeze_min_obs} | {alpha} | {momentum} | {weight_sum} | {route} | {local} | {weights} |".format(
                method=method,
                note=METHOD_NOTES.get(method, ""),
                placement=policy_get(policy, "EXPERT_PLACEMENT_POLICY", "default"),
                adaptive=policy_get(policy, "PLACEMENT_ADAPTIVE_MODE", "fixed"),
                lr=policy_get(policy, "PLACEMENT_ADAPTIVE_LEARNING_RATE", "-"),
                lr_threshold=policy_get(
                    policy, "PLACEMENT_ADAPTIVE_LR_UPDATE_THRESHOLD", "0"
                ),
                freeze_min_obs=policy_get(
                    policy, "PLACEMENT_ADAPTIVE_LR_FREEZE_MIN_OBSERVATIONS", "1"
                ),
                alpha=policy_get(policy, "PLACEMENT_ADAPTIVE_OBS_ALPHA", "-"),
                momentum=policy_get(policy, "PLACEMENT_ADAPTIVE_MOMENTUM", "-"),
                weight_sum=policy_get(policy, "PLACEMENT_ADAPTIVE_WEIGHT_SUM", "-"),
                route=policy_get(policy, "SAME_RANK_ROUTE_POLICY", "default"),
                local=policy_get(policy, "LOCAL_FLOW_SCHEDULE_POLICY", "default"),
                weights="[" + ",".join([
                    policy_get(policy, "PLACEMENT_REPLICA_BALANCE_WEIGHT", "0"),
                    policy_get(policy, "PLACEMENT_EXTERNAL_WEIGHT", "0"),
                    policy_get(policy, "PLACEMENT_L1_VARIANCE_WEIGHT", "0"),
                    policy_get(policy, "PLACEMENT_L1_MAX_WEIGHT", "0"),
                    policy_get(policy, "PLACEMENT_GPU_VARIANCE_WEIGHT", "0"),
                    policy_get(policy, "PLACEMENT_PROBE_WEIGHT", "0"),
                ]) + "]",
            )
        )

    summary_lines = []
    for row in rows:
        tpot_ns = row_float(row, "average_TPOT_ns")
        if base_ns > 0.0 and tpot_ns > 0.0:
            delta = reference if row.get("method") == reference else f"{(tpot_ns / base_ns - 1.0) * 100.0:+.2f}%"
        else:
            delta = "n/a"
        summary_lines.append(
            "| {method} | {status} | {completed} | {latency_us} | {tpot_us} | {delta} | {trace_done} | {remaining} | {qavg} |".format(
                method=row.get("method", ""),
                status=row.get("status", ""),
                completed=row.get("completed_tasks", 0),
                latency_us=fmt_us(row.get("average_latency_ns", 0)),
                tpot_us=fmt_us(row.get("average_TPOT_ns", 0)),
                delta=delta,
                trace_done=row.get("trace_completed", 0),
                remaining=row.get("trace_remaining_accesses", 0),
                qavg=row.get("queued_local_flows_avg", 0),
            )
        )

    text = f"""# Adaptive Placement Weight Comparison

Generated by `simulation/run_dispatch_adaptive_placement_visual.py`.

Reference method: `{reference}`.

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

## Method Matrix

| method | description | expert placement | adaptive mode | learning rate | lr update threshold | freeze min obs | obs alpha | momentum | weight sum | route policy | local queue | initial weights |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---|
{chr(10).join(method_lines)}

## Result Summary

| method | status | completed tasks | avg latency us | avg TPOT us | TPOT delta vs reference | trace completed | remaining trace accesses | avg queued flows |
|---|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(summary_lines)}

## Notes

- The compared placement objective uses six metrics: `replica`, `external`, `l1_var`, `l1_max`, `gpu_heat_var`, and `access_latency`.
- The legacy fixed incremental reference uses `[4,4,4,4,4,4]`; score-v2 priority uses `[6,8,1,1.5,0.5,7]`. EMA keeps the total weight budget normalized to `24.0` with a positive per-dimension floor.
- `adaptive_grad_ema_bad_init` uses the normalized placement objective components as partial derivatives with respect to their weights.
- The EMA baseline tracks recent observations; a component above its recent baseline gets a larger future weight, while the total weight budget is kept fixed.
- `PLACEMENT_ADAPTIVE_LR_UPDATE_THRESHOLD` permanently freezes a dimension at its current weight once its dynamic learning rate falls below the threshold after `PLACEMENT_ADAPTIVE_LR_FREEZE_MIN_OBSERVATIONS`; frozen dimensions are removed from later partial/momentum updates and are logged with effective `lr=0`.
- These modes adapt algorithm-one hyperparameters only. Routing and local queue scheduling are held fixed.

## Artifacts

- `summary.csv`: parsed metrics.
- `best_case.txt`: lowest parsed TPOT case.
- `<method>/policy.conf`: exact config.
- `<method>/mncc.log`: placement adaptive logs and mnCCL logs.
- `figures/adaptive_placement_tpot.png`: TPOT delta relative to the selected reference method.
- `figures/adaptive_weights_<method>.png`: per-weight convergence trajectories parsed from placement adaptive logs.
- `figures/adaptive_weight_distance.png`: normalized distance-to-final weight vector, used to inspect convergence speed.
- `figures/latency_tpot_bar.png`: latency and TPOT comparison.
- `figures/queue_pressure_bar.png`: local queue pressure comparison.
- `figures/expert_heatmap_<method>.png`: placement/access heatmap.
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
    plot_adaptive_delta(out_dir, rows)
    weight_plot.plot_from_result_dir(out_dir, ordered_methods)
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
