#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import run_dispatch_full_compare_visual as full


ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "results" / "hw_dispatch_ablation_100x"


BEST_ASYNC_PROBE = {
    "EXPERT_PLACEMENT_POLICY": "probe_balanced_domain_spread",
    "SAME_RANK_ROUTE_POLICY": "probe_rtt_delta",
    "PLACEMENT_ACCESS_MODE": "posterior",
    "ASYNC_ROUTE_PROBE_ENABLE": "1",
    "ASYNC_ROUTE_LIVE_ROUTE": "1",
    "TRACE_USE_DEVICE_PLACEMENT": "0",
    **full.COMMON_PROBE_WEIGHTS,
}


BEST_ASYNC_PROBE_SCORE_V2 = {
    "EXPERT_PLACEMENT_POLICY": "probe_balanced_domain_spread",
    "SAME_RANK_ROUTE_POLICY": "probe_rtt_delta",
    "PLACEMENT_ACCESS_MODE": "posterior",
    "ASYNC_ROUTE_PROBE_ENABLE": "1",
    "ASYNC_ROUTE_LIVE_ROUTE": "1",
    "TRACE_USE_DEVICE_PLACEMENT": "0",
    **full.SCORE_V2_PRIORITY_WEIGHTS,
}


def with_overrides(*overrides: dict[str, str]) -> dict[str, str]:
    settings = dict(BEST_ASYNC_PROBE_SCORE_V2)
    for override in overrides:
        settings.update(override)
    return settings


ABLATION_METHODS: dict[str, dict[str, str]] = {
    "baseline_uniform": dict(full.METHODS["baseline_uniform"]),
    "baseline_trace": dict(full.METHODS["baseline_trace"]),
    "best_async_probe": dict(BEST_ASYNC_PROBE),
    "best_async_probe_score_v2": dict(BEST_ASYNC_PROBE_SCORE_V2),
    "sync_probe_route": with_overrides(
        {
            "ASYNC_ROUTE_PROBE_ENABLE": "0",
            "ASYNC_ROUTE_LIVE_ROUTE": "0",
        }
    ),
    "async_probe_no_live": with_overrides({"ASYNC_ROUTE_LIVE_ROUTE": "0"}),
    "default_route": with_overrides(
        {
            "SAME_RANK_ROUTE_POLICY": "default",
            "ASYNC_ROUTE_PROBE_ENABLE": "0",
            "ASYNC_ROUTE_LIVE_ROUTE": "0",
        }
    ),
    "no_posterior_access": with_overrides({"PLACEMENT_ACCESS_MODE": "initial"}),
    "no_probe_path_cost": with_overrides(
        {
            "PLACEMENT_PROBE_WEIGHT": "0.0",
            "PLACEMENT_PROBE_DELTA_WEIGHT": "0.0",
            "PLACEMENT_PROBE_RTT_WEIGHT": "0.0",
        }
    ),
    "no_replica_balance": with_overrides({"PLACEMENT_REPLICA_BALANCE_WEIGHT": "0.0"}),
    "no_external_cost": with_overrides({"PLACEMENT_EXTERNAL_WEIGHT": "0.0"}),
    "no_l1_variance": with_overrides({"PLACEMENT_L1_VARIANCE_WEIGHT": "0.0"}),
    "no_l1_max": with_overrides({"PLACEMENT_L1_MAX_WEIGHT": "0.0"}),
    "no_gpu_variance": with_overrides({"PLACEMENT_GPU_VARIANCE_WEIGHT": "0.0"}),
    "no_probe_cost": with_overrides({"PLACEMENT_PROBE_WEIGHT": "0.0"}),
    "domain_spread_base": with_overrides(
        {"EXPERT_PLACEMENT_POLICY": "probe_balanced_domain_spread_base"}
    ),
    "default_placement": with_overrides({"EXPERT_PLACEMENT_POLICY": "default"}),
}


DEFAULT_ABLATION_METHODS = [
    "baseline_trace",
    "best_async_probe_score_v2",
    "no_replica_balance",
    "no_external_cost",
    "no_l1_variance",
    "no_l1_max",
    "no_gpu_variance",
    "no_probe_cost",
]


METHOD_NOTES = {
    "baseline_uniform": "Trace device placement + default routing + uniform trace targets.",
    "baseline_trace": "Trace device placement + default routing + original trace targets.",
    "best_async_probe": "Full current algorithm: posterior placement, domain-spread placement, async probe, live route refresh.",
    "best_async_probe_score_v2": "Best full algorithm: posterior placement, domain-spread placement, async probe, live route refresh, score-v2 weights.",
    "sync_probe_route": "Ablates async/background route probing; keeps probe_rtt_delta routing without live async scores.",
    "async_probe_no_live": "Keeps background probes but ablates live route score refresh.",
    "default_route": "Ablates probe_rtt_delta route selection; uses default local-first/random routing.",
    "no_posterior_access": "Ablates posterior expert access estimates; placement uses initial access prior.",
    "no_probe_path_cost": "Ablates probe-aware terms inside expert placement scoring.",
    "no_replica_balance": "Ablates same-expert replica distribution balance in placement scoring.",
    "no_external_cost": "Ablates external-access minimization in placement scoring.",
    "no_l1_variance": "Ablates L1 variance balance in placement scoring.",
    "no_l1_max": "Ablates L1 max balance in placement scoring.",
    "no_gpu_variance": "Ablates GPU variance balance in placement scoring.",
    "no_probe_cost": "Ablates probe cost in placement scoring.",
    "domain_spread_base": "Uses the base domain-spread placement without the latest hot-surplus improvements.",
    "default_placement": "Ablates optimized placement entirely; keeps route probe stack.",
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


def policy_for_method(
    args: Any,
    out_dir: Path,
    method: str,
    settings: dict[str, str],
) -> dict[str, str]:
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


def best_tpot_ns(rows: list[dict[str, Any]], best_method: str) -> float:
    for row in rows:
        if row.get("method") == best_method:
            tpot = row_float(row, "average_TPOT_ns")
            if tpot > 0.0:
                return tpot
    positives = [row_float(row, "average_TPOT_ns") for row in rows]
    positives = [value for value in positives if value > 0.0]
    return min(positives) if positives else 0.0


def plot_ablation_tpot(
    out_dir: Path,
    rows: list[dict[str, Any]],
    best_method: str = "best_async_probe_score_v2",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    labels = [str(row["method"]) for row in rows]
    tpot_us = [row_float(row, "average_TPOT_ns") / 1000.0 for row in rows]
    base_ns = best_tpot_ns(rows, best_method)
    colors = []
    for label in labels:
        if label == best_method:
            colors.append("#2ca25f")
        elif label.startswith("baseline"):
            colors.append("#7f8c8d")
        else:
            colors.append("#f28e2b")

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 1.25), 5.4))
    bars = ax.bar(x, tpot_us, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.set_ylabel("avg TPOT (us)")
    ax.set_title("Ablation TPOT")
    ax.grid(True, axis="y", alpha=0.25)
    for bar, row in zip(bars, rows):
        tpot_ns = row_float(row, "average_TPOT_ns")
        if base_ns > 0.0 and tpot_ns > 0.0:
            delta = (tpot_ns / base_ns - 1.0) * 100.0
            label = "best" if row.get("method") == best_method else f"{delta:+.1f}%"
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
    fig.tight_layout()
    fig.savefig(fig_dir / "ablation_tpot_vs_best.png", dpi=170)
    plt.close(fig)


def selected_methods(args: Any) -> dict[str, dict[str, str]]:
    if args.methods == ["all"]:
        return ABLATION_METHODS
    unknown = [name for name in args.methods if name not in ABLATION_METHODS]
    if unknown:
        valid = ", ".join(ABLATION_METHODS)
        raise SystemExit(f"unknown methods: {', '.join(unknown)}; valid: {valid}")
    return {name: ABLATION_METHODS[name] for name in args.methods}


def parse_args() -> Any:
    full.METHODS = ABLATION_METHODS
    full.DEFAULT_COMPARE_METHODS = DEFAULT_ABLATION_METHODS
    full.DEFAULT_OUT = DEFAULT_OUT
    full.PARSER_DESCRIPTION = (
        "Run dispatch policy ablations and generate TPOT, queue, and expert heatmap plots."
    )
    args = full.parse_args()

    if (
        "--trace-access-divisor" not in sys.argv
        and "--trace-access-100x" not in sys.argv
    ):
        args.trace_access_divisor = 100
    return args


def write_ablation_readme(
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
    base_ns = best_tpot_ns(rows, "best_async_probe_score_v2")

    method_lines = []
    for method, settings in methods.items():
        policy = policy_for_method(args, out_dir, method, settings)
        method_lines.append(
            "| {method} | {note} | {placement} | {route} | {access} | {async_probe} | {live_route} | {trace_device} | {rep_w} | {ext_w} | {l1_var_w} | {l1_max_w} | {gpu_w} | {probe_w} |".format(
                method=method,
                note=METHOD_NOTES.get(method, ""),
                placement=policy_get(policy, "EXPERT_PLACEMENT_POLICY", "default"),
                route=policy_get(policy, "SAME_RANK_ROUTE_POLICY", "default"),
                access=policy_get(policy, "PLACEMENT_ACCESS_MODE", "initial"),
                async_probe=policy_get(policy, "ASYNC_ROUTE_PROBE_ENABLE", "0"),
                live_route=policy_get(policy, "ASYNC_ROUTE_LIVE_ROUTE", "0"),
                trace_device=policy_get(policy, "TRACE_USE_DEVICE_PLACEMENT", "0"),
                rep_w=policy_get(policy, "PLACEMENT_REPLICA_BALANCE_WEIGHT", "0"),
                ext_w=policy_get(policy, "PLACEMENT_EXTERNAL_WEIGHT", "0"),
                l1_var_w=policy_get(policy, "PLACEMENT_L1_VARIANCE_WEIGHT", "0"),
                l1_max_w=policy_get(policy, "PLACEMENT_L1_MAX_WEIGHT", "0"),
                gpu_w=policy_get(policy, "PLACEMENT_GPU_VARIANCE_WEIGHT", "0"),
                probe_w=policy_get(policy, "PLACEMENT_PROBE_WEIGHT", "0"),
            )
        )

    summary_lines = []
    for row in rows:
        tpot_ns = row_float(row, "average_TPOT_ns")
        delta = "n/a"
        if base_ns > 0.0 and tpot_ns > 0.0:
            delta_value = (tpot_ns / base_ns - 1.0) * 100.0
            delta = (
                "best"
                if row.get("method") == "best_async_probe_score_v2"
                else f"{delta_value:+.2f}%"
            )
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

    text = f"""# Dispatch Policy Ablation

Generated by `simulation/run_dispatch_ablation_visual.py`.

The reference method is `best_async_probe_score_v2`. Exact per-run `policy.conf` files are authoritative, especially when this README is regenerated with `--skip-run`.

## Workload

- network config: `{policy_get(first_policy, "NETWORK_CONFIG", args.network_config)}`
- trace dispatch: `{policy_get(first_policy, "TRACE_DISPATCH_ENABLE", int(args.trace_dispatch))}`
- trace layer id: `{policy_get(first_policy, "TRACE_LAYER_ID", args.trace_layer_id)}`
- trace access divisor: `{trace_divisor}`
- trace stop on complete: `{policy_get(first_policy, "TRACE_STOP_ON_COMPLETE", int(args.trace_stop_on_complete))}`
- dispatch need: `{policy_get(first_policy, "DISPATCH_NEED", args.dispatch_need)}`
- expert num: `{policy_get(first_policy, "EXPERT_NUM", args.expert_num)}`
- expert per GPU: `{policy_get(first_policy, "EXPERT_PER_GPU", args.expert_per_gpu)}`
- top-k: `{policy_get(first_policy, "DISPATCH_TOPK", args.topk)}`
- dispatch N: `{policy_get(first_policy, "DISPATCH_N", args.dispatch_n)}`
- FFN params per expert: `{policy_get(first_policy, "DISPATCH_EXPERT_FFN_PARAMS", full.payload_params(args))}`
- precision bytes: `{policy_get(first_policy, "DISPATCH_PRECISION_BYTES", args.precision_bytes)}`
- batch size: `{policy_get(first_policy, "DISPATCH_BATCH_SIZE", args.batch_size)}`
- route probe max inflight: `{policy_get(first_policy, "ROUTE_PROBE_MAX_INFLIGHT", args.route_probe_max_inflight)}`
- async route probe interval ns: `{policy_get(first_policy, "ASYNC_ROUTE_PROBE_INTERVAL_NS", args.async_route_probe_interval_ns)}`
- async route probe budget: `{policy_get(first_policy, "ASYNC_ROUTE_PROBE_BUDGET", args.async_route_probe_budget)}`
- async route probe refresh ns: `{policy_get(first_policy, "ASYNC_ROUTE_PROBE_REFRESH_NS", args.async_route_probe_refresh_ns)}`
- comparison heatmap snapshot: `{args.placement_heatmap_snapshot}`
- access heat display scale: `{args.access_heat_scale}`

## Ablation Matrix

| method | ablated component | expert placement | route policy | access mode | async probe | live route | trace placement | replica w | external w | l1 var w | l1 max w | gpu var w | probe w |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(method_lines)}

## Result Summary

| method | status | completed tasks | avg latency us | avg TPOT us | TPOT delta vs best | trace completed | remaining trace accesses | avg queued flows |
|---|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(summary_lines)}

## Artifacts

- `summary.csv`: parsed cross-method metrics.
- `best_case.txt`: lowest parsed TPOT case.
- `<method>/policy.conf`: exact policy config for each run.
- `<method>/mncc.log`: mnCCL log.
- `<method>/mncc_flow_finish.csv`: per-flow completion records.
- `figures/ablation_tpot_vs_best.png`: TPOT and percentage delta relative to `best_async_probe_score_v2`.
- `figures/latency_tpot_bar.png`: latency and TPOT comparison.
- `figures/queue_pressure_bar.png`: local queue pressure comparison.
- `figures/tpot_relative_to_baseline_trace.png`: TPOT normalized to `baseline_trace` when that method is present.
- `figures/expert_heatmap_<method>.png`: placement/access heatmap for the configured comparison snapshot. The access panel uses the configured display scale only for readability.
- `figures/expert_placement_latest_<method>.png`: placement-only heatmap for the latest placement update.
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
    full.plot_relative_tpot(out_dir, rows)
    plot_ablation_tpot(out_dir, rows)
    if args.print_placement_detail:
        full.plot_heatmaps(
            out_dir,
            ordered_methods,
            args.heatmap_clip_percentile,
            args.placement_heatmap_snapshot,
            args.access_heat_scale,
        )
    write_ablation_readme(out_dir, args, rows, methods)
    print(f"summary: {out_dir / 'summary.csv'}")
    print(f"figures: {out_dir / 'figures'}")
    print(f"readme: {out_dir / 'README.md'}")


if __name__ == "__main__":
    main()
