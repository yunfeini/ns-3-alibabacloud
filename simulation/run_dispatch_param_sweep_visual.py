#!/usr/bin/env python3
from __future__ import annotations

import csv
import itertools
import sys
from pathlib import Path
from typing import Any

import run_dispatch_full_compare_visual as full


ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "results" / "hw_param_sweep_100x"
REFERENCE_METHOD = "ref_ew3_posterior"
BASELINE_TRACE = dict(full.METHODS["baseline_trace"])


BASE_STRATEGY = {
    "EXPERT_PLACEMENT_POLICY": "probe_balanced_domain_spread",
    "SAME_RANK_ROUTE_POLICY": "probe_rtt_delta",
    "PLACEMENT_ACCESS_MODE": "posterior",
    "ASYNC_ROUTE_PROBE_ENABLE": "1",
    "ASYNC_ROUTE_LIVE_ROUTE": "1",
    "TRACE_USE_DEVICE_PLACEMENT": "0",
    **full.COMMON_PROBE_WEIGHTS,
}


def float_tag(value: float) -> str:
    text = f"{value:g}"
    return text.replace("-", "m").replace(".", "p")


def int_tag(value: int) -> str:
    if value >= 1_000_000 and value % 1_000_000 == 0:
        return f"{value // 1_000_000}m"
    if value >= 1_000 and value % 1_000 == 0:
        return f"{value // 1_000}k"
    return str(value)


def setting_with(
    external_weight: float,
    access_mode: str,
    interval_ns: int,
    budget: int,
    refresh_ns: int,
    drift_threshold: float | None = None,
    drift_min_samples: int | None = None,
) -> dict[str, str]:
    settings = dict(BASE_STRATEGY)
    settings.update(
        {
            "PLACEMENT_EXTERNAL_WEIGHT": f"{external_weight:g}",
            "PLACEMENT_ACCESS_MODE": access_mode,
            "ASYNC_ROUTE_PROBE_INTERVAL_NS": str(interval_ns),
            "ASYNC_ROUTE_PROBE_BUDGET": str(budget),
            "ASYNC_ROUTE_PROBE_REFRESH_NS": str(refresh_ns),
        }
    )
    if access_mode == "initial_until_drift":
        settings["PLACEMENT_DRIFT_THRESHOLD"] = f"{drift_threshold if drift_threshold is not None else 0.1:g}"
        settings["PLACEMENT_DRIFT_MIN_SAMPLES"] = str(drift_min_samples if drift_min_samples is not None else 8192)
    return settings


def build_cases(args: Any) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    cases: dict[str, dict[str, str]] = {}
    manifest: dict[str, dict[str, str]] = {}

    if args.include_baseline_trace:
        cases["baseline_trace"] = dict(BASELINE_TRACE)
        manifest["baseline_trace"] = {
            "external_weight": "",
            "access_mode": "baseline_trace",
            "drift_threshold": "",
            "drift_min_samples": "",
            "async_interval_ns": "",
            "async_budget": "",
            "async_refresh_ns": "",
        }

    for external_weight, interval_ns, budget, refresh_ns in itertools.product(
        args.external_weights,
        args.route_probe_intervals_ns,
        args.route_probe_budgets,
        args.route_probe_refresh_ns,
    ):
        for mode in args.access_modes:
            if mode == "drift":
                for threshold, min_samples in itertools.product(
                    args.drift_thresholds,
                    args.drift_min_samples,
                ):
                    name = (
                        f"ew{float_tag(external_weight)}_drift"
                        f"{float_tag(threshold)}_s{int_tag(min_samples)}"
                        f"_i{int_tag(interval_ns)}_b{budget}_r{int_tag(refresh_ns)}"
                    )
                    cases[name] = setting_with(
                        external_weight,
                        "initial_until_drift",
                        interval_ns,
                        budget,
                        refresh_ns,
                        threshold,
                        min_samples,
                    )
                    manifest[name] = {
                        "external_weight": f"{external_weight:g}",
                        "access_mode": "initial_until_drift",
                        "drift_threshold": f"{threshold:g}",
                        "drift_min_samples": str(min_samples),
                        "async_interval_ns": str(interval_ns),
                        "async_budget": str(budget),
                        "async_refresh_ns": str(refresh_ns),
                    }
                continue

            name = (
                f"ew{float_tag(external_weight)}_{mode}"
                f"_i{int_tag(interval_ns)}_b{budget}_r{int_tag(refresh_ns)}"
            )
            if (
                external_weight == 3.0
                and mode == "posterior"
                and interval_ns == 20_000
                and budget == 128
                and refresh_ns == 500_000
            ):
                name = REFERENCE_METHOD
            cases[name] = setting_with(
                external_weight,
                mode,
                interval_ns,
                budget,
                refresh_ns,
            )
            manifest[name] = {
                "external_weight": f"{external_weight:g}",
                "access_mode": mode,
                "drift_threshold": "",
                "drift_min_samples": "",
                "async_interval_ns": str(interval_ns),
                "async_budget": str(budget),
                "async_refresh_ns": str(refresh_ns),
            }

    return cases, manifest


def parse_args() -> Any:
    full.DEFAULT_OUT = DEFAULT_OUT
    full.DEFAULT_COMPARE_METHODS = ["all"]
    full.METHODS = {"all": {}}
    full.PARSER_DESCRIPTION = (
        "Run parameter sweeps for the current dispatch strategy and generate visual summaries."
    )
    parser = full.build_parser()
    parser.add_argument(
        "--external-weights",
        nargs="+",
        type=float,
        default=[0.0, 0.5, 1.0, 2.0, 3.0],
        help="PLACEMENT_EXTERNAL_WEIGHT values to scan.",
    )
    parser.add_argument(
        "--access-modes",
        nargs="+",
        choices=["initial", "posterior", "drift"],
        default=["initial", "posterior", "drift"],
        help="'drift' maps to PLACEMENT_ACCESS_MODE=initial_until_drift.",
    )
    parser.add_argument(
        "--drift-thresholds",
        nargs="+",
        type=float,
        default=[0.05, 0.15],
        help="PLACEMENT_DRIFT_THRESHOLD values for access mode drift.",
    )
    parser.add_argument(
        "--drift-min-samples",
        nargs="+",
        type=int,
        default=[8192],
        help="PLACEMENT_DRIFT_MIN_SAMPLES values for access mode drift.",
    )
    parser.add_argument(
        "--route-probe-intervals-ns",
        nargs="+",
        type=int,
        default=[20_000],
        help="ASYNC_ROUTE_PROBE_INTERVAL_NS values to scan.",
    )
    parser.add_argument(
        "--route-probe-budgets",
        nargs="+",
        type=int,
        default=[128],
        help="ASYNC_ROUTE_PROBE_BUDGET values to scan.",
    )
    parser.add_argument(
        "--route-probe-refresh-ns",
        nargs="+",
        type=int,
        default=[500_000],
        help="ASYNC_ROUTE_PROBE_REFRESH_NS values to scan.",
    )
    parser.add_argument(
        "--no-baseline-trace",
        dest="include_baseline_trace",
        action="store_false",
        help="Do not include baseline_trace as a context row.",
    )
    parser.set_defaults(include_baseline_trace=True)
    args = parser.parse_args()
    if (
        "--trace-access-divisor" not in sys.argv
        and "--trace-access-100x" not in sys.argv
    ):
        args.trace_access_divisor = 100
    return args


def selected_cases(
    args: Any,
    cases: dict[str, dict[str, str]],
    manifest: dict[str, dict[str, str]],
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    if args.methods == ["all"]:
        return cases, manifest
    unknown = [method for method in args.methods if method not in cases]
    if unknown:
        valid = ", ".join(cases)
        raise SystemExit(f"unknown methods: {', '.join(unknown)}; valid: {valid}")
    selected = {method: cases[method] for method in args.methods}
    selected_manifest = {method: manifest[method] for method in args.methods}
    return selected, selected_manifest


def row_float(row: dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def write_manifest(out_dir: Path, manifest: dict[str, dict[str, str]]) -> None:
    keys = [
        "method",
        "external_weight",
        "access_mode",
        "drift_threshold",
        "drift_min_samples",
        "async_interval_ns",
        "async_budget",
        "async_refresh_ns",
    ]
    with (out_dir / "sweep_manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for method, values in manifest.items():
            row = {"method": method}
            row.update(values)
            writer.writerow(row)


def plot_sweep(out_dir: Path, rows: list[dict[str, Any]], manifest: dict[str, dict[str, str]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    ranked = sorted(rows, key=lambda row: row_float(row, "average_TPOT_ns") or float("inf"))
    labels = [str(row["method"]) for row in ranked]
    tpot_us = [row_float(row, "average_TPOT_ns") / 1000.0 for row in ranked]
    best = min([value for value in tpot_us if value > 0.0], default=0.0)

    colors = []
    for label in labels:
        if label == REFERENCE_METHOD:
            colors.append("#4e79a7")
        elif label.startswith("baseline"):
            colors.append("#7f8c8d")
        else:
            colors.append("#59a14f")

    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 0.8), 5.6))
    bars = ax.bar(np.arange(len(labels)), tpot_us, color=colors)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("avg TPOT (us)")
    ax.set_title("Parameter Sweep TPOT Ranking")
    ax.grid(True, axis="y", alpha=0.25)
    for bar, value in zip(bars, tpot_us):
        if best > 0 and value > 0:
            label = "best" if abs(value - best) < 1e-9 else f"{(value / best - 1.0) * 100.0:+.1f}%"
            ax.text(bar.get_x() + bar.get_width() / 2.0, value, label, ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "param_sweep_tpot_rank.png", dpi=170)
    plt.close(fig)

    points = []
    for row in rows:
        method = str(row["method"])
        meta = manifest.get(method, {})
        if method.startswith("baseline") or not meta.get("external_weight"):
            continue
        try:
            external_weight = float(meta["external_weight"])
        except (TypeError, ValueError):
            continue
        points.append((external_weight, meta.get("access_mode", ""), row_float(row, "average_TPOT_ns") / 1000.0, method))
    if points:
        fig, ax = plt.subplots(figsize=(9, 5.4))
        access_modes = sorted({point[1] for point in points})
        for access_mode in access_modes:
            series = sorted([point for point in points if point[1] == access_mode])
            ax.plot(
                [point[0] for point in series],
                [point[2] for point in series],
                marker="o",
                label=access_mode,
            )
        ax.set_xlabel("PLACEMENT_EXTERNAL_WEIGHT")
        ax.set_ylabel("avg TPOT (us)")
        ax.set_title("External Weight Sweep")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / "external_weight_tpot.png", dpi=170)
        plt.close(fig)


def write_readme(
    out_dir: Path,
    args: Any,
    rows: list[dict[str, Any]],
    manifest: dict[str, dict[str, str]],
) -> None:
    ranked = sorted(rows, key=lambda row: row_float(row, "average_TPOT_ns") or float("inf"))
    best_ns = row_float(ranked[0], "average_TPOT_ns") if ranked else 0.0
    reference_row = next((row for row in rows if row.get("method") == REFERENCE_METHOD), None)
    reference_ns = row_float(reference_row, "average_TPOT_ns") if reference_row else 0.0
    summary_lines = []
    for row in ranked:
        method = str(row["method"])
        meta = manifest.get(method, {})
        tpot_ns = row_float(row, "average_TPOT_ns")
        delta_best = "n/a"
        delta_ref = "n/a"
        if best_ns > 0 and tpot_ns > 0:
            delta_best = "best" if method == ranked[0].get("method") else f"{(tpot_ns / best_ns - 1.0) * 100.0:+.2f}%"
        if reference_ns > 0 and tpot_ns > 0:
            delta_ref = "ref" if method == REFERENCE_METHOD else f"{(tpot_ns / reference_ns - 1.0) * 100.0:+.2f}%"
        summary_lines.append(
            "| {method} | {status} | {tpot:.3f} | {delta_best} | {delta_ref} | {ew} | {mode} | {drift} | {samples} | {qavg:.1f} |".format(
                method=method,
                status=row.get("status", ""),
                tpot=tpot_ns / 1000.0,
                delta_best=delta_best,
                delta_ref=delta_ref,
                ew=meta.get("external_weight", ""),
                mode=meta.get("access_mode", ""),
                drift=meta.get("drift_threshold", ""),
                samples=meta.get("drift_min_samples", ""),
                qavg=row_float(row, "queued_local_flows_avg"),
            )
        )

    text = f"""# Dispatch Parameter Sweep

Generated by `simulation/run_dispatch_param_sweep_visual.py`.

This sweep keeps the latest strategy structure fixed: `probe_balanced_domain_spread` placement, `probe_rtt_delta` routing, async route probing, and live route refresh. It scans placement external-access weight and access-frequency mode/gating. Exact per-run `policy.conf` files are authoritative.

## Workload

- trace access divisor: `{100 if args.trace_access_100x else max(1, args.trace_access_divisor)}`
- dispatch need: `{args.dispatch_need}`
- expert num: `{args.expert_num}`
- expert per GPU: `{args.expert_per_gpu}`
- top-k: `{args.topk}`
- route probe max inflight: `{args.route_probe_max_inflight}`
- access heat display scale: `{args.access_heat_scale}`
- comparison heatmap snapshot: `{args.placement_heatmap_snapshot}`

## Sweep Grid

- external weights: `{", ".join(f"{value:g}" for value in args.external_weights)}`
- access modes: `{", ".join(args.access_modes)}`
- drift thresholds: `{", ".join(f"{value:g}" for value in args.drift_thresholds)}`
- drift min samples: `{", ".join(str(value) for value in args.drift_min_samples)}`
- async probe intervals ns: `{", ".join(str(value) for value in args.route_probe_intervals_ns)}`
- async probe budgets: `{", ".join(str(value) for value in args.route_probe_budgets)}`
- async probe refresh ns: `{", ".join(str(value) for value in args.route_probe_refresh_ns)}`

## Result Ranking

| method | status | avg TPOT us | delta vs best | delta vs ref | external w | access mode | drift threshold | drift min samples | avg queued flows |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|
{chr(10).join(summary_lines)}

## Artifacts

- `summary.csv`: parsed cross-method metrics.
- `sweep_manifest.csv`: generated parameter grid.
- `<method>/policy.conf`: exact policy config for each run.
- `figures/param_sweep_tpot_rank.png`: ranked TPOT bars.
- `figures/external_weight_tpot.png`: external weight sweep curves by access mode.
- `figures/latency_tpot_bar.png`: raw latency/TPOT comparison.
- `figures/queue_pressure_bar.png`: queue pressure comparison.
- `figures/expert_heatmap_<method>.png`: comparison heatmap.
- `figures/expert_placement_latest_<method>.png`: latest placement-only heatmap.
"""
    (out_dir / "README.md").write_text(text)


def main() -> None:
    args = parse_args()
    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cases, manifest = build_cases(args)
    cases, manifest = selected_cases(args, cases, manifest)

    if args.skip_run:
        rows = full.parse_existing(out_dir, list(cases))
    else:
        rows = [
            full.run_method(args, out_dir, method, settings)
            for method, settings in cases.items()
        ]

    if not rows:
        raise SystemExit("no rows parsed; nothing to plot")

    full.write_summary(out_dir, rows)
    write_manifest(out_dir, manifest)
    ordered_methods = [str(row["method"]) for row in rows]
    full.remove_obsolete_visualizations(out_dir)
    full.plot_metric_bars(out_dir, rows)
    full.plot_relative_tpot(out_dir, rows)
    plot_sweep(out_dir, rows, manifest)
    if args.print_placement_detail:
        full.plot_heatmaps(
            out_dir,
            ordered_methods,
            args.heatmap_clip_percentile,
            args.placement_heatmap_snapshot,
            args.access_heat_scale,
        )
    write_readme(out_dir, args, rows, manifest)
    print(f"summary: {out_dir / 'summary.csv'}")
    print(f"manifest: {out_dir / 'sweep_manifest.csv'}")
    print(f"figures: {out_dir / 'figures'}")
    print(f"readme: {out_dir / 'README.md'}")


if __name__ == "__main__":
    main()
