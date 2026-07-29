#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import run_dispatch_full_compare_visual as full


os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_dispatch_l012_compare")

ROOT = Path(__file__).resolve().parent
DEFAULT_TOPOLOGY = ROOT / "examples" / "HW" / "topo2_2_8.txt"
DEFAULT_BASE_CONFIG = ROOT / "examples" / "HW" / "test.sh"
DEFAULT_OUT = ROOT / "results" / "topo2_2_8_uniform_full_l012_moe58_equiv50x"
ORIGINAL_FULL_BASE_CONFIG = full.base_config

BASELINE_METHOD = "baseline_uniform"
FULL_METHOD = "full"
DISPLAY_METHODS = {
    BASELINE_METHOD: "uniform_trace",
    FULL_METHOD: "full",
}
DEFAULT_COMPARE_METHODS = [BASELINE_METHOD, FULL_METHOD]
TIER_ORDER = ["local", "L0", "L1", "L2"]
METHOD_ALIASES = {
    "uniform_trace": BASELINE_METHOD,
}

FULL_SETTINGS = {
    "EXPERT_PLACEMENT_POLICY": "probe_balanced_domain_spread",
    "SAME_RANK_ROUTE_POLICY": "probe_rtt_delta",
    "PLACEMENT_ACCESS_MODE": "posterior",
    "ASYNC_ROUTE_PROBE_ENABLE": "1",
    "ASYNC_ROUTE_LIVE_ROUTE": "1",
    "TRACE_USE_DEVICE_PLACEMENT": "0",
    **full.SCORE_V2_PRIORITY_WEIGHTS,
    "PLACEMENT_L1_VARIANCE_WEIGHT": "0.0",
}

METHODS: dict[str, dict[str, str]] = {
    BASELINE_METHOD: dict(full.METHODS["baseline_uniform"]),
    FULL_METHOD: FULL_SETTINGS,
}


def l012_base_config(args: argparse.Namespace) -> dict[str, str]:
    cfg = ORIGINAL_FULL_BASE_CONFIG(args)
    cfg["TRACE_MOE_LAYER_COUNT"] = str(max(0, args.trace_moe_layer_count))
    return cfg


def resolve_sim_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def fmt_label(method: str) -> str:
    return DISPLAY_METHODS.get(method, method)


def safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * p / 100.0
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def make_network_config(base_config: Path, out_path: Path, topology_file: Path) -> None:
    lines = base_config.read_text().splitlines()
    out_lines: list[str] = []
    replaced = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("TOPOLOGY_FILE "):
            out_lines.append(f"TOPOLOGY_FILE {topology_file.resolve()}")
            replaced = True
        else:
            out_lines.append(line)
    if not replaced:
        raise ValueError(f"TOPOLOGY_FILE line not found in {base_config}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines) + "\n")


def read_topology_file_from_config(config_path: Path) -> Path:
    if not config_path.exists():
        raise FileNotFoundError(f"network config not found: {config_path}")
    for line in config_path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("TOPOLOGY_FILE "):
            value = stripped.split(None, 1)[1]
            return resolve_sim_path(value)
    raise ValueError(f"TOPOLOGY_FILE line not found in {config_path}")


def effective_trace_moe_layer_count(args: argparse.Namespace) -> int:
    configured = max(0, int(args.trace_moe_layer_count))
    if configured > 0:
        return configured

    device_path = resolve_sim_path(args.trace_device_file)
    try:
        data = json.loads(device_path.read_text())
    except (OSError, json.JSONDecodeError):
        return 1

    if isinstance(data, dict):
        try:
            moe_layer_count = int(data.get("moe_layer_count", 0) or 0)
        except (TypeError, ValueError):
            moe_layer_count = 0
        if moe_layer_count > 0:
            return moe_layer_count
        layers = data.get("layer_list", [])
    else:
        layers = data
    return max(1, len(layers) if isinstance(layers, list) else 1)


def build_network_inputs(args: argparse.Namespace) -> tuple[Path, Path]:
    explicit_network_config = any(
        arg == "--network-config" or arg.startswith("--network-config=")
        for arg in sys.argv[1:]
    )

    if args.topology_file is not None:
        topology_path = resolve_sim_path(args.topology_file)
    elif explicit_network_config:
        topology_path = read_topology_file_from_config(resolve_sim_path(args.network_config))
    else:
        topology_path = DEFAULT_TOPOLOGY

    if not topology_path.exists():
        raise FileNotFoundError(
            f"topology file not found: {topology_path}. "
            f"Run simulation/examples/HW/gen_topo.py first if needed."
        )

    if explicit_network_config:
        network_config = resolve_sim_path(args.network_config)
        if not network_config.exists():
            raise FileNotFoundError(f"network config not found: {network_config}")
        return topology_path, network_config

    generated = args.output / "network_configs" / f"{topology_path.stem}.conf"
    make_network_config(resolve_sim_path(args.base_network_config), generated, topology_path)
    return topology_path, generated.resolve()


@dataclass
class TopologyInfo:
    node_num: int
    switch_set: set[int]
    gpu_nodes: list[int]
    adjacency: list[list[tuple[int, int]]]
    pair_level: list[list[int]]


def classify_edge(src: int, dst: int, rate: str, delay: str, switch_set: set[int]) -> int:
    if src in switch_set and dst in switch_set:
        if rate.startswith("1600") or delay.startswith("602"):
            return 2
        return 1
    return 0


def load_topology(path: Path) -> TopologyInfo:
    tokens = path.read_text().split()
    if len(tokens) < 7:
        raise ValueError(f"invalid topology header: {path}")

    node_num = int(tokens[0])
    nvswitch_num = int(tokens[2])
    switch_num = int(tokens[3])
    link_num = int(tokens[4])

    idx = 7
    nvswitch_ids = [int(v) for v in tokens[idx: idx + nvswitch_num]]
    idx += nvswitch_num
    switch_ids = [int(v) for v in tokens[idx: idx + switch_num]]
    idx += switch_num
    if (len(tokens) - idx) % 5 != 0:
        raise ValueError(f"invalid topology link payload: {path}")

    switch_set = set(nvswitch_ids) | set(switch_ids)
    gpu_nodes = [i for i in range(node_num) if i not in switch_set]
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(node_num)]

    link_count = 0
    for pos in range(idx, len(tokens), 5):
        src = int(tokens[pos])
        dst = int(tokens[pos + 1])
        rate = tokens[pos + 2]
        delay = tokens[pos + 3]
        level = classify_edge(src, dst, rate, delay, switch_set)
        adjacency[src].append((dst, level))
        adjacency[dst].append((src, level))
        link_count += 1

    if link_count != link_num:
        print(
            f"[warn] topology link count mismatch: header={link_num}, parsed={link_count}",
            flush=True,
        )

    pair_level = [[-1 for _ in range(node_num)] for _ in range(node_num)]
    for src in gpu_nodes:
        dist = [10**9] * node_num
        dist[src] = 0
        heap: list[tuple[int, int]] = [(0, src)]
        while heap:
            cur, node = heapq.heappop(heap)
            if cur != dist[node]:
                continue
            for nxt, level in adjacency[node]:
                nxt_level = max(cur, level)
                if nxt_level < dist[nxt]:
                    dist[nxt] = nxt_level
                    heapq.heappush(heap, (nxt_level, nxt))
        for dst in gpu_nodes:
            pair_level[src][dst] = dist[dst]

    return TopologyInfo(
        node_num=node_num,
        switch_set=switch_set,
        gpu_nodes=gpu_nodes,
        adjacency=adjacency,
        pair_level=pair_level,
    )


def tier_for_pair(src: int, dst: int, topology: TopologyInfo) -> str:
    if src == dst:
        return "local"
    if src < 0 or dst < 0 or src >= topology.node_num or dst >= topology.node_num:
        return "unknown"
    level = topology.pair_level[src][dst]
    return {0: "L0", 1: "L1", 2: "L2"}.get(level, "unknown")


def parse_float(value: str | None) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def parse_int(value: str | None) -> int:
    try:
        return int(float(value or 0.0))
    except (TypeError, ValueError):
        return 0


def parse_flow_fct_ns(row: dict[str, str]) -> float:
    return parse_float(row.get("actual_fct_ns") or row.get("actual_fct"))


def analyze_tiers(run_dir: Path, method: str, topology: TopologyInfo) -> list[dict[str, Any]]:
    flow_path = run_dir / "mncc_flow_finish.csv"
    tier_data: dict[str, dict[str, Any]] = {}

    def bucket(tier: str) -> dict[str, Any]:
        if tier not in tier_data:
            tier_data[tier] = {
                "flow_count": 0,
                "bytes": 0,
                "actuals": [],
                "job_max": defaultdict(float),
            }
        return tier_data[tier]

    if flow_path.exists():
        with flow_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                src = parse_int(row.get("srcNode"))
                dst = parse_int(row.get("dstNode"))
                tier = tier_for_pair(src, dst, topology)
                data = bucket(tier)
                actual = parse_flow_fct_ns(row)
                job_id = row.get("jobId", "")
                data["flow_count"] += 1
                data["bytes"] += parse_int(row.get("msg_size"))
                data["actuals"].append(actual)
                if actual > data["job_max"][job_id]:
                    data["job_max"][job_id] = actual
    else:
        print(f"[warn] missing flow file: {flow_path}", flush=True)

    rows: list[dict[str, Any]] = []
    tiers = TIER_ORDER + [tier for tier in tier_data if tier not in TIER_ORDER]
    for tier in tiers:
        data = tier_data.get(tier, {
            "flow_count": 0,
            "bytes": 0,
            "actuals": [],
            "job_max": defaultdict(float),
        })
        actuals = data["actuals"]
        job_values = list(data["job_max"].values())
        rows.append(
            {
                "method": method,
                "display_method": fmt_label(method),
                "tier": tier,
                "flow_count": data["flow_count"],
                "bytes": data["bytes"],
                "flow_avg_actual_fct_ns": safe_mean(actuals),
                "flow_p95_actual_fct_ns": percentile(actuals, 95.0),
                "flow_max_actual_fct_ns": max(actuals) if actuals else 0.0,
                "job_count": len(job_values),
                "job_mean_tpot_ns": safe_mean(job_values),
                "job_p95_tpot_ns": percentile(job_values, 95.0),
                "job_max_tpot_ns": max(job_values) if job_values else 0.0,
            }
        )
    return rows


def write_csv_file(path: Path, rows: list[dict[str, Any]]) -> None:
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


def plot_matrix_heatmap(
    matrix: list[list[float]],
    row_labels: list[str],
    col_labels: list[str],
    out_path: Path,
    title: str,
    cbar_label: str,
    annotate: Callable[[float], str] | None = None,
    cmap_name: str = "viridis",
    display_matrix: list[list[float]] | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    data = np.array(display_matrix if display_matrix is not None else matrix, dtype=float)
    annotate_data = np.array(matrix, dtype=float)
    masked = np.ma.masked_invalid(data)
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("#e0e0e0")

    fig, ax = plt.subplots(figsize=(max(7.5, len(col_labels) * 1.1), max(4.5, len(row_labels) * 0.55)))
    im = ax.imshow(masked, aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=18, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(title)
    ax.set_xlabel("interconnect tier")
    ax.set_ylabel("method")
    ax.grid(False)

    finite = data[np.isfinite(data)]
    value_max = float(finite.max()) if finite.size else 0.0
    threshold = value_max * 0.6
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = annotate_data[i, j]
            display_value = data[i, j]
            if np.isnan(value):
                text = "n/a"
            elif annotate is None:
                text = f"{value:.2f}"
            else:
                text = annotate(value)
            color = "white" if np.isfinite(display_value) and display_value > threshold else "#111111"
            ax.text(j, i, text, ha="center", va="center", fontsize=9, color=color)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_tier_outputs(out_dir: Path, tier_rows: list[dict[str, Any]], methods: list[str]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    tier_lookup = {(row["method"], row["tier"]): row for row in tier_rows}
    plot_tiers = [tier for tier in TIER_ORDER if any((m, tier) in tier_lookup for m in methods)]
    display_methods = [fmt_label(method) for method in methods]

    tpot_matrix: list[list[float]] = []
    count_matrix: list[list[float]] = []
    avg_fct_matrix: list[list[float]] = []
    count_display_matrix: list[list[float]] = []
    for method in methods:
        tpot_row: list[float] = []
        count_row: list[float] = []
        count_display_row: list[float] = []
        fct_row: list[float] = []
        for tier in plot_tiers:
            row = tier_lookup.get((method, tier))
            if row is None:
                tpot_row.append(float("nan"))
                count_row.append(float("nan"))
                count_display_row.append(float("nan"))
                fct_row.append(float("nan"))
            else:
                tpot_row.append(float(row["job_mean_tpot_ns"]) / 1000.0)
                count_row.append(float(row["flow_count"]))
                count_display_row.append(math.log1p(float(row["flow_count"])))
                fct_row.append(float(row["flow_avg_actual_fct_ns"]) / 1000.0)
        tpot_matrix.append(tpot_row)
        count_matrix.append(count_row)
        count_display_matrix.append(count_display_row)
        avg_fct_matrix.append(fct_row)

    plot_matrix_heatmap(
        tpot_matrix,
        display_methods,
        plot_tiers,
        fig_dir / "tier_job_tpot_heatmap.png",
        "Tiered TPOT proxy by interconnect level",
        "job-critical TPOT (us)",
        annotate=lambda v: f"{v:.1f}",
        cmap_name="magma",
    )
    plot_matrix_heatmap(
        count_matrix,
        display_methods,
        plot_tiers,
        fig_dir / "tier_flow_count_heatmap.png",
        "Flow count by interconnect level",
        "log1p(flow count)",
        annotate=lambda v: f"{int(v)}",
        cmap_name="viridis",
        display_matrix=count_display_matrix,
    )
    plot_matrix_heatmap(
        avg_fct_matrix,
        display_methods,
        plot_tiers,
        fig_dir / "tier_avg_flow_fct_heatmap.png",
        "Mean flow completion time by interconnect level",
        "mean actual_fct (us)",
        annotate=lambda v: f"{v:.1f}",
        cmap_name="plasma",
    )

    tiers = plot_tiers
    x = np.arange(len(tiers))
    width = min(0.32, 0.8 / max(1, len(methods)))
    fig, ax = plt.subplots(figsize=(max(8.5, len(tiers) * 1.4), 5.2))
    for idx, method in enumerate(methods):
        values = [
            float(tier_lookup.get((method, tier), {}).get("job_mean_tpot_ns", 0.0)) / 1000.0
            for tier in tiers
        ]
        offset = (idx - (len(methods) - 1) / 2.0) * width
        bars = ax.bar(x + offset, values, width, label=fmt_label(method))
        for bar, value in zip(bars, values):
            if value > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height(),
                    f"{value:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
    ax.set_xticks(x)
    ax.set_xticklabels(tiers)
    ax.set_ylabel("job-critical TPOT (us)")
    ax.set_title("Tiered TPOT proxy comparison")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "tier_job_tpot_bar.png", dpi=180)
    plt.close(fig)

def plot_relative_to_uniform(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    baseline = next((row for row in rows if row.get("method") == BASELINE_METHOD), None)
    if baseline is None:
        return
    baseline_tpot = float(baseline.get("average_TPOT_ns", 0.0) or 0.0)
    if baseline_tpot <= 0.0:
        return

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    labels = [fmt_label(str(row["method"])) for row in rows]
    values = [
        float(row.get("average_TPOT_ns", 0.0) or 0.0) / baseline_tpot
        for row in rows
    ]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    colors = [
        "#8a8a8a" if row.get("method") == BASELINE_METHOD else "#2ca25f"
        for row in rows
    ]
    bars = ax.bar(np.arange(len(rows)), values, color=colors)
    ax.axhline(1.0, color="#444444", linestyle="--", linewidth=1)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("TPOT / uniform_trace")
    ax.set_title("Relative TPOT")
    ax.grid(True, axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        if value > 0.0:
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{value:.2f}x",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    fig.tight_layout()
    fig.savefig(fig_dir / "tpot_relative_to_uniform_trace.png", dpi=180)
    plt.close(fig)


def selected_methods(args: Any) -> dict[str, dict[str, str]]:
    if args.methods == ["all"]:
        return METHODS
    canonical_methods: list[str] = []
    for name in args.methods:
        canonical = METHOD_ALIASES.get(name, name)
        if canonical not in canonical_methods:
            canonical_methods.append(canonical)
    unknown = [name for name in canonical_methods if name not in METHODS]
    if unknown:
        valid = ", ".join(list(METHODS) + list(METHOD_ALIASES))
        raise SystemExit(f"unknown methods: {', '.join(unknown)}; valid: {valid}")
    return {name: METHODS[name] for name in canonical_methods}


def write_readme(
    out_dir: Path,
    args: Any,
    rows: list[dict[str, Any]],
    tier_rows: list[dict[str, Any]],
    topology: Path,
) -> None:
    methods = ", ".join(fmt_label(str(row["method"])) for row in rows)
    text = f"""# Uniform vs Full on New Topology

Generated by `simulation/run_dispatch_l012_topology_compare_visual.py`.

Methods: {methods}

## Workload

- topology file: `{topology}`
- network config: `{args.network_config}`
- trace dispatch: `{args.trace_dispatch}`
- trace placement layer id: `{args.trace_layer_id}`
- trace MoE layer count: `{args.trace_moe_layer_count}` (`0` means auto from `device.json`)
- trace access divisor: `{100 if args.trace_access_100x else max(1, args.trace_access_divisor)}`
- dispatch need: `{args.dispatch_need}`
- expert num: `{args.expert_num}`
- expert per GPU: `{args.expert_per_gpu}`
- top-k: `{args.topk}`
- simulation stop time: `{args.stop_time}`

## Method Mapping

- `uniform_trace` = `baseline_uniform`, using trace device placement and uniform per-layer access targets.
- `full` = posterior placement + `probe_balanced_domain_spread` + `probe_rtt_delta` + async live route + score-v2 weights, with `PLACEMENT_L1_VARIANCE_WEIGHT=0`

Trace dispatch consumes decode targets in MoE layer order. With the default setting this is 58 layers: `0,1,...,57,0,1,...`; each layer keeps its own total target count after `TRACE_ACCESS_DIVISOR` scaling.

## Tier Definition

The script classifies each flow by the minimax path tier on the topology graph:

- `local`: `srcNode == dstNode`
- `L0`: only level-0 links on the best path
- `L1`: at least one level-1 switch-link, no `L2`
- `L2`: path uses the upper-tier `L2` interconnect

Edge levels come from `gen_topo.py`:

- `0ns` host / tray links -> `L0`
- `400Gbps` + `850ns` switch links -> `L1`
- `1600Gbps` + `602ns` upper-switch links -> `L2`

Tiered TPOT in the plots is computed as the per-job maximum flow FCT inside each tier, then averaged across jobs. New runs read `actual_fct_ns`; old CSV files with `actual_fct` are still accepted as ns.

## Result Summary

| method | status | completed tasks | avg latency ns | avg TPOT ns |
|---|---|---:|---:|---:|
{chr(10).join(
    f"| {fmt_label(str(row.get('method', '')))} | {row.get('status', '')} | {row.get('completed_tasks', 0)} | {row.get('average_latency_ns', 0)} | {row.get('average_TPOT_ns', 0)} |"
    for row in rows
)}

## Tier Summary

| method | tier | flow count | bytes | job TPOT ns |
|---|---|---:|---:|---:|
{chr(10).join(
    f"| {fmt_label(str(row.get('method', '')))} | {row.get('tier', '')} | {row.get('flow_count', 0)} | {row.get('bytes', 0)} | {row.get('job_mean_tpot_ns', 0):.1f} |"
    for row in tier_rows
)}

## Artifacts

- `summary.csv`: overall parsed metrics.
- `tier_summary.csv`: L0/L1/L2 tiered flow and TPOT breakdown.
- `figures/expert_heatmap_<method>.png`: placement/access heatmap.
- `figures/expert_placement_latest_<method>.png`: latest placement-only heatmap.
- `figures/tier_job_tpot_heatmap.png`: TPOT heatmap by tier.
- `figures/tier_flow_count_heatmap.png`: flow-count heatmap by tier.
- `figures/tier_avg_flow_fct_heatmap.png`: mean flow FCT heatmap by tier.
- `figures/tier_job_tpot_bar.png`: tiered TPOT grouped bar chart.
- `figures/tpot_relative_to_uniform_trace.png`: overall TPOT normalized to `uniform_trace`.
"""
    (out_dir / "README.md").write_text(text)


def parse_args() -> Any:
    full.METHODS = METHODS
    full.DEFAULT_COMPARE_METHODS = DEFAULT_COMPARE_METHODS
    full.DEFAULT_OUT = DEFAULT_OUT
    full.PARSER_DESCRIPTION = (
        "Run uniform_trace vs full on the new topology with 58-layer trace dispatch and tiered TPOT plots."
    )
    full.base_config = l012_base_config
    parser = full.build_parser()
    parser.add_argument(
        "--topology-file",
        type=Path,
        default=None,
        help="Topology file used for tier classification and network config generation.",
    )
    parser.add_argument(
        "--base-network-config",
        type=Path,
        default=DEFAULT_BASE_CONFIG,
        help="Base network config template used to generate the run config.",
    )
    parser.add_argument(
        "--trace-moe-layer-count",
        type=int,
        default=58,
        help="Number of MoE decode layers consumed in order. 0 means infer from device.json.",
    )
    args = parser.parse_args()
    explicit_trace_divisor = any(
        arg == "--trace-access-divisor" or arg.startswith("--trace-access-divisor=")
        for arg in sys.argv[1:]
    ) or "--trace-access-100x" in sys.argv
    args.trace_access_divisor_explicit = explicit_trace_divisor
    if not explicit_trace_divisor:
        layer_count = effective_trace_moe_layer_count(args)
        args.trace_access_divisor = 50 * layer_count
    return args


def main() -> None:
    args = parse_args()
    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    args.output = out_dir

    topology_path, network_config = build_network_inputs(args)
    args.network_config = str(network_config if network_config.is_absolute() else network_config.resolve())

    methods = selected_methods(args)
    effective_layers = effective_trace_moe_layer_count(args)
    effective_divisor = 100 if args.trace_access_100x else max(1, args.trace_access_divisor)
    print(
        "[info] topology={topology} methods={methods} trace_moe_layers={layers} "
        "trace_access_divisor={divisor}".format(
            topology=topology_path,
            methods=",".join(fmt_label(method) for method in methods),
            layers=effective_layers,
            divisor=effective_divisor,
        ),
        flush=True,
    )
    equiv50_divisor = 50 * max(1, effective_layers)
    if args.trace_access_divisor_explicit and effective_layers > 1 and effective_divisor < equiv50_divisor:
        print(
            "[warn] divisor is lower than 50 * trace_moe_layers; this runs more traffic than "
            f"the old decode0 50x workload. Use --trace-access-divisor {equiv50_divisor} "
            "for an equivalent total scale.",
            flush=True,
        )
    if args.skip_run:
        rows = full.parse_existing(out_dir, list(methods))
    else:
        rows = [full.run_method(args, out_dir, method, settings) for method, settings in methods.items()]

    if not rows:
        raise SystemExit("no rows parsed; nothing to plot")

    topology = load_topology(topology_path)
    tier_rows: list[dict[str, Any]] = []
    for method in methods:
        tier_rows.extend(analyze_tiers(out_dir / method, method, topology))

    full.write_summary(out_dir, rows)
    write_csv_file(out_dir / "tier_summary.csv", tier_rows)
    full.remove_obsolete_visualizations(out_dir)
    full.plot_metric_bars(out_dir, rows)
    plot_relative_to_uniform(out_dir, rows)
    plot_tier_outputs(out_dir, tier_rows, list(methods))
    if args.print_placement_detail:
        full.plot_heatmaps(
            out_dir,
            list(methods),
            args.heatmap_clip_percentile,
            args.placement_heatmap_snapshot,
            args.access_heat_scale,
        )
    write_readme(out_dir, args, rows, tier_rows, topology_path)
    print(f"summary: {out_dir / 'summary.csv'}")
    print(f"tier_summary: {out_dir / 'tier_summary.csv'}")
    print(f"figures: {out_dir / 'figures'}")
    print(f"readme: {out_dir / 'README.md'}")


if __name__ == "__main__":
    main()
