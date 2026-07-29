#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_dispatch_full_compare")

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "results" / "hw_full_async_compare"
ARTIFACTS = ("mncc.log", "mncc_flow_finish.csv", "mncc_cluster_timeseries.csv")
DEFAULT_COMPARE_METHODS = ["baseline_uniform", "baseline_trace", "posterior_async_probe"]
PARSER_DESCRIPTION = "Run full dispatch policy comparisons and generate latency, queue, and expert heatmap plots."


COMMON_PROBE_WEIGHTS = {
    "PLACEMENT_EXTERNAL_WEIGHT": "3.0",
    "PLACEMENT_REPLICA_BALANCE_WEIGHT": "8.0",
    "PLACEMENT_L1_VARIANCE_WEIGHT": "1.0",
    "PLACEMENT_L1_MAX_WEIGHT": "2.0",
    "PLACEMENT_GPU_VARIANCE_WEIGHT": "2.0",
    "PLACEMENT_PROBE_WEIGHT": "0.25",
    "PLACEMENT_PROBE_DELTA_WEIGHT": "0.5",
    "PLACEMENT_PROBE_RTT_WEIGHT": "0.10",
}

SCORE_V2_PRIORITY_WEIGHTS = {
    "PLACEMENT_REPLICA_BALANCE_WEIGHT": "6.0",
    "PLACEMENT_EXTERNAL_WEIGHT": "8.0",
    "PLACEMENT_L1_VARIANCE_WEIGHT": "1.0",
    "PLACEMENT_L1_MAX_WEIGHT": "1.5",
    "PLACEMENT_GPU_VARIANCE_WEIGHT": "0.5",
    "PLACEMENT_PROBE_WEIGHT": "7.0",
}


METHODS: dict[str, dict[str, str]] = {
    "baseline_uniform": {
        "EXPERT_PLACEMENT_POLICY": "default",
        "SAME_RANK_ROUTE_POLICY": "default",
        "PLACEMENT_ACCESS_MODE": "initial",
        "TRACE_USE_DEVICE_PLACEMENT": "1",
        "TRACE_ACCESS_TARGETS_AS_EXPERT_FREQ": "0",
        "TRACE_UNIFORM_ACCESS_TARGETS": "1",
    },
    "baseline_trace": {
        "EXPERT_PLACEMENT_POLICY": "default",
        "SAME_RANK_ROUTE_POLICY": "default",
        "PLACEMENT_ACCESS_MODE": "initial",
        "TRACE_USE_DEVICE_PLACEMENT": "1",
        "TRACE_ACCESS_TARGETS_AS_EXPERT_FREQ": "1",
        "TRACE_UNIFORM_ACCESS_TARGETS": "0",
    },
    "posterior_rtt": {
        "EXPERT_PLACEMENT_POLICY": "probe_balanced_domain_spread",
        "SAME_RANK_ROUTE_POLICY": "probe_rtt_delta",
        "PLACEMENT_ACCESS_MODE": "posterior",
        "TRACE_USE_DEVICE_PLACEMENT": "0",
        **COMMON_PROBE_WEIGHTS,
    },
    "posterior_domain_base": {
        "EXPERT_PLACEMENT_POLICY": "probe_balanced_domain_spread_base",
        "SAME_RANK_ROUTE_POLICY": "probe_rtt_delta",
        "PLACEMENT_ACCESS_MODE": "posterior",
        "TRACE_USE_DEVICE_PLACEMENT": "0",
        **COMMON_PROBE_WEIGHTS,
    },
    "posterior_hot_surplus": {
        "EXPERT_PLACEMENT_POLICY": "probe_balanced_hot_surplus",
        "SAME_RANK_ROUTE_POLICY": "probe_rtt_delta",
        "PLACEMENT_ACCESS_MODE": "posterior",
        "TRACE_USE_DEVICE_PLACEMENT": "0",
        **COMMON_PROBE_WEIGHTS,
    },
    "posterior_domain_spread": {
        "EXPERT_PLACEMENT_POLICY": "probe_balanced_domain_spread",
        "SAME_RANK_ROUTE_POLICY": "probe_rtt_delta",
        "PLACEMENT_ACCESS_MODE": "posterior",
        "TRACE_USE_DEVICE_PLACEMENT": "0",
        **COMMON_PROBE_WEIGHTS,
    },
    "posterior_async_probe": {
        "EXPERT_PLACEMENT_POLICY": "probe_balanced_domain_spread",
        "SAME_RANK_ROUTE_POLICY": "probe_rtt_delta",
        "PLACEMENT_ACCESS_MODE": "posterior",
        "ASYNC_ROUTE_PROBE_ENABLE": "1",
        "ASYNC_ROUTE_LIVE_ROUTE": "1",
        "TRACE_USE_DEVICE_PLACEMENT": "0",
        **COMMON_PROBE_WEIGHTS,
    },
    "posterior_async_probe_score_v2": {
        "EXPERT_PLACEMENT_POLICY": "probe_balanced_domain_spread",
        "SAME_RANK_ROUTE_POLICY": "probe_rtt_delta",
        "PLACEMENT_ACCESS_MODE": "posterior",
        "ASYNC_ROUTE_PROBE_ENABLE": "1",
        "ASYNC_ROUTE_LIVE_ROUTE": "1",
        "TRACE_USE_DEVICE_PLACEMENT": "0",
        **SCORE_V2_PRIORITY_WEIGHTS,
    },
    "posterior_probe_cost": {
        "EXPERT_PLACEMENT_POLICY": "probe_balanced_probe_cost",
        "SAME_RANK_ROUTE_POLICY": "probe_rtt_delta",
        "PLACEMENT_ACCESS_MODE": "posterior",
        "TRACE_USE_DEVICE_PLACEMENT": "0",
        **COMMON_PROBE_WEIGHTS,
    },
    "posterior_topology_diverse": {
        "EXPERT_PLACEMENT_POLICY": "probe_balanced_topology_diverse",
        "SAME_RANK_ROUTE_POLICY": "probe_rtt_delta",
        "PLACEMENT_ACCESS_MODE": "posterior",
        "TRACE_USE_DEVICE_PLACEMENT": "0",
        **COMMON_PROBE_WEIGHTS,
    },
    "posterior_queue025": {
        "EXPERT_PLACEMENT_POLICY": "probe_balanced_full",
        "SAME_RANK_ROUTE_POLICY": "probe_rtt_delta",
        "PLACEMENT_ACCESS_MODE": "posterior",
        "ROUTE_QUEUE_WEIGHT": "0.25",
        "ROUTE_QUEUE_NORM": "256",
        "TRACE_USE_DEVICE_PLACEMENT": "0",
        **COMMON_PROBE_WEIGHTS,
    },
    "posterior_queue100": {
        "EXPERT_PLACEMENT_POLICY": "probe_balanced_full",
        "SAME_RANK_ROUTE_POLICY": "probe_rtt_delta",
        "PLACEMENT_ACCESS_MODE": "posterior",
        "ROUTE_QUEUE_WEIGHT": "1.0",
        "ROUTE_QUEUE_NORM": "256",
        "TRACE_USE_DEVICE_PLACEMENT": "0",
        **COMMON_PROBE_WEIGHTS,
    },
    "initial_static_rtt": {
        "EXPERT_PLACEMENT_POLICY": "probe_balanced_full",
        "SAME_RANK_ROUTE_POLICY": "probe_rtt_delta",
        "PLACEMENT_ACCESS_MODE": "initial",
        "TRACE_USE_DEVICE_PLACEMENT": "0",
        **COMMON_PROBE_WEIGHTS,
    },
    "drift_hot_update_rtt": {
        "EXPERT_PLACEMENT_POLICY": "probe_balanced_full",
        "SAME_RANK_ROUTE_POLICY": "probe_rtt_delta",
        "PLACEMENT_ACCESS_MODE": "initial_until_drift",
        "PLACEMENT_DRIFT_THRESHOLD": "0.30",
        "PLACEMENT_DRIFT_MIN_SAMPLES": "4096",
        "TRACE_USE_DEVICE_PLACEMENT": "0",
        **COMMON_PROBE_WEIGHTS,
    },
    "drift_hot_update_queue": {
        "EXPERT_PLACEMENT_POLICY": "probe_balanced_full",
        "SAME_RANK_ROUTE_POLICY": "probe_rtt_delta",
        "PLACEMENT_ACCESS_MODE": "initial_until_drift",
        "PLACEMENT_DRIFT_THRESHOLD": "0.30",
        "PLACEMENT_DRIFT_MIN_SAMPLES": "4096",
        "ROUTE_QUEUE_WEIGHT": "0.25",
        "ROUTE_QUEUE_NORM": "256",
        "TRACE_USE_DEVICE_PLACEMENT": "0",
        **COMMON_PROBE_WEIGHTS,
    },
}


def smooth_freq(count: int, high_first: bool) -> str:
    if count <= 1:
        return "10.0000"
    values: list[str] = []
    for i in range(count):
        ratio = i / float(count - 1)
        value = 10.0 - 9.0 * ratio if high_first else 1.0 + 9.0 * ratio
        values.append(f"{value:.4f}")
    return " ".join(values)


def uniform_freq(count: int) -> str:
    return " ".join("1.0000" for _ in range(max(1, count)))


def sim_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def trace_device_template_info(args: argparse.Namespace) -> dict[str, int | str]:
    path = sim_path(args.trace_device_file)
    info: dict[str, int | str] = {
        "path": str(path),
        "layer": args.trace_layer_id,
        "devices": 0,
        "slots": 0,
        "experts_per_device_min": 0,
        "experts_per_device_max": 0,
    }
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return info

    layers = data.get("layer_list", []) if isinstance(data, dict) else data
    if not isinstance(layers, list):
        return info
    target = None
    for layer in layers:
        if isinstance(layer, dict) and int(layer.get("layer_id", -1)) == args.trace_layer_id:
            target = layer
            break
    if target is None and 0 <= args.trace_layer_id < len(layers):
        target = layers[args.trace_layer_id]
    if not isinstance(target, dict):
        return info

    devices = target.get("device_list", [])
    if not isinstance(devices, list):
        return info
    counts: list[int] = []
    for device in devices:
        if not isinstance(device, dict):
            continue
        experts = device.get("device_expert", [])
        if isinstance(experts, list):
            counts.append(len(experts))
    info["devices"] = len(counts)
    info["slots"] = sum(counts)
    info["experts_per_device_min"] = min(counts) if counts else 0
    info["experts_per_device_max"] = max(counts) if counts else 0
    return info


def payload_params(args: argparse.Namespace) -> int:
    if args.dispatch_expert_ffn_params is not None:
        return args.dispatch_expert_ffn_params
    if args.payload_preset == "deepseek-v3-1b":
        return 7_168
    return 262_144


def base_config(args: argparse.Namespace) -> dict[str, str]:
    switch_task = args.access_switch_task
    if switch_task < 0:
        switch_task = max(1, args.num_tasks // 2)
    trace_access_divisor = 100 if args.trace_access_100x else args.trace_access_divisor
    trace_access_divisor = max(1, trace_access_divisor)
    return {
        "ENABLE_POLICY_GROUP": "1",
        "PRINT_SUBMISSIONS": "0",
        "PRINT_EXPERT_PLACEMENT_DETAIL": "1" if args.print_placement_detail else "0",
        "SIMULATION_STOP_TIME": f"{args.stop_time:.9g}",
        "PG": str(args.pg),
        "DISPATCH_NEED": str(args.dispatch_need),
        "EXPERT_NUM": str(args.expert_num),
        "EXPERT_PER_GPU": str(args.expert_per_gpu),
        "DISPATCH_TOPK": str(args.topk),
        "EXPERT_MEM_BYTES": str(args.expert_mem_bytes),
        "CLUSTER_MONITOR_INTERVAL_NS": str(args.monitor_interval_ns),
        "DISPATCH_N": str(args.dispatch_n),
        "DISPATCH_EXPERT_FFN_PARAMS": str(payload_params(args)),
        "DISPATCH_PRECISION_BYTES": str(args.precision_bytes),
        "DISPATCH_BATCH_SIZE": str(args.batch_size),
        "NEED_PLACEMENT_POLICY": "default",
        "LOCAL_FLOW_SCHEDULE_POLICY": args.local_flow_schedule_policy,
        "GLOBAL_NEED_UPDATE_POLICY": "default",
        "PD_SPLIT_POLICY": "default",
        "NUM_TASKS": str(args.num_tasks),
        "TASK_SEED": str(args.task_seed),
        "FIRST_SUBMIT_TIME": f"{args.first_submit_time:.9g}",
        "SUBMIT_INTERVAL": f"{args.submit_interval:.9g}",
        "EXPERT_ACCESS_SWITCH_TASK": str(switch_task),
        "PROBE_HIGH_PG": str(args.probe_high_pg),
        "PROBE_LOW_PG": str(args.probe_low_pg),
        "PROBE_BYTES": str(args.probe_bytes),
        "ROUTE_PROBE_MAX_INFLIGHT": str(args.route_probe_max_inflight),
        "ASYNC_ROUTE_PROBE_ENABLE": "0",
        "ASYNC_ROUTE_LIVE_ROUTE": "0",
        "ASYNC_ROUTE_PROBE_INTERVAL_NS": str(args.async_route_probe_interval_ns),
        "ASYNC_ROUTE_PROBE_BUDGET": str(args.async_route_probe_budget),
        "ASYNC_ROUTE_PROBE_REFRESH_NS": str(args.async_route_probe_refresh_ns),
        "ASYNC_ROUTE_PROBE_TOPK": str(args.async_route_probe_topk),
        "EXPERT_ACCESS_FREQ": smooth_freq(args.expert_num, high_first=True),
        "EXPERT_ACCESS_FREQ_AFTER": smooth_freq(args.expert_num, high_first=False),
        "TRACE_DISPATCH_ENABLE": "1" if args.trace_dispatch else "0",
        "TRACE_LAYER_ID": str(args.trace_layer_id),
        "TRACE_ACCESS_DIVISOR": str(trace_access_divisor),
        "TRACE_DEVICE_FILE": args.trace_device_file,
        "TRACE_DECODE_DIR": args.trace_decode_dir,
        "TRACE_STOP_ON_COMPLETE": "1" if args.trace_stop_on_complete else "0",
        "TRACE_USE_DEVICE_PLACEMENT": "0",
        "TRACE_ACCESS_TARGETS_AS_EXPERT_FREQ": "1",
        "TRACE_UNIFORM_ACCESS_TARGETS": "0",
        "TRACE_UNIFORM_ACCESS_TARGET_COUNT": str(args.trace_uniform_access_target_count),
    }


def config_text(args: argparse.Namespace, method_settings: dict[str, str]) -> str:
    cfg = base_config(args)
    cfg.update(method_settings)
    return "\n".join(f"{key} {value}" for key, value in cfg.items()) + "\n"


def settings_for_method(
    args: argparse.Namespace,
    method: str,
    method_settings: dict[str, str],
) -> dict[str, str]:
    settings = dict(method_settings)
    if args.trace_placement_mode == "all":
        settings["TRACE_USE_DEVICE_PLACEMENT"] = "1"
    elif args.trace_placement_mode == "none":
        settings["TRACE_USE_DEVICE_PLACEMENT"] = "0"
    else:
        settings["TRACE_USE_DEVICE_PLACEMENT"] = "1" if method.startswith("baseline") else "0"
    if method == "baseline_uniform":
        settings["EXPERT_ACCESS_FREQ"] = uniform_freq(args.expert_num)
        settings["EXPERT_ACCESS_FREQ_AFTER"] = uniform_freq(args.expert_num)
    return settings


def clean_current_outputs() -> None:
    for filename in ARTIFACTS:
        path = ROOT / filename
        if path.exists():
            path.unlink()


def copy_artifacts(run_dir: Path) -> None:
    for filename in ARTIFACTS:
        src = ROOT / filename
        if src.exists():
            shutil.copy2(src, run_dir / filename)


def run_method(
    args: argparse.Namespace,
    out_dir: Path,
    method: str,
    method_settings: dict[str, str],
) -> dict[str, Any]:
    run_dir = out_dir / method
    run_dir.mkdir(parents=True, exist_ok=True)
    network_path = ROOT / args.network_config
    if not network_path.exists():
        raise FileNotFoundError(f"network config not found: {network_path}")
    outputs_dir = network_path.parent / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    clean_current_outputs()
    cfg_path = run_dir / "policy.conf"
    cfg_path.write_text(config_text(args, settings_for_method(args, method, method_settings)))

    command = [
        "./ns3",
        "run",
        f"scratch/NormalNetwork --networkConfig={args.network_config} --policyConfig={cfg_path}",
    ]
    print(f"[run] {method}", flush=True)
    start = time.time()
    status = "ok"
    try:
        subprocess.run(command, cwd=ROOT, check=True)
    except subprocess.CalledProcessError as exc:
        status = f"failed:{exc.returncode}"
        if not args.keep_going:
            raise
    wall_time = time.time() - start
    copy_artifacts(run_dir)
    row = parse_run(run_dir, method)
    row["status"] = status
    row["wall_time_s"] = f"{wall_time:.3f}"
    return row


def re_float(pattern: str, text: str, default: float = 0.0) -> float:
    match = re.search(pattern, text)
    if not match:
        return default
    try:
        return float(match.group(1))
    except ValueError:
        return default


def re_int(pattern: str, text: str, default: int = 0) -> int:
    return int(re_float(pattern, text, float(default)))


def re_int_last(pattern: str, text: str, default: int = 0) -> int:
    matches = re.findall(pattern, text)
    if not matches:
        return default
    value = matches[-1]
    if isinstance(value, tuple):
        value = value[-1]
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * p / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    frac = rank - lower
    return ordered[lower] * (1.0 - frac) + ordered[upper] * frac


def load_timeseries(csv_path: Path) -> list[dict[str, float]]:
    if not csv_path.exists():
        return []
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    out: list[dict[str, float]] = []
    for row in rows:
        parsed: dict[str, float] = {}
        for key, value in row.items():
            try:
                parsed[key] = float(value)
            except (TypeError, ValueError):
                parsed[key] = 0.0
        if "total_gpu_mem_bytes" in parsed and parsed["total_gpu_mem_bytes"] > 0:
            parsed["gpu_mem_utilization"] = (
                parsed.get("used_gpu_mem_bytes", 0.0) / parsed["total_gpu_mem_bytes"]
            )
        out.append(parsed)
    return out


def summarize_timeseries(rows: list[dict[str, float]]) -> dict[str, float]:
    summary: dict[str, float] = {"timeseries_samples": float(len(rows))}
    fields = [
        "queued_local_flows",
        "active_local_gpus",
        "gpu_utilization",
        "gpu_mem_utilization",
        "expert_slot_utilization",
        "expert_fragmentation",
        "placement_gpu_utilization",
        "used_gpu_mem_bytes",
        "cn_fragmentation",
        "sched_success_rate",
        "fragment_block_rate",
    ]
    for field in fields:
        values = [row.get(field, 0.0) for row in rows if field in row]
        if not values:
            summary[f"{field}_avg"] = 0.0
            summary[f"{field}_p95"] = 0.0
            summary[f"{field}_max"] = 0.0
            continue
        summary[f"{field}_avg"] = sum(values) / len(values)
        summary[f"{field}_p95"] = percentile(values, 95.0)
        summary[f"{field}_max"] = max(values)
    return summary


def parse_run(run_dir: Path, method: str) -> dict[str, Any]:
    log_path = run_dir / "mncc.log"
    log_text = log_path.read_text(errors="ignore") if log_path.exists() else ""
    total_routes = 0
    local_routes = 0
    for match in re.finditer(r"routes=(\d+) local_routes=(\d+)", log_text):
        total_routes += int(match.group(1))
        local_routes += int(match.group(2))
    access_drifts = [
        float(match.group(1))
        for match in re.finditer(r"access_drift=([0-9.eE+-]+)", log_text)
    ]

    flow_rows = 0
    flow_path = run_dir / "mncc_flow_finish.csv"
    if flow_path.exists():
        with flow_path.open() as f:
            flow_rows = max(0, sum(1 for _ in f) - 1)

    ts_summary = summarize_timeseries(load_timeseries(run_dir / "mncc_cluster_timeseries.csv"))

    row: dict[str, Any] = {
        "method": method,
        "status": "parsed",
        "completed_tasks": re_int(r"completed_tasks=(\d+)", log_text),
        "average_latency_ns": re_int(r"average_latency_ns=(\d+)", log_text),
        "average_TPOT_ns": re_int(r"average_TPOT_ns=(\d+)", log_text),
        "submitted_collectives": sum(1 for _ in re.finditer(r"submitted alltoall", log_text)),
        "flow_rows": flow_rows,
        "data_submitted_flows": re_int_last(r"data_submitted_flows=(\d+)", log_text),
        "data_submitted_bytes": re_int_last(r"data_submitted_bytes=(\d+)", log_text),
        "probe_submitted_flows": re_int_last(r"probe_submitted_flows=(\d+)", log_text),
        "probe_submitted_bytes": re_int_last(r"probe_submitted_bytes=(\d+)", log_text),
        "total_submitted_flows": re_int_last(r"total_submitted_flows=(\d+)", log_text),
        "total_submitted_bytes": re_int_last(r"total_submitted_bytes=(\d+)", log_text),
        "probe_submitted_flow_ratio": re_float(
            r"probe_submitted_flow_ratio=([0-9.eE+-]+)", log_text
        ),
        "probe_submitted_byte_ratio": re_float(
            r"probe_submitted_byte_ratio=([0-9.eE+-]+)", log_text
        ),
        "probe_submitted_percent": re_float(
            r"probe_submitted_percent=([0-9.eE+-]+)", log_text
        ),
        "probe_finished_flows": re_int_last(r"probe_finished_flows=(\d+)", log_text),
        "probe_finished_bytes": re_int_last(r"probe_finished_bytes=(\d+)", log_text),
        "probe_inflight": re_int_last(r"probe_inflight=(\d+)", log_text),
        "local_route_ratio": local_routes / total_routes if total_routes else 0.0,
        "drift_hot_updates": sum(1 for _ in re.finditer(r"drift_hot_update=1", log_text)),
        "max_access_drift": max(access_drifts) if access_drifts else 0.0,
        "trace_completed": re_int_last(r"completed=(\d+)", log_text),
        "trace_submitted_tasks": re_int_last(r"submitted_trace_tasks=(\d+)", log_text),
        "trace_target_accesses": re_int_last(r"target_accesses=(\d+)", log_text),
        "trace_observed_accesses": re_int_last(r"observed_accesses=(\d+)", log_text),
        "trace_remaining_accesses": re_int_last(r"remaining_accesses=(\d+)", log_text),
        "trace_unmet_experts": re_int_last(r"unmet_experts=(\d+)", log_text),
        "trace_max_expert_remaining": re_int_last(r"max_expert_remaining=(\d+)", log_text),
        "log_path": str(log_path),
    }
    row.update(ts_summary)
    return row


def write_summary(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with (out_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    def best_key(row: dict[str, Any]) -> tuple[float, float]:
        completed = float(row.get("completed_tasks", 0) or 0)
        tpot = float(row.get("average_TPOT_ns", 0) or 0)
        trace_target = float(row.get("trace_target_accesses", 0) or 0)
        trace_completed = float(row.get("trace_completed", 0) or 0)
        if completed <= 0 or tpot <= 0 or (trace_target > 0 and trace_completed <= 0):
            tpot = float("inf")
        return (tpot, float(row.get("queued_local_flows_avg", 0) or 0))

    best = sorted(rows, key=best_key)[0]
    (out_dir / "best_case.txt").write_text(str(best) + "\n")


def plot_metric_bars(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    labels = [str(row["method"]) for row in rows]
    x = np.arange(len(labels))
    width = 0.36

    latency_us = [float(row.get("average_latency_ns", 0.0)) / 1000.0 for row in rows]
    tpot_us = [float(row.get("average_TPOT_ns", 0.0)) / 1000.0 for row in rows]
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.8), 5))
    ax.bar(x - width / 2, latency_us, width, label="avg latency (us)")
    ax.bar(x + width / 2, tpot_us, width, label="avg TPOT (us)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("us")
    ax.set_title("Dispatch latency and TPOT")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "latency_tpot_bar.png", dpi=170)
    plt.close(fig)

    q_avg = [float(row.get("queued_local_flows_avg", 0.0)) for row in rows]
    q_p95 = [float(row.get("queued_local_flows_p95", 0.0)) for row in rows]
    q_max = [float(row.get("queued_local_flows_max", 0.0)) for row in rows]
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.8), 5))
    ax.bar(x - width, q_avg, width, label="avg queued flows")
    ax.bar(x, q_p95, width, label="p95 queued flows")
    ax.bar(x + width, q_max, width, label="max queued flows")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("flows")
    ax.set_title("Local queue pressure")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "queue_pressure_bar.png", dpi=170)
    plt.close(fig)


def plot_relative_tpot(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    baseline_row = next((row for row in rows if row.get("method") == "baseline_trace"), None)
    if baseline_row is None:
        return
    baseline_tpot = float(baseline_row.get("average_TPOT_ns", 0.0) or 0.0)
    if baseline_tpot <= 0.0:
        return

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    labels = [str(row["method"]) for row in rows]
    ratios = [
        float(row.get("average_TPOT_ns", 0.0) or 0.0) / baseline_tpot
        for row in rows
    ]
    colors = [
        "#7f8c8d" if label.startswith("baseline") else "#2ca25f"
        for label in labels
    ]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.9), 5))
    bars = ax.bar(x, ratios, color=colors)
    ax.axhline(1.0, color="#555555", linestyle="--", linewidth=1, label="baseline_trace")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("TPOT / baseline_trace")
    ax.set_title("Relative TPOT")
    ax.grid(True, axis="y", alpha=0.25)
    for bar, ratio in zip(bars, ratios):
        if ratio <= 0.0:
            label = "n/a"
        else:
            improvement = (1.0 - ratio) * 100.0
            label = f"{ratio:.2f}x\n{improvement:+.1f}%"
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            label,
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "tpot_relative_to_baseline_trace.png", dpi=170)
    plt.close(fig)


def remove_obsolete_visualizations(out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    if not fig_dir.exists():
        return
    obsolete = list(fig_dir.glob("time_series_*.png"))
    obsolete.append(fig_dir / "util_fragmentation_bar.png")
    for path in obsolete:
        if path.exists():
            path.unlink()


def plot_heatmaps(
    out_dir: Path,
    methods: list[str],
    clip_percentile: float,
    placement_snapshot: str,
    access_heat_scale: str,
) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_dispatch_full_compare")
    for method in methods:
        run_dir = out_dir / method
        log_path = run_dir / "mncc.log"
        cfg_path = run_dir / "policy.conf"
        if not log_path.exists() or not cfg_path.exists():
            continue
        jobs = [
            (
                fig_dir / f"expert_heatmap_{method}.png",
                placement_snapshot,
                "both",
            ),
            (
                fig_dir / f"expert_placement_latest_{method}.png",
                "latest",
                "placement",
            ),
        ]
        for output, snapshot, view in jobs:
            try:
                subprocess.run(
                    [
                        "python3",
                        str(ROOT / "plot_expert_heatmap.py"),
                        "--log",
                        str(log_path),
                        "--config",
                        str(cfg_path),
                        "--kind",
                        "generic",
                        "--output",
                        str(output),
                        "--clip-percentile",
                        str(clip_percentile),
                        "--snapshot",
                        snapshot,
                        "--view",
                        view,
                        "--access-scale",
                        access_heat_scale,
                    ],
                    cwd=ROOT,
                    env=env,
                    check=True,
                )
            except subprocess.CalledProcessError as exc:
                print(
                    f"[warn] heatmap failed for {method} snapshot={snapshot} view={view}: {exc}",
                    flush=True,
                )


def write_readme(out_dir: Path, args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    methods = ", ".join(str(row["method"]) for row in rows)
    trace_access_divisor = 100 if args.trace_access_100x else max(1, args.trace_access_divisor)
    template = trace_device_template_info(args)
    template_devices = int(template.get("devices", 0) or 0)
    full_repeats = args.dispatch_need // template_devices if template_devices else 0
    remainder = args.dispatch_need % template_devices if template_devices else 0
    method_lines = []
    for row in rows:
        method = str(row["method"])
        settings = settings_for_method(args, method, METHODS.get(method, {}))
        method_lines.append(
            "| {method} | {placement} | {route} | {access} | {trace_device} | {uniform_targets} | {trace_target_freq} | {queue_weight} | {async_probe} | {live_route} |".format(
                method=method,
                placement=settings.get("EXPERT_PLACEMENT_POLICY", "default"),
                route=settings.get("SAME_RANK_ROUTE_POLICY", "default"),
                access=settings.get("PLACEMENT_ACCESS_MODE", "initial"),
                trace_device=settings.get("TRACE_USE_DEVICE_PLACEMENT", "0"),
                uniform_targets=settings.get("TRACE_UNIFORM_ACCESS_TARGETS", "0"),
                trace_target_freq=settings.get("TRACE_ACCESS_TARGETS_AS_EXPERT_FREQ", "1"),
                queue_weight=settings.get("ROUTE_QUEUE_WEIGHT", "0"),
                async_probe=settings.get("ASYNC_ROUTE_PROBE_ENABLE", "0"),
                live_route=settings.get("ASYNC_ROUTE_LIVE_ROUTE", "0"),
            )
        )
    summary_lines = []
    for row in rows:
        summary_lines.append(
            "| {method} | {status} | {completed} | {latency} | {tpot} | {trace_done} | {remaining} | {qavg} |".format(
                method=row.get("method", ""),
                status=row.get("status", ""),
                completed=row.get("completed_tasks", 0),
                latency=row.get("average_latency_ns", 0),
                tpot=row.get("average_TPOT_ns", 0),
                trace_done=row.get("trace_completed", 0),
                remaining=row.get("trace_remaining_accesses", 0),
                qavg=row.get("queued_local_flows_avg", 0),
            )
        )
    text = f"""# Dispatch Full Comparison

Generated by `simulation/run_dispatch_full_compare_visual.py`.

Methods: {methods}

## Workload

- network config: `{args.network_config}`
- trace dispatch: `{args.trace_dispatch}`
- trace layer id: `{args.trace_layer_id}`
- trace access divisor: `{trace_access_divisor}`
- trace device file: `{args.trace_device_file}`
- trace decode dir: `{args.trace_decode_dir}`
- trace stop on complete: `{args.trace_stop_on_complete}`
- trace placement mode: `{args.trace_placement_mode}`
- trace uniform access target count: `{args.trace_uniform_access_target_count}` (`0` means auto ceil(total/expert_num))
- comparison heatmap snapshot: `{args.placement_heatmap_snapshot}`
- access heat display scale: `{args.access_heat_scale}`
- tasks: `{args.num_tasks}` {"(ignored by trace dispatch)" if args.trace_dispatch else ""}
- dispatch need: `{args.dispatch_need}`
- expert num: `{args.expert_num}`
- expert per GPU: `{args.expert_per_gpu}`
- top-k: `{args.topk}`
- payload preset: `{args.payload_preset}`
- dispatch expert FFN params: `{payload_params(args)}`
- precision bytes: `{args.precision_bytes}`
- batch size: `{args.batch_size}`
- dispatch N: `{args.dispatch_n}`
- submit interval: `{args.submit_interval}`
- route probe max inflight: `{args.route_probe_max_inflight}`
- async route probe interval ns: `{args.async_route_probe_interval_ns}`
- async route probe budget: `{args.async_route_probe_budget}`
- async route probe refresh ns: `{args.async_route_probe_refresh_ns}`
- async route probe top-k: `{args.async_route_probe_topk}`
- simulation stop time: `{args.stop_time}`

Each dispatch source sends up to `{args.topk}` expert flows. Per expert-flow payload is:

```text
{payload_params(args)} * {args.precision_bytes} * {args.batch_size} * {args.dispatch_n} bytes
```

## Baseline Trace Placement Tiling

When `TRACE_USE_DEVICE_PLACEMENT=1`, the simulator reads layer `{args.trace_layer_id}` from:

```text
{args.trace_device_file}
```

The trace layer contains `{template_devices}` device templates and `{template.get("slots", 0)}` expert slots. For a `{args.dispatch_need}`-GPU dispatch placement, `BuildTraceDeviceExpertPlan()` now tiles this template over all selected GPUs:

```text
template_device_id = selected_gpu_index % {max(1, template_devices)}
```

For the current setting this gives:

```text
full repeats = {full_repeats}
remainder    = {remainder}
```

So the 32-card baseline placement is copied 8 times when `dispatch_need=256`. This applies to methods whose effective `TRACE_USE_DEVICE_PLACEMENT` is `1`; with the default `trace-placement-mode=baseline-only`, that includes both baseline methods.

## Baseline Variants

- `baseline_uniform`: uses trace placement tiling, default local-first/random routing, uniform `EXPERT_ACCESS_FREQ`, and uniform trace dispatch targets. Every expert receives the same target access count. If `--trace-uniform-access-target-count=0`, the simulator uses `ceil(total_trace_accesses / expert_num)`.
- `baseline_trace`: uses the same trace placement tiling and default routing, but preserves the trace-derived per-expert access targets. It therefore satisfies the original trace access counts after `TRACE_ACCESS_DIVISOR` scaling.
- `posterior_async_probe`: current tuned default, using posterior access estimates, `probe_balanced_domain_spread` placement, `probe_rtt_delta` routing, async background probing, and live route score refresh.
- `posterior_async_probe_score_v2`: same full policy as `posterior_async_probe`, but with score-v2 priority weights: replica/external first, L1/GPU heat balance as guardrails, and access latency/probe cost weighted strongly.

## Method Policies

| method | expert placement | route policy | access mode | trace device placement | uniform trace targets | trace targets as expert freq | route queue weight | async probe | live route |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(method_lines)}

Both baseline variants use trace device placement plus default routing. Default routing first selects a local or same-host expert replica if one exists; otherwise it randomly selects among non-local candidates. Probe methods use the expert placement policy listed in the table above and `probe_rtt_delta` routing.

## Result Summary

| method | status | completed tasks | avg latency ns | avg TPOT ns | trace completed | remaining trace accesses | avg queued flows |
|---|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(summary_lines)}

Artifacts:
- `summary.csv`: parsed cross-method metrics.
- `<method>/policy.conf`: exact policy config for a run.
- `<method>/mncc.log`: mnCCL log.
- `<method>/mncc_flow_finish.csv`: per-flow completion records.
- `figures/latency_tpot_bar.png`: latency and TPOT comparison.
- `figures/tpot_relative_to_baseline_trace.png`: TPOT normalized to `baseline_trace`.
- `figures/queue_pressure_bar.png`: local queue pressure comparison.
- `figures/expert_heatmap_<method>.png`: placement/access heatmap for the configured comparison snapshot. The access panel uses the configured display scale only for readability.
- `figures/expert_placement_latest_<method>.png`: placement-only heatmap for the latest placement update.
"""
    (out_dir / "README.md").write_text(text)


def selected_methods(args: argparse.Namespace) -> dict[str, dict[str, str]]:
    if args.methods == ["all"]:
        return METHODS
    unknown = [name for name in args.methods if name not in METHODS]
    if unknown:
        valid = ", ".join(METHODS)
        raise SystemExit(f"unknown methods: {', '.join(unknown)}; valid: {valid}")
    return {name: METHODS[name] for name in args.methods}


def parse_existing(out_dir: Path, methods: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in methods:
        run_dir = out_dir / method
        if not run_dir.exists():
            print(f"[warn] missing run dir for --skip-run: {run_dir}", flush=True)
            continue
        row = parse_run(run_dir, method)
        row["status"] = "parsed_existing"
        rows.append(row)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=PARSER_DESCRIPTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT, help="Result directory.")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_COMPARE_METHODS,
        help=f"Methods to run, or 'all'. Available: {', '.join(METHODS)}",
    )
    parser.add_argument("--skip-run", action="store_true", help="Only parse and plot existing run dirs.")
    parser.add_argument("--keep-going", action="store_true", help="Continue after a failed method.")
    parser.add_argument(
        "--network-config",
        default="./examples/HW/test.sh",
        help="Path relative to simulation/ passed to --networkConfig.",
    )
    parser.add_argument(
        "--payload-preset",
        choices=["practical", "deepseek-v3-1b"],
        default="deepseek-v3-1b",
        help="Payload preset. practical is quick; deepseek-v3-1b uses the calibrated per-expert payload.",
    )
    parser.add_argument(
        "--dispatch-expert-ffn-params",
        type=int,
        default=None,
        help="Override FFN params per expert. If omitted, payload preset decides.",
    )
    parser.add_argument("--num-tasks", type=int, default=64)
    parser.add_argument("--stop-time", type=float, default=0.6)
    parser.add_argument("--first-submit-time", type=float, default=0.00001)
    parser.add_argument("--submit-interval", type=float, default=0.00001)
    parser.add_argument("--task-seed", type=int, default=20260709)
    parser.add_argument("--pg", type=int, default=3)
    parser.add_argument("--dispatch-need", type=int, default=256)
    parser.add_argument("--expert-num", type=int, default=256)
    parser.add_argument("--expert-per-gpu", type=int, default=9)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--dispatch-n", type=int, default=1)
    parser.add_argument("--precision-bytes", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--expert-mem-bytes", type=int, default=67_108_864)
    parser.add_argument("--monitor-interval-ns", type=int, default=100_000)
    parser.add_argument("--probe-high-pg", type=int, default=0)
    parser.add_argument("--probe-low-pg", type=int, default=6)
    parser.add_argument("--probe-bytes", type=int, default=1)
    parser.add_argument("--route-probe-max-inflight", type=int, default=256)
    parser.add_argument("--async-route-probe-interval-ns", type=int, default=20_000)
    parser.add_argument("--async-route-probe-budget", type=int, default=128)
    parser.add_argument("--async-route-probe-refresh-ns", type=int, default=500_000)
    parser.add_argument("--async-route-probe-topk", type=int, default=2)
    parser.add_argument(
        "--trace-dispatch",
        dest="trace_dispatch",
        action="store_true",
        help="Use HW layer trace targets and keep dispatching until all expert counts are reached.",
    )
    parser.add_argument(
        "--no-trace-dispatch",
        dest="trace_dispatch",
        action="store_false",
        help="Use the older fixed NUM_TASKS random dispatch workload.",
    )
    parser.set_defaults(trace_dispatch=True)
    parser.add_argument("--trace-layer-id", type=int, default=0)
    parser.add_argument(
        "--trace-access-divisor",
        type=int,
        default=1,
        help="Divide HW trace expert access targets by this value using ceil for nonzero counts.",
    )
    parser.add_argument(
        "--trace-access-100x",
        action="store_true",
        help="Shortcut for --trace-access-divisor 100.",
    )
    parser.add_argument(
        "--trace-uniform-access-target-count",
        type=int,
        default=0,
        help="Per-expert target count for baseline_uniform. 0 means ceil(total trace accesses / expert_num).",
    )
    parser.add_argument(
        "--trace-device-file",
        default="./examples/HW/data/device.json",
        help="HW device placement JSON path relative to simulation/.",
    )
    parser.add_argument(
        "--trace-decode-dir",
        default="./examples/HW/data",
        help="Directory containing decode_<device>.csv files relative to simulation/.",
    )
    parser.add_argument(
        "--trace-placement-mode",
        choices=["baseline-only", "all", "none"],
        default="baseline-only",
        help="Which methods use TRACE_USE_DEVICE_PLACEMENT from device.json.",
    )
    parser.add_argument(
        "--trace-stop-on-complete",
        dest="trace_stop_on_complete",
        action="store_true",
        help="Stop ns-3 immediately after all trace expert access targets are reached.",
    )
    parser.add_argument(
        "--no-trace-stop-on-complete",
        dest="trace_stop_on_complete",
        action="store_false",
        help="Keep running until SIMULATION_STOP_TIME after trace targets are reached.",
    )
    parser.set_defaults(trace_stop_on_complete=True)
    parser.add_argument(
        "--access-switch-task",
        type=int,
        default=-1,
        help="Task index for EXPERT_ACCESS_FREQ_AFTER. -1 means half of NUM_TASKS.",
    )
    parser.add_argument(
        "--local-flow-schedule-policy",
        default="default",
        help="Local per-GPU flow scheduling policy.",
    )
    parser.add_argument(
        "--no-placement-detail",
        dest="print_placement_detail",
        action="store_false",
        help="Disable per-GPU expert placement log lines and heatmap source data.",
    )
    parser.set_defaults(print_placement_detail=True)
    parser.add_argument("--heatmap-clip-percentile", type=float, default=99.0)
    parser.add_argument(
        "--placement-heatmap-snapshot",
        choices=["latest", "first", "all"],
        default="all",
        help="Expert placement records used by comparison heatmaps. 'all' accumulates every task placement.",
    )
    parser.add_argument(
        "--access-heat-scale",
        choices=["linear", "log1p", "sqrt"],
        default="log1p",
        help="Display transform for access heatmaps; does not change simulation metrics.",
    )
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = selected_methods(args)

    if args.skip_run:
        rows = parse_existing(out_dir, list(methods))
    else:
        rows = [
            run_method(args, out_dir, method, settings)
            for method, settings in methods.items()
        ]

    if not rows:
        raise SystemExit("no rows parsed; nothing to plot")
    write_summary(out_dir, rows)
    ordered_methods = [str(row["method"]) for row in rows]
    remove_obsolete_visualizations(out_dir)
    plot_metric_bars(out_dir, rows)
    plot_relative_tpot(out_dir, rows)
    if args.print_placement_detail:
        plot_heatmaps(
            out_dir,
            ordered_methods,
            args.heatmap_clip_percentile,
            args.placement_heatmap_snapshot,
            args.access_heat_scale,
        )
    write_readme(out_dir, args, rows)
    print(f"summary: {out_dir / 'summary.csv'}")
    print(f"figures: {out_dir / 'figures'}")
    print(f"readme: {out_dir / 'README.md'}")


if __name__ == "__main__":
    main()
