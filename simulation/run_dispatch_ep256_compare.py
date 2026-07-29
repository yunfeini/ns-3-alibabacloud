#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "dispatch_ep256_compare"


METHODS = {
    "baseline": {
        "expert_placement": "default",
        "route": "default",
    },
    "probe_balanced_full": {
        "expert_placement": "probe_balanced_full",
        "route": "probe_rtt_delta",
        "overrides": {
            "PLACEMENT_EXTERNAL_WEIGHT": "3.0",
            "PLACEMENT_REPLICA_BALANCE_WEIGHT": "8.0",
            "PLACEMENT_L1_VARIANCE_WEIGHT": "1.0",
            "PLACEMENT_L1_MAX_WEIGHT": "2.0",
            "PLACEMENT_GPU_VARIANCE_WEIGHT": "2.0",
            "PLACEMENT_PROBE_WEIGHT": "0.25",
            "PLACEMENT_PROBE_DELTA_WEIGHT": "0.5",
            "PLACEMENT_PROBE_RTT_WEIGHT": "0.10",
        },
    },
}


def expert_freq() -> str:
    # High-skew but smooth enough to make placement heatmaps readable.
    values = []
    for i in range(256):
        values.append(10.0 - 9.0 * i / 255.0)
    return " ".join(f"{v:.4f}" for v in values)


def config_text(method: str, policy: dict[str, str]) -> str:
    overrides = "\n".join(
        f"{key} {value}" for key, value in policy.get("overrides", {}).items()
    )
    if overrides:
        overrides += "\n"
    return f"""ENABLE_POLICY_GROUP 1
PRINT_SUBMISSIONS 0
PRINT_EXPERT_PLACEMENT_DETAIL 1
SIMULATION_STOP_TIME 0.08

PG 3
DISPATCH_NEED 256
EXPERT_NUM 256
EXPERT_PER_GPU 9
DISPATCH_TOPK 8
EXPERT_MEM_BYTES 67108864
CLUSTER_MONITOR_INTERVAL_NS 100000

DISPATCH_N 1
DISPATCH_EXPERT_FFN_PARAMS 262144
DISPATCH_PRECISION_BYTES 1
DISPATCH_BATCH_SIZE 16

NEED_PLACEMENT_POLICY default
EXPERT_PLACEMENT_POLICY {policy["expert_placement"]}
SAME_RANK_ROUTE_POLICY {policy["route"]}
LOCAL_FLOW_SCHEDULE_POLICY default
GLOBAL_NEED_UPDATE_POLICY default
PD_SPLIT_POLICY default

NUM_TASKS 16
TASK_SEED 20260709
FIRST_SUBMIT_TIME 0.00001
SUBMIT_INTERVAL 0.005

PROBE_HIGH_PG 0
PROBE_LOW_PG 6
PROBE_BYTES 1
ROUTE_PROBE_MAX_INFLIGHT 256

{overrides}\
EXPERT_ACCESS_FREQ {expert_freq()}
"""


def run_method(method: str, policy: dict[str, str]) -> dict[str, float | str]:
    run_dir = OUT / method
    run_dir.mkdir(parents=True, exist_ok=True)
    (ROOT / "examples" / "HW" / "outputs").mkdir(parents=True, exist_ok=True)
    for name in ["mncc.log", "mncc_flow_finish.csv", "mncc_cluster_timeseries.csv"]:
        path = ROOT / name
        if path.exists():
            path.unlink()
    cfg = run_dir / "policy.conf"
    cfg.write_text(config_text(method, policy))

    cmd = [
        "./ns3",
        "run",
        f"scratch/NormalNetwork --networkConfig=./examples/HW/test.sh --policyConfig={cfg}",
    ]
    print("[run]", method, flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)

    for name in ["mncc.log", "mncc_flow_finish.csv", "mncc_cluster_timeseries.csv"]:
        src = ROOT / name
        if src.exists():
            shutil.copy2(src, run_dir / name)

    return parse_run(run_dir, method)


def parse_run(run_dir: Path, method: str) -> dict[str, float | str]:
    log_text = (run_dir / "mncc.log").read_text(errors="ignore")
    completed = int(re.search(r"completed_tasks=(\d+)", log_text).group(1))
    avg_latency = int(re.search(r"average_latency_ns=(\d+)", log_text).group(1))
    avg_tpot = int(re.search(r"average_TPOT_ns=(\d+)", log_text).group(1))
    submitted = sum(1 for _ in re.finditer(r"submitted alltoall", log_text))
    flow_rows = 0
    flow_path = run_dir / "mncc_flow_finish.csv"
    if flow_path.exists():
        with flow_path.open() as f:
            flow_rows = max(0, sum(1 for _ in f) - 1)

    q_avg = q_max = active_avg = active_max = 0.0
    ts_path = run_dir / "mncc_cluster_timeseries.csv"
    if ts_path.exists():
        with ts_path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        if rows:
            queues = [float(r["queued_local_flows"]) for r in rows]
            active = [float(r["active_local_gpus"]) for r in rows]
            q_avg = sum(queues) / len(queues)
            q_max = max(queues)
            active_avg = sum(active) / len(active)
            active_max = max(active)

    return {
        "method": method,
        "completed_tasks": completed,
        "average_latency_ns": avg_latency,
        "average_TPOT_ns": avg_tpot,
        "submitted_collectives": submitted,
        "flow_rows": flow_rows,
        "queued_local_flows_avg": q_avg,
        "queued_local_flows_max": q_max,
        "active_local_gpus_avg": active_avg,
        "active_local_gpus_max": active_max,
    }


def write_summary(rows: list[dict[str, float | str]]) -> None:
    with (OUT / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict[str, float | str]]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    fig_dir = OUT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    methods = [str(r["method"]) for r in rows]
    x = np.arange(len(methods))

    latency_us = [float(r["average_latency_ns"]) / 1000.0 for r in rows]
    tpot_us = [float(r["average_TPOT_ns"]) / 1000.0 for r in rows]
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, latency_us, width, label="avg latency (us)")
    ax.bar(x + width / 2, tpot_us, width, label="avg TPOT (us)")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15)
    ax.set_ylabel("us")
    ax.set_title("EP256 dispatch latency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "latency_tpot_bar.png", dpi=160)
    plt.close(fig)

    q_avg = [float(r["queued_local_flows_avg"]) for r in rows]
    q_max = [float(r["queued_local_flows_max"]) for r in rows]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, q_avg, width, label="avg queued flows")
    ax.bar(x + width / 2, q_max, width, label="max queued flows")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15)
    ax.set_ylabel("flows")
    ax.set_title("EP256 local queue pressure")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "queue_bar.png", dpi=160)
    plt.close(fig)

    for method in methods:
        run_dir = OUT / method
        heatmap = ROOT / "plot_expert_heatmap.py"
        subprocess.run(
            [
                "python3",
                str(heatmap),
                "--log",
                str(run_dir / "mncc.log"),
                "--config",
                str(run_dir / "policy.conf"),
                "--kind",
                "generic",
                "--output",
                str(fig_dir / f"expert_heatmap_{method}.png"),
                "--clip-percentile",
                "99",
            ],
            cwd=ROOT,
            check=True,
        )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for method, policy in METHODS.items():
        rows.append(run_method(method, policy))
    write_summary(rows)
    plot(rows)
    print("summary:", OUT / "summary.csv")
    print("figures:", OUT / "figures")


if __name__ == "__main__":
    main()
