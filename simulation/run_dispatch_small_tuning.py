#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "dispatch_small_tuning"


CASES = {
    "baseline": {
        "EXPERT_PLACEMENT_POLICY": "default",
        "SAME_RANK_ROUTE_POLICY": "default",
    },
    "posterior_rtt": {
        "EXPERT_PLACEMENT_POLICY": "probe_balanced_full",
        "SAME_RANK_ROUTE_POLICY": "probe_rtt_delta",
        "PLACEMENT_ACCESS_MODE": "posterior",
        "PLACEMENT_EXTERNAL_WEIGHT": "5.0",
        "PLACEMENT_REPLICA_BALANCE_WEIGHT": "4.0",
        "PLACEMENT_L1_VARIANCE_WEIGHT": "2.0",
        "PLACEMENT_L1_MAX_WEIGHT": "4.0",
        "PLACEMENT_GPU_VARIANCE_WEIGHT": "1.0",
        "PLACEMENT_PROBE_WEIGHT": "1.0",
        "PLACEMENT_PROBE_DELTA_WEIGHT": "1.0",
        "PLACEMENT_PROBE_RTT_WEIGHT": "0.25",
    },
    "initial_static": {
        "EXPERT_PLACEMENT_POLICY": "probe_balanced_full",
        "SAME_RANK_ROUTE_POLICY": "probe_rtt_delta",
        "PLACEMENT_ACCESS_MODE": "initial",
        "PLACEMENT_EXTERNAL_WEIGHT": "3.0",
        "PLACEMENT_REPLICA_BALANCE_WEIGHT": "8.0",
        "PLACEMENT_L1_VARIANCE_WEIGHT": "2.0",
        "PLACEMENT_L1_MAX_WEIGHT": "2.0",
        "PLACEMENT_GPU_VARIANCE_WEIGHT": "2.0",
        "PLACEMENT_PROBE_WEIGHT": "0.25",
        "PLACEMENT_PROBE_DELTA_WEIGHT": "0.5",
        "PLACEMENT_PROBE_RTT_WEIGHT": "0.10",
    },
    "initial_drift_rtt": {
        "EXPERT_PLACEMENT_POLICY": "probe_balanced_full",
        "SAME_RANK_ROUTE_POLICY": "probe_rtt_delta",
        "PLACEMENT_ACCESS_MODE": "initial_until_drift",
        "PLACEMENT_DRIFT_THRESHOLD": "0.18",
        "PLACEMENT_DRIFT_MIN_SAMPLES": "512",
        "PLACEMENT_EXTERNAL_WEIGHT": "3.0",
        "PLACEMENT_REPLICA_BALANCE_WEIGHT": "8.0",
        "PLACEMENT_L1_VARIANCE_WEIGHT": "1.0",
        "PLACEMENT_L1_MAX_WEIGHT": "2.0",
        "PLACEMENT_GPU_VARIANCE_WEIGHT": "2.0",
        "PLACEMENT_PROBE_WEIGHT": "0.25",
        "PLACEMENT_PROBE_DELTA_WEIGHT": "0.5",
        "PLACEMENT_PROBE_RTT_WEIGHT": "0.10",
    },
    "initial_drift_queue_025": {
        "EXPERT_PLACEMENT_POLICY": "probe_balanced_full",
        "SAME_RANK_ROUTE_POLICY": "probe_rtt_delta",
        "PLACEMENT_ACCESS_MODE": "initial_until_drift",
        "PLACEMENT_DRIFT_THRESHOLD": "0.18",
        "PLACEMENT_DRIFT_MIN_SAMPLES": "512",
        "ROUTE_QUEUE_WEIGHT": "0.25",
        "ROUTE_QUEUE_NORM": "16",
        "PLACEMENT_EXTERNAL_WEIGHT": "3.0",
        "PLACEMENT_REPLICA_BALANCE_WEIGHT": "8.0",
        "PLACEMENT_L1_VARIANCE_WEIGHT": "1.0",
        "PLACEMENT_L1_MAX_WEIGHT": "2.0",
        "PLACEMENT_GPU_VARIANCE_WEIGHT": "2.0",
        "PLACEMENT_PROBE_WEIGHT": "0.25",
        "PLACEMENT_PROBE_DELTA_WEIGHT": "0.5",
        "PLACEMENT_PROBE_RTT_WEIGHT": "0.10",
    },
    "initial_drift_queue_100": {
        "EXPERT_PLACEMENT_POLICY": "probe_balanced_full",
        "SAME_RANK_ROUTE_POLICY": "probe_rtt_delta",
        "PLACEMENT_ACCESS_MODE": "initial_until_drift",
        "PLACEMENT_DRIFT_THRESHOLD": "0.18",
        "PLACEMENT_DRIFT_MIN_SAMPLES": "512",
        "ROUTE_QUEUE_WEIGHT": "1.00",
        "ROUTE_QUEUE_NORM": "16",
        "PLACEMENT_EXTERNAL_WEIGHT": "3.0",
        "PLACEMENT_REPLICA_BALANCE_WEIGHT": "8.0",
        "PLACEMENT_L1_VARIANCE_WEIGHT": "1.0",
        "PLACEMENT_L1_MAX_WEIGHT": "2.0",
        "PLACEMENT_GPU_VARIANCE_WEIGHT": "2.0",
        "PLACEMENT_PROBE_WEIGHT": "0.25",
        "PLACEMENT_PROBE_DELTA_WEIGHT": "0.5",
        "PLACEMENT_PROBE_RTT_WEIGHT": "0.10",
    },
    "initial_static_queue_100": {
        "EXPERT_PLACEMENT_POLICY": "probe_balanced_full",
        "SAME_RANK_ROUTE_POLICY": "probe_rtt_delta",
        "PLACEMENT_ACCESS_MODE": "initial",
        "ROUTE_QUEUE_WEIGHT": "1.00",
        "ROUTE_QUEUE_NORM": "16",
        "PLACEMENT_EXTERNAL_WEIGHT": "3.0",
        "PLACEMENT_REPLICA_BALANCE_WEIGHT": "8.0",
        "PLACEMENT_L1_VARIANCE_WEIGHT": "1.0",
        "PLACEMENT_L1_MAX_WEIGHT": "2.0",
        "PLACEMENT_GPU_VARIANCE_WEIGHT": "2.0",
        "PLACEMENT_PROBE_WEIGHT": "0.25",
        "PLACEMENT_PROBE_DELTA_WEIGHT": "0.5",
        "PLACEMENT_PROBE_RTT_WEIGHT": "0.10",
    },
    "initial_drift_late_queue_100": {
        "EXPERT_PLACEMENT_POLICY": "probe_balanced_full",
        "SAME_RANK_ROUTE_POLICY": "probe_rtt_delta",
        "PLACEMENT_ACCESS_MODE": "initial_until_drift",
        "PLACEMENT_DRIFT_THRESHOLD": "0.30",
        "PLACEMENT_DRIFT_MIN_SAMPLES": "512",
        "ROUTE_QUEUE_WEIGHT": "1.00",
        "ROUTE_QUEUE_NORM": "16",
        "PLACEMENT_EXTERNAL_WEIGHT": "3.0",
        "PLACEMENT_REPLICA_BALANCE_WEIGHT": "8.0",
        "PLACEMENT_L1_VARIANCE_WEIGHT": "1.0",
        "PLACEMENT_L1_MAX_WEIGHT": "2.0",
        "PLACEMENT_GPU_VARIANCE_WEIGHT": "2.0",
        "PLACEMENT_PROBE_WEIGHT": "0.25",
        "PLACEMENT_PROBE_DELTA_WEIGHT": "0.5",
        "PLACEMENT_PROBE_RTT_WEIGHT": "0.10",
    },
    "initial_drift_late_queue_200": {
        "EXPERT_PLACEMENT_POLICY": "probe_balanced_full",
        "SAME_RANK_ROUTE_POLICY": "probe_rtt_delta",
        "PLACEMENT_ACCESS_MODE": "initial_until_drift",
        "PLACEMENT_DRIFT_THRESHOLD": "0.30",
        "PLACEMENT_DRIFT_MIN_SAMPLES": "512",
        "ROUTE_QUEUE_WEIGHT": "2.00",
        "ROUTE_QUEUE_NORM": "16",
        "PLACEMENT_EXTERNAL_WEIGHT": "3.0",
        "PLACEMENT_REPLICA_BALANCE_WEIGHT": "8.0",
        "PLACEMENT_L1_VARIANCE_WEIGHT": "1.0",
        "PLACEMENT_L1_MAX_WEIGHT": "2.0",
        "PLACEMENT_GPU_VARIANCE_WEIGHT": "2.0",
        "PLACEMENT_PROBE_WEIGHT": "0.25",
        "PLACEMENT_PROBE_DELTA_WEIGHT": "0.5",
        "PLACEMENT_PROBE_RTT_WEIGHT": "0.10",
    },
}


def expert_freq() -> str:
    return " ".join(str(v) for v in range(10, 0, -1)) + " " + " ".join("1" for _ in range(6))


def expert_freq_after() -> str:
    return " ".join(str(v) for v in [1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


def config_text(settings: dict[str, str]) -> str:
    base = {
        "ENABLE_POLICY_GROUP": "1",
        "PRINT_SUBMISSIONS": "0",
        "PRINT_EXPERT_PLACEMENT_DETAIL": "1",
        "SIMULATION_STOP_TIME": "0.12",
        "PG": "3",
        "DISPATCH_NEED": "16",
        "EXPERT_NUM": "16",
        "EXPERT_PER_GPU": "2",
        "DISPATCH_TOPK": "8",
        "EXPERT_MEM_BYTES": "67108864",
        "CLUSTER_MONITOR_INTERVAL_NS": "100000",
        "DISPATCH_N": "1",
        "DISPATCH_EXPERT_FFN_PARAMS": "65536",
        "DISPATCH_PRECISION_BYTES": "1",
        "DISPATCH_BATCH_SIZE": "16",
        "NEED_PLACEMENT_POLICY": "default",
        "LOCAL_FLOW_SCHEDULE_POLICY": "default",
        "GLOBAL_NEED_UPDATE_POLICY": "default",
        "PD_SPLIT_POLICY": "default",
        "NUM_TASKS": "32",
        "TASK_SEED": "20260709",
        "FIRST_SUBMIT_TIME": "0.00001",
        "SUBMIT_INTERVAL": "0.0004",
        "EXPERT_ACCESS_SWITCH_TASK": "16",
        "PROBE_HIGH_PG": "0",
        "PROBE_LOW_PG": "6",
        "PROBE_BYTES": "1",
        "ROUTE_PROBE_MAX_INFLIGHT": "64",
        "EXPERT_ACCESS_FREQ": expert_freq(),
        "EXPERT_ACCESS_FREQ_AFTER": expert_freq_after(),
    }
    base.update(settings)
    return "\n".join(f"{k} {v}" for k, v in base.items()) + "\n"


def run_case(name: str, settings: dict[str, str]) -> dict[str, float | str]:
    run_dir = OUT / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (ROOT / "examples" / "rdma-test" / "outputs").mkdir(parents=True, exist_ok=True)
    for filename in ["mncc.log", "mncc_flow_finish.csv", "mncc_cluster_timeseries.csv"]:
        path = ROOT / filename
        if path.exists():
            path.unlink()
    cfg = run_dir / "policy.conf"
    cfg.write_text(config_text(settings))
    cmd = [
        "./ns3",
        "run",
        f"scratch/NormalNetwork --networkConfig=./examples/rdma-test/test_ondpm.sh --policyConfig={cfg}",
    ]
    print("[run]", name, flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    for filename in ["mncc.log", "mncc_flow_finish.csv", "mncc_cluster_timeseries.csv"]:
        src = ROOT / filename
        if src.exists():
            shutil.copy2(src, run_dir / filename)
    return parse_case(name, run_dir)


def parse_case(name: str, run_dir: Path) -> dict[str, float | str]:
    log = (run_dir / "mncc.log").read_text(errors="ignore")
    completed = int(re.search(r"completed_tasks=(\d+)", log).group(1))
    avg_latency = int(re.search(r"average_latency_ns=(\d+)", log).group(1))
    avg_tpot = int(re.search(r"average_TPOT_ns=(\d+)", log).group(1))
    local_routes = 0
    total_routes = 0
    for m in re.finditer(r"routes=(\d+) local_routes=(\d+)", log):
      total_routes += int(m.group(1))
      local_routes += int(m.group(2))
    drift_hot_updates = sum(1 for _ in re.finditer(r"drift_hot_update=1", log))
    access_drifts = [
        float(m.group(1))
        for m in re.finditer(r"access_drift=([0-9.eE+-]+)", log)
    ]
    q_avg = q_max = active_avg = active_max = 0.0
    ts = run_dir / "mncc_cluster_timeseries.csv"
    if ts.exists():
        with ts.open(newline="") as f:
            rows = list(csv.DictReader(f))
        if rows:
            queues = [float(r["queued_local_flows"]) for r in rows]
            active = [float(r["active_local_gpus"]) for r in rows]
            q_avg = sum(queues) / len(queues)
            q_max = max(queues)
            active_avg = sum(active) / len(active)
            active_max = max(active)
    flow_rows = 0
    flow = run_dir / "mncc_flow_finish.csv"
    if flow.exists():
        with flow.open() as f:
            flow_rows = max(0, sum(1 for _ in f) - 1)
    return {
        "case": name,
        "completed_tasks": completed,
        "average_latency_ns": avg_latency,
        "average_TPOT_ns": avg_tpot,
        "queued_local_flows_avg": q_avg,
        "queued_local_flows_max": q_max,
        "active_local_gpus_avg": active_avg,
        "active_local_gpus_max": active_max,
        "flow_rows": flow_rows,
        "local_route_ratio": local_routes / total_routes if total_routes else 0.0,
        "drift_hot_updates": drift_hot_updates,
        "max_access_drift": max(access_drifts) if access_drifts else 0.0,
    }


def write_summary(rows: list[dict[str, float | str]]) -> None:
    with (OUT / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    best = sorted(rows, key=lambda r: (float(r["average_TPOT_ns"]), float(r["queued_local_flows_avg"])))[0]
    (OUT / "best_case.txt").write_text(str(best) + "\n")


def plot(rows: list[dict[str, float | str]]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    fig_dir = OUT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    labels = [str(r["case"]) for r in rows]
    x = np.arange(len(labels))

    tpot_us = [float(r["average_TPOT_ns"]) / 1000.0 for r in rows]
    latency_us = [float(r["average_latency_ns"]) / 1000.0 for r in rows]
    fig, ax = plt.subplots(figsize=(12, 5))
    width = 0.36
    ax.bar(x - width / 2, latency_us, width, label="avg latency (us)")
    ax.bar(x + width / 2, tpot_us, width, label="avg TPOT (us)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_title("Small-scale dispatch tuning latency")
    ax.set_ylabel("us")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "latency_tpot_bar.png", dpi=160)
    plt.close(fig)

    q_avg = [float(r["queued_local_flows_avg"]) for r in rows]
    q_max = [float(r["queued_local_flows_max"]) for r in rows]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, q_avg, width, label="avg queued flows")
    ax.bar(x + width / 2, q_max, width, label="max queued flows")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_title("Small-scale dispatch tuning queue pressure")
    ax.set_ylabel("flows")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "queue_bar.png", dpi=160)
    plt.close(fig)

    best_name = str(sorted(rows, key=lambda r: (float(r["average_TPOT_ns"]), float(r["queued_local_flows_avg"])))[0]["case"])
    subprocess.run(
        [
            "python3",
            str(ROOT / "plot_expert_heatmap.py"),
            "--log",
            str(OUT / best_name / "mncc.log"),
            "--config",
            str(OUT / best_name / "policy.conf"),
            "--kind",
            "generic",
            "--output",
            str(fig_dir / f"expert_heatmap_{best_name}.png"),
            "--clip-percentile",
            "99",
        ],
        cwd=ROOT,
        check=True,
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [run_case(name, settings) for name, settings in CASES.items()]
    write_summary(rows)
    plot(rows)
    print("summary:", OUT / "summary.csv")
    print("best:", OUT / "best_case.txt")
    print("figures:", OUT / "figures")


if __name__ == "__main__":
    main()
