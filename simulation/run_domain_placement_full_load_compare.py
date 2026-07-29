#!/usr/bin/env python3
import argparse
import csv
import math
import shutil
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NETWORK_CONFIG = ROOT / "examples/HW/test.sh"

LOAD_CONFIGS = {
    "light": ROOT / "scratch/pipeline_policy_domain_balanced_freq10_light.conf",
    "medium": ROOT / "scratch/pipeline_policy_domain_balanced_freq10_medium.conf",
    "heavy": ROOT / "scratch/pipeline_policy_domain_balanced_freq10_heavy.conf",
}

PRESETS = {
    "many-small": {
        "SIMULATION_STOP_TIME": "0.06",
        "NUM_TASKS": "256",
        "PREFILL_LENGTH_MIN": "8",
        "PREFILL_LENGTH_MAX": "16",
        "DECODE_LENGTH_MIN": "2",
        "DECODE_LENGTH_MAX": "4",
        "SUBMIT_INTERVAL": "0.000012",
        "CLUSTER_MONITOR_INTERVAL_NS": "20000",
    },
}

LOAD_PRESETS = {
    "spaced-long": {
        "light": {
            "SIMULATION_STOP_TIME": "0.04",
            "NUM_TASKS": "32",
            "PREFILL_LENGTH_MIN": "64",
            "PREFILL_LENGTH_MAX": "128",
            "DECODE_LENGTH_MIN": "12",
            "DECODE_LENGTH_MAX": "24",
            "SUBMIT_INTERVAL": "0.00008",
            "CLUSTER_MONITOR_INTERVAL_NS": "20000",
        },
        "medium": {
            "SIMULATION_STOP_TIME": "0.07",
            "NUM_TASKS": "64",
            "PREFILL_LENGTH_MIN": "96",
            "PREFILL_LENGTH_MAX": "192",
            "DECODE_LENGTH_MIN": "16",
            "DECODE_LENGTH_MAX": "32",
            "SUBMIT_INTERVAL": "0.00006",
            "CLUSTER_MONITOR_INTERVAL_NS": "20000",
        },
        "heavy": {
            "SIMULATION_STOP_TIME": "0.10",
            "NUM_TASKS": "96",
            "PREFILL_LENGTH_MIN": "128",
            "PREFILL_LENGTH_MAX": "256",
            "DECODE_LENGTH_MIN": "24",
            "DECODE_LENGTH_MAX": "48",
            "SUBMIT_INTERVAL": "0.00005",
            "CLUSTER_MONITOR_INTERVAL_NS": "20000",
        },
    },
}

METHOD_POLICIES = {
    "baseline": "default",
    "probe_balanced": "probe_balanced_domain_spread",
    "spread_base": "probe_balanced_domain_spread_base",
    "hot_surplus": "probe_balanced_hot_surplus",
    "probe_cost": "probe_balanced_probe_cost",
    "topology_diverse": "probe_balanced_topology_diverse",
    "full": "probe_balanced_full",
    "probe_balanced_no_route": "probe_balanced_domain_spread",
}

METHOD_POLICY_OVERRIDES = {
    "baseline": {
        "NEED_PLACEMENT_POLICY": "default",
        "EXPERT_PLACEMENT_POLICY": "default",
        "SAME_RANK_ROUTE_POLICY": "default",
        "LOCAL_FLOW_SCHEDULE_POLICY": "default",
        "GLOBAL_NEED_UPDATE_POLICY": "default",
        "PD_SPLIT_POLICY": "default",
    },
    "probe_balanced": {
        "EXPERT_PLACEMENT_POLICY": "probe_balanced_domain_spread",
        "PD_SPLIT_POLICY": "default",
        "PLACEMENT_EXTERNAL_WEIGHT": "5.0",
        "PLACEMENT_REPLICA_BALANCE_WEIGHT": "4.0",
        "PLACEMENT_L1_VARIANCE_WEIGHT": "2.0",
        "PLACEMENT_L1_MAX_WEIGHT": "4.0",
        "PLACEMENT_GPU_VARIANCE_WEIGHT": "1.0",
        "PLACEMENT_PROBE_WEIGHT": "0",
        "PLACEMENT_FORCE_NEW_L1_DOMAIN": "1",
    },
    "spread_base": {
        "EXPERT_PLACEMENT_POLICY": "probe_balanced_domain_spread_base",
        "PD_SPLIT_POLICY": "default",
        "PLACEMENT_EXTERNAL_WEIGHT": "5.0",
        "PLACEMENT_REPLICA_BALANCE_WEIGHT": "4.0",
        "PLACEMENT_L1_VARIANCE_WEIGHT": "2.0",
        "PLACEMENT_L1_MAX_WEIGHT": "4.0",
        "PLACEMENT_GPU_VARIANCE_WEIGHT": "1.0",
        "PLACEMENT_PROBE_WEIGHT": "0",
        "PLACEMENT_FORCE_NEW_L1_DOMAIN": "1",
    },
    "hot_surplus": {
        "EXPERT_PLACEMENT_POLICY": "probe_balanced_hot_surplus",
        "PD_SPLIT_POLICY": "default",
        "PLACEMENT_EXTERNAL_WEIGHT": "5.0",
        "PLACEMENT_REPLICA_BALANCE_WEIGHT": "4.0",
        "PLACEMENT_L1_VARIANCE_WEIGHT": "2.0",
        "PLACEMENT_L1_MAX_WEIGHT": "4.0",
        "PLACEMENT_GPU_VARIANCE_WEIGHT": "1.0",
        "PLACEMENT_PROBE_WEIGHT": "0",
        "PLACEMENT_FORCE_NEW_L1_DOMAIN": "1",
    },
    "probe_cost": {
        "EXPERT_PLACEMENT_POLICY": "probe_balanced_probe_cost",
        "PD_SPLIT_POLICY": "default",
        "PLACEMENT_EXTERNAL_WEIGHT": "5.0",
        "PLACEMENT_REPLICA_BALANCE_WEIGHT": "4.0",
        "PLACEMENT_L1_VARIANCE_WEIGHT": "2.0",
        "PLACEMENT_L1_MAX_WEIGHT": "4.0",
        "PLACEMENT_GPU_VARIANCE_WEIGHT": "1.0",
        "PLACEMENT_PROBE_WEIGHT": "1.0",
        "PLACEMENT_PROBE_DELTA_WEIGHT": "1.0",
        "PLACEMENT_PROBE_RTT_WEIGHT": "0.25",
        "PLACEMENT_FORCE_NEW_L1_DOMAIN": "1",
    },
    "topology_diverse": {
        "EXPERT_PLACEMENT_POLICY": "probe_balanced_topology_diverse",
        "PD_SPLIT_POLICY": "default",
        "PLACEMENT_EXTERNAL_WEIGHT": "5.0",
        "PLACEMENT_REPLICA_BALANCE_WEIGHT": "4.0",
        "PLACEMENT_L1_VARIANCE_WEIGHT": "2.0",
        "PLACEMENT_L1_MAX_WEIGHT": "4.0",
        "PLACEMENT_GPU_VARIANCE_WEIGHT": "1.0",
        "PLACEMENT_PROBE_WEIGHT": "0",
        "PLACEMENT_FORCE_NEW_L1_DOMAIN": "1",
    },
    "full": {
        "EXPERT_PLACEMENT_POLICY": "probe_balanced_full",
        "PD_SPLIT_POLICY": "default",
        "PLACEMENT_EXTERNAL_WEIGHT": "5.0",
        "PLACEMENT_REPLICA_BALANCE_WEIGHT": "4.0",
        "PLACEMENT_L1_VARIANCE_WEIGHT": "2.0",
        "PLACEMENT_L1_MAX_WEIGHT": "4.0",
        "PLACEMENT_GPU_VARIANCE_WEIGHT": "1.0",
        "PLACEMENT_PROBE_WEIGHT": "1.0",
        "PLACEMENT_PROBE_DELTA_WEIGHT": "1.0",
        "PLACEMENT_PROBE_RTT_WEIGHT": "0.25",
        "PLACEMENT_FORCE_NEW_L1_DOMAIN": "1",
    },
    "probe_balanced_no_route": {
        "EXPERT_PLACEMENT_POLICY": "probe_balanced_domain_spread",
        "SAME_RANK_ROUTE_POLICY": "default",
        "PD_SPLIT_POLICY": "default",
        "PLACEMENT_EXTERNAL_WEIGHT": "5.0",
        "PLACEMENT_REPLICA_BALANCE_WEIGHT": "4.0",
        "PLACEMENT_L1_VARIANCE_WEIGHT": "2.0",
        "PLACEMENT_L1_MAX_WEIGHT": "4.0",
        "PLACEMENT_GPU_VARIANCE_WEIGHT": "1.0",
        "PLACEMENT_PROBE_WEIGHT": "0",
        "PLACEMENT_FORCE_NEW_L1_DOMAIN": "1",
    },
}

RUN_OUTPUTS = [
    "mncc.log",
    "mncc_cluster_timeseries.csv",
    "mncc_flow_finish.csv",
]

PLACEMENT_KEYS = [
    "objective",
    "l1_max_load",
    "l1_variance",
    "external_access_cost",
    "replica_domain_imbalance",
    "gpu_load_variance",
    "probe_path_cost",
    "posterior_access_samples",
]

TIMESERIES_KEYS = [
    "gpu_utilization",
    "expert_slot_utilization",
    "expert_fragmentation",
    "placement_gpu_utilization",
    "cn_fragmentation",
    "sched_success_rate",
    "fragment_block_rate",
    "queued_local_flows",
    "active_local_gpus",
]


def config_overrides(args, load_name: str) -> dict[str, str]:
    overrides: dict[str, str] = {}
    if args.preset:
        if args.preset in PRESETS:
            overrides.update(PRESETS[args.preset])
        else:
            overrides.update(LOAD_PRESETS[args.preset][load_name])
    mapping = {
        "SIMULATION_STOP_TIME": args.stop_time,
        "NUM_TASKS": args.num_tasks,
        "PRINT_SUBMISSIONS": args.print_submissions,
        "PRINT_EXPERT_PLACEMENT_DETAIL": args.print_expert_placement_detail,
        "NEED_PREFILL": args.need_prefill,
        "NEED_DECODE": args.need_decode,
        "EXPERT_NUM": args.expert_num,
        "EXPERT_PER_GPU": args.expert_per_gpu,
        "EXPERT_ACCESS_PRIOR": args.expert_access_prior,
        "PREFILL_LENGTH_MIN": args.prefill_min,
        "PREFILL_LENGTH_MAX": args.prefill_max,
        "DECODE_LENGTH_MIN": args.decode_min,
        "DECODE_LENGTH_MAX": args.decode_max,
        "SUBMIT_INTERVAL": args.submit_interval,
        "CLUSTER_MONITOR_INTERVAL_NS": args.monitor_interval_ns,
        "PLACEMENT_EXTERNAL_WEIGHT": args.placement_external_weight,
        "PLACEMENT_REPLICA_BALANCE_WEIGHT": args.placement_replica_balance_weight,
        "PLACEMENT_L1_VARIANCE_WEIGHT": args.placement_l1_variance_weight,
        "PLACEMENT_L1_MAX_WEIGHT": args.placement_l1_max_weight,
        "PLACEMENT_GPU_VARIANCE_WEIGHT": args.placement_gpu_variance_weight,
        "PLACEMENT_PROBE_WEIGHT": args.placement_probe_weight,
        "PLACEMENT_PROBE_DELTA_WEIGHT": args.placement_probe_delta_weight,
        "PLACEMENT_PROBE_RTT_WEIGHT": args.placement_probe_rtt_weight,
        "PLACEMENT_FORCE_NEW_L1_DOMAIN": args.placement_force_new_l1_domain,
    }
    for key, value in mapping.items():
        if value is not None:
            overrides[key] = str(value)
    if args.pd_ratio:
        pd_total = host_gpu_count_from_topology() if args.pd_total_from_topo else args.pd_total
        prefill, decode = parse_pd_ratio(args.pd_ratio, pd_total)
        overrides["NEED_PREFILL"] = str(prefill)
        overrides["NEED_DECODE"] = str(decode)
    return overrides


def parse_pd_ratio(value: str, total: int) -> tuple[int, int]:
    if ":" in value:
        lhs, rhs = value.split(":", 1)
        p_weight = float(lhs)
        d_weight = float(rhs)
    else:
        p_weight = float(value)
        d_weight = 1.0
    if p_weight <= 0 or d_weight <= 0:
        raise ValueError(f"invalid PD ratio: {value}")
    total = max(2, int(total))
    prefill = int(round(total * p_weight / (p_weight + d_weight)))
    prefill = max(1, min(total - 1, prefill))
    decode = total - prefill
    return prefill, decode


def read_config_with_policy(base_path: Path, method_name: str,
                            overrides: dict[str, str] | None = None) -> str:
    lines = []
    policy_overrides = METHOD_POLICY_OVERRIDES[method_name]
    overrides = overrides or {}
    combined_overrides = dict(policy_overrides)
    combined_overrides.update(overrides)
    seen_overrides = set()
    for line in base_path.read_text().splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            key = stripped.split(maxsplit=1)[0]
            if key in combined_overrides:
                lines.append(f"{key} {combined_overrides[key]}")
                seen_overrides.add(key)
            else:
                lines.append(line)
        else:
            lines.append(line)
    for key, value in combined_overrides.items():
        if key not in seen_overrides:
            lines.append(f"{key} {value}")
    return "\n".join(lines) + "\n"


def read_kv_config(path: Path) -> dict[str, str]:
    cfg = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            cfg[parts[0]] = parts[1]
    return cfg


def write_kv_config(path: Path, cfg: dict[str, str]) -> None:
    with path.open("w") as f:
        for key, value in cfg.items():
            f.write(f"{key} {value}\n")


def make_network_config(out_path: Path, run_dir: Path, monitor: bool) -> None:
    cfg = read_kv_config(NETWORK_CONFIG)
    cfg["MON_ENABLE"] = "1" if monitor else "0"
    cfg["FCT_OUTPUT_FILE"] = str(run_dir / "fct.txt")
    cfg["PFC_OUTPUT_FILE"] = str(run_dir / "pfc.txt")
    cfg["QLEN_MON_FILE"] = str(run_dir / "qlen.txt")
    cfg["BW_MON_FILE"] = str(run_dir / "bw.txt")
    cfg["RATE_MON_FILE"] = str(run_dir / "rate.txt")
    cfg["CNP_MON_FILE"] = str(run_dir / "cnp.txt")
    cfg["TRACE_OUTPUT_FILE"] = str(run_dir / "full_load_compare.tr")
    write_kv_config(out_path, cfg)


def resolve_config_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def host_gpu_count_from_topology() -> int:
    cfg = read_kv_config(NETWORK_CONFIG)
    topo = resolve_config_path(cfg.get("TOPOLOGY_FILE", "examples/HW/topo.txt"))
    with topo.open() as f:
        parts = f.readline().split()
    if len(parts) < 4:
        raise ValueError(f"invalid topology header in {topo}")
    node_num = int(parts[0])
    nvswitch_num = int(parts[2])
    switch_num = int(parts[3])
    hosts = node_num - nvswitch_num - switch_num
    if hosts <= 0:
        raise ValueError(f"invalid host count parsed from {topo}: {hosts}")
    return hosts


def clean_root_outputs() -> None:
    for name in RUN_OUTPUTS:
        path = ROOT / name
        if path.exists():
            path.unlink()


def move_root_outputs(run_dir: Path) -> None:
    for name in RUN_OUTPUTS:
        src = ROOT / name
        if src.exists():
            dst = run_dir / name
            if dst.exists():
                dst.unlink()
            shutil.move(str(src), dst)


def token_map(line: str) -> dict[str, str]:
    values = {}
    for token in line.split():
        if "=" in token:
            key, value = token.split("=", 1)
            values[key] = value.rstrip(",")
    return values


def as_float(value: str | None, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def as_int(value: str | None, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except ValueError:
        return default


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * pct))
    return ordered[max(0, min(idx, len(ordered) - 1))]


def summarize_log(log_path: Path) -> dict[str, object]:
    row: dict[str, object] = {
        "completed_tasks": 0,
        "average_TTFT_ns": 0,
        "average_TPOT_ns": 0,
        "incomplete_tasks": 0,
        "placement_retry_count": 0,
    }
    placements: dict[str, dict[str, list[float]]] = {}
    if not log_path.exists():
        return row

    for line in log_path.read_text(errors="replace").splitlines():
        stripped = line.strip()
        if stripped.startswith("completed_tasks="):
            row["completed_tasks"] = as_int(stripped.split("=", 1)[1])
        elif stripped.startswith("average_TTFT_ns="):
            row["average_TTFT_ns"] = as_int(stripped.split("=", 1)[1])
        elif stripped.startswith("average_TPOT_ns="):
            row["average_TPOT_ns"] = as_int(stripped.split("=", 1)[1])
        if "incomplete at simulation end" in line:
            row["incomplete_tasks"] = int(row["incomplete_tasks"]) + 1
        if "placement failed" in line:
            row["placement_retry_count"] = int(row["placement_retry_count"]) + 1
        if "PipelinePolicy: expert placement dispatch" not in line:
            continue
        values = token_map(line)
        kind = values.get("kind", "unknown")
        bucket = placements.setdefault(kind, {key: [] for key in PLACEMENT_KEYS})
        for key in PLACEMENT_KEYS:
            bucket[key].append(as_float(values.get(key)))

    for kind, metrics in placements.items():
        row[f"{kind}_placement_samples"] = len(next(iter(metrics.values()), []))
        for key, values in metrics.items():
            row[f"{kind}_{key}_avg"] = mean(values)
            row[f"{kind}_{key}_max"] = max(values) if values else 0.0
    return row


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def summarize_timeseries(csv_path: Path) -> dict[str, object]:
    rows = load_csv_rows(csv_path)
    active = [
        row for row in rows
        if as_float(row.get("active_tasks")) > 0
        or as_float(row.get("outstanding_collectives")) > 0
        or as_float(row.get("queued_local_flows")) > 0
        or as_float(row.get("active_local_gpus")) > 0
    ]
    samples = active or rows
    out: dict[str, object] = {
        "timeseries_samples": len(samples),
    }
    for key in TIMESERIES_KEYS:
        values = [as_float(row.get(key)) for row in samples]
        out[f"{key}_avg"] = mean(values)
        out[f"{key}_p95"] = percentile(values, 0.95)
        out[f"{key}_max"] = max(values) if values else 0.0

    mem_used = [as_float(row.get("used_gpu_mem_bytes")) for row in samples]
    mem_total = [as_float(row.get("total_gpu_mem_bytes")) for row in samples]
    mem_ratio = [
        used / total
        for used, total in zip(mem_used, mem_total)
        if total > 0 and math.isfinite(used) and math.isfinite(total)
    ]
    out["gpu_mem_utilization_avg"] = mean(mem_ratio)
    out["gpu_mem_utilization_p95"] = percentile(mem_ratio, 0.95)
    out["gpu_mem_utilization_max"] = max(mem_ratio) if mem_ratio else 0.0
    return out


def summarize_run(run_dir: Path, load_name: str, method_name: str, policy: str,
                  wall_time_s: float = 0.0) -> dict[str, object]:
    row: dict[str, object] = {
        "load": load_name,
        "method": method_name,
        "policy": policy,
        "wall_time_s": round(wall_time_s, 3),
        "log_path": str(run_dir / "mncc.log"),
    }
    row.update(summarize_log(run_dir / "mncc.log"))
    row.update(summarize_timeseries(run_dir / "mncc_cluster_timeseries.csv"))
    return row


def run_case(load_name: str, method_name: str, args) -> dict[str, object]:
    policy = METHOD_POLICIES[method_name]
    run_dir = args.output / load_name / method_name
    run_dir.mkdir(parents=True, exist_ok=True)

    network_cfg = run_dir / "network.conf"
    policy_cfg = run_dir / "policy.conf"
    make_network_config(network_cfg, run_dir, args.monitor)
    policy_cfg.write_text(read_config_with_policy(
        LOAD_CONFIGS[load_name], method_name, config_overrides(args, load_name)))

    cmd = [
        "./ns3",
        "run",
        f"scratch/NormalNetwork --networkConfig={network_cfg} --policyConfig={policy_cfg}",
    ]
    print(f"[run] load={load_name} method={method_name} policy={policy}", flush=True)
    wall_time_s = 0.0
    if not args.skip_run:
        clean_root_outputs()
        started = time.monotonic()
        subprocess.run(cmd, cwd=ROOT, check=True)
        wall_time_s = time.monotonic() - started
        move_root_outputs(run_dir)

    row = summarize_run(run_dir, load_name, method_name, policy, wall_time_s)
    print(
        "[done] "
        f"load={load_name} method={method_name} "
        f"completed={row.get('completed_tasks', 0)} "
        f"TTFT={row.get('average_TTFT_ns', 0)}ns "
        f"TPOT={row.get('average_TPOT_ns', 0)}ns "
        f"wall={wall_time_s:.1f}s",
        flush=True,
    )
    return row


def write_summary(rows: list[dict[str, object]], path: Path) -> None:
    preferred = [
        "load",
        "method",
        "policy",
        "completed_tasks",
        "average_TTFT_ns",
        "average_TPOT_ns",
        "incomplete_tasks",
        "placement_retry_count",
        "prefill_objective_avg",
        "prefill_l1_max_load_avg",
        "prefill_l1_variance_avg",
        "prefill_external_access_cost_avg",
        "prefill_replica_domain_imbalance_avg",
        "prefill_gpu_load_variance_avg",
        "prefill_probe_path_cost_avg",
        "prefill_posterior_access_samples_avg",
        "decode_objective_avg",
        "decode_l1_max_load_avg",
        "decode_l1_variance_avg",
        "decode_external_access_cost_avg",
        "decode_replica_domain_imbalance_avg",
        "decode_gpu_load_variance_avg",
        "decode_probe_path_cost_avg",
        "decode_posterior_access_samples_avg",
        "gpu_mem_utilization_avg",
        "gpu_mem_utilization_p95",
        "expert_fragmentation_avg",
        "expert_fragmentation_p95",
        "cn_fragmentation_avg",
        "cn_fragmentation_p95",
        "queued_local_flows_p95",
        "queued_local_flows_max",
        "wall_time_s",
        "log_path",
    ]
    all_fields = []
    for field in preferred:
        if any(field in row for row in rows):
            all_fields.append(field)
    for row in rows:
        for field in row:
            if field not in all_fields:
                all_fields.append(field)

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in all_fields})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run full-load comparison for expert placement policies."
    )
    parser.add_argument("--loads", nargs="+", default=list(LOAD_CONFIGS), choices=list(LOAD_CONFIGS))
    parser.add_argument("--methods", nargs="+", default=list(METHOD_POLICIES), choices=list(METHOD_POLICIES))
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "domain_placement_full_load_compare",
    )
    parser.add_argument("--monitor", action="store_true", help="Enable detailed network monitors.")
    parser.add_argument("--skip-run", action="store_true", help="Only parse existing run outputs.")
    preset_choices = sorted(set(PRESETS) | set(LOAD_PRESETS))
    parser.add_argument("--preset", choices=preset_choices,
                        help="Apply a named workload override preset.")
    parser.add_argument("--num-tasks", type=int, help="Override NUM_TASKS.")
    parser.add_argument("--print-submissions", help="Override PRINT_SUBMISSIONS.")
    parser.add_argument("--print-expert-placement-detail",
                        help="Override PRINT_EXPERT_PLACEMENT_DETAIL.")
    parser.add_argument("--need-prefill", type=int, help="Override NEED_PREFILL.")
    parser.add_argument("--need-decode", type=int, help="Override NEED_DECODE.")
    parser.add_argument("--pd-ratio",
                        help="Set NEED_PREFILL:NEED_DECODE by ratio, e.g. 3:1.")
    parser.add_argument("--pd-total", type=int, default=160,
                        help="Total GPUs assigned across P and D when --pd-ratio is used.")
    parser.add_argument("--pd-total-from-topo", action="store_true",
                        help="Use host/GPU count parsed from TOPOLOGY_FILE as --pd-total.")
    parser.add_argument("--expert-num", "--ep-num", dest="expert_num", type=int,
                        help="Override EXPERT_NUM.")
    parser.add_argument("--expert-per-gpu", type=int, help="Override EXPERT_PER_GPU.")
    parser.add_argument("--expert-access-prior",
                        help="Override EXPERT_ACCESS_PRIOR for posterior placement estimates.")
    parser.add_argument("--prefill-min", type=int, help="Override PREFILL_LENGTH_MIN.")
    parser.add_argument("--prefill-max", type=int, help="Override PREFILL_LENGTH_MAX.")
    parser.add_argument("--decode-min", type=int, help="Override DECODE_LENGTH_MIN.")
    parser.add_argument("--decode-max", type=int, help="Override DECODE_LENGTH_MAX.")
    parser.add_argument("--submit-interval", help="Override SUBMIT_INTERVAL.")
    parser.add_argument("--stop-time", help="Override SIMULATION_STOP_TIME.")
    parser.add_argument("--monitor-interval-ns", type=int,
                        help="Override CLUSTER_MONITOR_INTERVAL_NS.")
    parser.add_argument("--placement-external-weight",
                        help="Override PLACEMENT_EXTERNAL_WEIGHT.")
    parser.add_argument("--placement-replica-balance-weight",
                        help="Override PLACEMENT_REPLICA_BALANCE_WEIGHT.")
    parser.add_argument("--placement-l1-variance-weight",
                        help="Override PLACEMENT_L1_VARIANCE_WEIGHT.")
    parser.add_argument("--placement-l1-max-weight",
                        help="Override PLACEMENT_L1_MAX_WEIGHT.")
    parser.add_argument("--placement-gpu-variance-weight",
                        help="Override PLACEMENT_GPU_VARIANCE_WEIGHT.")
    parser.add_argument("--placement-probe-weight",
                        help="Override PLACEMENT_PROBE_WEIGHT.")
    parser.add_argument("--placement-probe-delta-weight",
                        help="Override PLACEMENT_PROBE_DELTA_WEIGHT.")
    parser.add_argument("--placement-probe-rtt-weight",
                        help="Override PLACEMENT_PROBE_RTT_WEIGHT.")
    parser.add_argument("--placement-force-new-l1-domain",
                        help="Override PLACEMENT_FORCE_NEW_L1_DOMAIN.")
    args = parser.parse_args()

    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)

    rows = []
    for load_name in args.loads:
        for method_name in args.methods:
            rows.append(run_case(load_name, method_name, args))

    summary_path = args.output / "summary.csv"
    write_summary(rows, summary_path)
    print(f"[summary] {summary_path}", flush=True)


if __name__ == "__main__":
    main()
