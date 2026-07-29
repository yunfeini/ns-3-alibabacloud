#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import re
from pathlib import Path
from typing import Any


os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_adaptive_weights")

WEIGHT_NAMES = [
    "replica",
    "external",
    "l1_var",
    "l1_max",
    "gpu_heat_var",
    "access_latency",
]

ADAPTIVE_RE = re.compile(
    r"placement adaptive (?P<event>\S+).*?"
    r"mode=(?P<mode>\S+).*?"
    r"step=(?P<step>\d+).*?"
    r"weights=\[(?P<weights>[^\]]*)\]"
    r"(?:.*?partials=\[(?P<partials>[^\]]*)\])?"
    r"(?:.*?ema=\[(?P<ema>[^\]]*)\])?"
    r"(?:.*?gradient=\[(?P<gradient>[^\]]*)\])?"
    r"(?:.*?lr=\[(?P<lr>[^\]]*)\])?"
    r"(?:.*?frozen=\[(?P<frozen>[^\]]*)\])?"
)


def parse_float_list(value: str | None) -> list[float]:
    if not value:
        return []
    out: list[float] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(float(part))
        except ValueError:
            out.append(float("nan"))
    return out


def parse_log(log_path: Path, method: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not log_path.exists():
        return records
    with log_path.open(errors="ignore") as f:
        for line in f:
            match = ADAPTIVE_RE.search(line)
            if not match:
                continue
            weights = parse_float_list(match.group("weights"))
            if not weights:
                continue
            record: dict[str, Any] = {
                "method": method,
                "index": len(records),
                "step": int(match.group("step")),
                "event": match.group("event"),
                "mode": match.group("mode"),
                "weights": weights,
                "partials": parse_float_list(match.group("partials")),
                "ema": parse_float_list(match.group("ema")),
                "gradient": parse_float_list(match.group("gradient")),
                "lr": parse_float_list(match.group("lr")),
                "frozen": parse_float_list(match.group("frozen")),
            }
            records.append(record)
    return records


def discover_logs(result_dir: Path, methods: list[str] | None = None) -> list[tuple[str, Path]]:
    if (result_dir / "mncc.log").exists():
        return [(result_dir.name, result_dir / "mncc.log")]
    if methods:
        return [
            (method, result_dir / method / "mncc.log")
            for method in methods
            if (result_dir / method / "mncc.log").exists()
        ]
    return [
        (path.name, path / "mncc.log")
        for path in sorted(result_dir.iterdir())
        if path.is_dir() and (path / "mncc.log").exists()
    ]


def write_records_csv(out_dir: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    out_path = out_dir / "adaptive_weights.csv"
    fieldnames = ["method", "index", "step", "event", "mode"]
    for prefix in ("weight", "partial", "ema", "gradient", "lr", "frozen"):
        fieldnames.extend(f"{prefix}_{name}" for name in WEIGHT_NAMES)

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row: dict[str, Any] = {
                "method": record["method"],
                "index": record["index"],
                "step": record["step"],
                "event": record["event"],
                "mode": record["mode"],
            }
            for prefix, key in (
                ("weight", "weights"),
                ("partial", "partials"),
                ("ema", "ema"),
                ("gradient", "gradient"),
                ("lr", "lr"),
                ("frozen", "frozen"),
            ):
                values = record.get(key, [])
                for i, name in enumerate(WEIGHT_NAMES):
                    row[f"{prefix}_{name}"] = values[i] if i < len(values) else ""
            writer.writerow(row)


def finite_series(records: list[dict[str, Any]], key: str, idx: int) -> list[float]:
    values: list[float] = []
    for record in records:
        array = record.get(key, [])
        value = array[idx] if idx < len(array) else float("nan")
        values.append(value if math.isfinite(value) else float("nan"))
    return values


def active_weight_indices(records: list[dict[str, Any]]) -> list[int]:
    active: list[int] = []
    for i in range(len(WEIGHT_NAMES)):
        values = finite_series(records, "weights", i)
        if any(math.isfinite(value) and abs(value) > 1e-12 for value in values):
            active.append(i)
    return active if active else list(range(len(WEIGHT_NAMES)))


def plot_method_weights(fig_dir: Path, method: str, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [record["index"] for record in records]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for i in active_weight_indices(records):
        name = WEIGHT_NAMES[i]
        ax.plot(x, finite_series(records, "weights", i), linewidth=1.8, label=name)
    ax.set_xlabel("adaptive update index")
    ax.set_ylabel("weight")
    ax.set_title(f"Adaptive placement weights: {method}")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / f"adaptive_weights_{method}.png", dpi=170)
    plt.close(fig)


def plot_method_partials(fig_dir: Path, method: str, records: list[dict[str, Any]]) -> None:
    if not records or not any(record.get("partials") for record in records):
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [record["index"] for record in records]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for i in active_weight_indices(records):
        name = WEIGHT_NAMES[i]
        ax.plot(x, finite_series(records, "partials", i), linewidth=1.5, label=name)
    ax.set_xlabel("adaptive update index")
    ax.set_ylabel("normalized partial derivative")
    ax.set_title(f"Placement objective partials: {method}")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / f"adaptive_partials_{method}.png", dpi=170)
    plt.close(fig)


def plot_method_learning_rates(fig_dir: Path, method: str, records: list[dict[str, Any]]) -> None:
    if not records or not any(record.get("lr") for record in records):
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [record["index"] for record in records]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for i in active_weight_indices(records):
        name = WEIGHT_NAMES[i]
        ax.plot(x, finite_series(records, "lr", i), linewidth=1.6, label=name)
    ax.set_xlabel("adaptive update index")
    ax.set_ylabel("effective learning rate")
    ax.set_title(f"Dynamic EMA learning rates: {method}")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / f"adaptive_lr_{method}.png", dpi=170)
    plt.close(fig)


def plot_distance_to_final(fig_dir: Path, grouped: dict[str, list[dict[str, Any]]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    plotted = False
    for method, records in grouped.items():
        if len(records) < 2:
            continue
        final = records[-1]["weights"]
        active = active_weight_indices(records)
        final_norm = math.sqrt(
            sum(final[i] * final[i] for i in active if i < len(final))
        )
        final_norm = max(final_norm, 1e-9)
        x = [record["index"] for record in records]
        dist: list[float] = []
        for record in records:
            weights = record["weights"]
            if len(weights) != len(final):
                dist.append(float("nan"))
                continue
            sq = sum((weights[i] - final[i]) * (weights[i] - final[i]) for i in active)
            dist.append(math.sqrt(sq) / final_norm)
        ax.plot(x, dist, linewidth=2.0, label=method)
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("adaptive update index")
    ax.set_ylabel("||w_t - w_final|| / ||w_final||")
    ax.set_title("Adaptive weight convergence")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "adaptive_weight_distance.png", dpi=170)
    plt.close(fig)


def plot_from_result_dir(result_dir: Path, methods: list[str] | None = None) -> list[dict[str, Any]]:
    fig_dir = result_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict[str, Any]]] = {}
    records: list[dict[str, Any]] = []
    for method, log_path in discover_logs(result_dir, methods):
        method_records = parse_log(log_path, method)
        if not method_records:
            continue
        grouped[method] = method_records
        records.extend(method_records)
        plot_method_weights(fig_dir, method, method_records)
        plot_method_partials(fig_dir, method, method_records)
        plot_method_learning_rates(fig_dir, method, method_records)
    write_records_csv(result_dir, records)
    if grouped:
        plot_distance_to_final(fig_dir, grouped)
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot adaptive placement weight trajectories from mncc.log files."
    )
    parser.add_argument("result_dir", type=Path, help="Result directory or one run directory.")
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="Optional method subdirectories to parse under result_dir.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = plot_from_result_dir(args.result_dir.resolve(), args.methods)
    print(f"parsed adaptive weight records: {len(records)}")
    print(f"figures: {args.result_dir.resolve() / 'figures'}")


if __name__ == "__main__":
    main()
