"""E10: DP feasibility by input structure. Run: python -m benchmarks.run_e10

Use --smoke for a short check. Full sweeps can take hours near the DP limit.
Use --resume to continue an interrupted output file with the current schema.
"""

import argparse
import csv
import hashlib
import json
import random
import statistics
from pathlib import Path

from benchmarks.generators import highly_varied, repeated_sizes, uniform
from benchmarks.run_e4 import measure_dp as shared_measure_dp
from benchmarks.run_e4 import measure_lpt as shared_measure_lpt


SIZES = list(range(8, 33, 2))
MACHINES = [2, 3, 4]
SEEDS = [1, 2, 3, 4, 5]
MAX_TIME = 100
DP_REPEATS = 3
LPT_REPEATS = 5
BATCH_SECONDS = 0.02
FAMILIES = ("uniform", "repeated", "varied")
FIELDS = [
    "family", "n", "m", "seed", "instance_seed", "jobs",
    "dp_status", "dp_error", "dp_solved_at_least_once",
    "dp_requested_repeats", "dp_completed_repeats", "optimum",
    "dp_stats_status", "dp_stats_error", "dp_states", "dp_cache_hits",
    "dp_calls", "dp_samples_seconds", "dp_min_seconds", "dp_median_seconds",
    "dp_iqr_seconds", "dp_memory_status", "dp_memory_error", "dp_peak_kib",
    "lpt_makespan", "lpt_batch_calls", "lpt_samples_seconds",
    "lpt_min_seconds", "lpt_median_seconds", "lpt_iqr_seconds",
    "lpt_peak_kib", "relative_error_percent",
]


def make_jobs(family, n, seed):
    if family == "uniform":
        return uniform(n, MAX_TIME, seed)
    if family == "repeated":
        return repeated_sizes(n, seed, max_time=MAX_TIME)
    if family == "varied":
        return highly_varied(n, seed, max_time=MAX_TIME)
    raise ValueError(f"unknown family: {family}")


def instance_seed(family, n, m, seed):
    value = hashlib.sha256(f"E10:{family}:{n}:{m}:{seed}".encode()).digest()[:8]
    return int.from_bytes(value, "big")


def measure_dp(jobs, m, timeout, repeats):
    result = shared_measure_dp(jobs, machines=m, repeats=repeats,
                               timeout=timeout)
    return {
        "dp_status": result["status"],
        "dp_error": result["error"],
        "dp_solved_at_least_once": result["solved_at_least_once"],
        "dp_requested_repeats": result["requested_repeats"],
        "dp_completed_repeats": result["completed_repeats"],
        "optimum": result["makespan"],
        "dp_stats_status": result["stats_status"],
        "dp_stats_error": result["stats_error"],
        "dp_states": result["states"],
        "dp_cache_hits": result["cache_hits"],
        "dp_calls": result["calls"],
        "dp_samples_seconds": result["samples_seconds"],
        **{"dp_" + key: result[key]
           for key in ("min_seconds", "median_seconds", "iqr_seconds")},
        "dp_memory_status": result["memory_status"],
        "dp_memory_error": result["memory_error"],
        "dp_peak_kib": (result["peak_python_bytes"] / 1024
                        if result["peak_python_bytes"] is not None else None),
    }


def measure_lpt(jobs, m):
    result = shared_measure_lpt(jobs, machines=m, repeats=LPT_REPEATS,
                                batch_seconds=BATCH_SECONDS)
    return {
        "lpt_makespan": result["makespan"],
        "lpt_batch_calls": result["batch_calls"],
        "lpt_samples_seconds": result["samples_seconds"],
        **{"lpt_" + key: result[key]
           for key in ("min_seconds", "median_seconds", "iqr_seconds")},
        "lpt_peak_kib": result["peak_python_bytes"] / 1024,
    }


def numeric(row, column):
    value = row.get(column)
    return None if value in (None, "") else float(value)


def truthy(value):
    return value is True or str(value).lower() == "true"


def median_known(rows, column):
    values = [numeric(row, column) for row in rows]
    values = [value for value in values if value is not None]
    return statistics.median(values) if values else None


def write_summary(path, rows):
    fields = [
        "family", "n", "m", "instances", "dp_solved_instances",
        "dp_solve_rate", "dp_all_repeats_instances", "dp_all_repeats_rate",
        "dp_partial_instances", "dp_timeouts", "dp_startup_timeouts",
        "dp_errors", "dp_stats_completed_instances",
        "dp_memory_completed_instances", "dp_runtime_median_solved_only",
        "dp_peak_kib_median_available_only", "dp_states_median_available_only",
        "lpt_runtime_median_all", "lpt_peak_kib_median_all",
        "known_optimum_instances", "lpt_optimal_fraction_known_only",
        "lpt_max_error_percent_known_only",
    ]
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        groups = sorted({(row["family"], int(row["n"]), int(row["m"]))
                         for row in rows})
        for family, n, m in groups:
            group = [row for row in rows
                     if (row["family"], int(row["n"]), int(row["m"]))
                     == (family, n, m)]
            solved = [row for row in group
                      if truthy(row["dp_solved_at_least_once"])]
            full = [row for row in group if row["dp_status"] == "ok"]
            known = [row for row in group if numeric(row, "optimum") is not None]
            errors = [numeric(row, "relative_error_percent") for row in known]
            errors = [error for error in errors if error is not None]
            writer.writerow({
                "family": family, "n": n, "m": m, "instances": len(group),
                "dp_solved_instances": len(solved),
                "dp_solve_rate": len(solved) / len(group),
                "dp_all_repeats_instances": len(full),
                "dp_all_repeats_rate": len(full) / len(group),
                "dp_partial_instances": sum(row["dp_status"] == "partial" for row in group),
                "dp_timeouts": sum(row["dp_status"] == "DNF" for row in group),
                "dp_startup_timeouts": sum(row["dp_status"] == "startup_timeout"
                                           for row in group),
                "dp_errors": sum(row["dp_status"] == "error" for row in group),
                "dp_stats_completed_instances": sum(row["dp_stats_status"] == "ok"
                                                     for row in group),
                "dp_memory_completed_instances": sum(row["dp_memory_status"] == "ok"
                                                      for row in group),
                "dp_runtime_median_solved_only": median_known(solved, "dp_median_seconds"),
                "dp_peak_kib_median_available_only": median_known(group, "dp_peak_kib"),
                "dp_states_median_available_only": median_known(group, "dp_states"),
                "lpt_runtime_median_all": median_known(group, "lpt_median_seconds"),
                "lpt_peak_kib_median_all": median_known(group, "lpt_peak_kib"),
                "known_optimum_instances": len(known),
                "lpt_optimal_fraction_known_only": (
                    sum(numeric(row, "lpt_makespan") == numeric(row, "optimum")
                        for row in known) / len(known) if known else None),
                "lpt_max_error_percent_known_only": max(errors) if errors else None,
            })


def load_existing(path):
    if not path.exists():
        return [], set()
    with path.open(newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != FIELDS:
            raise ValueError("existing output uses an older schema; choose a new --output "
                             "or rerun without --resume")
        rows = list(reader)
    keys = {(row["family"], int(row["n"]), int(row["m"]), int(row["seed"]))
            for row in rows}
    return rows, keys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true",
                        help="12 small instances; not report evidence")
    parser.add_argument("--timeout", type=float, default=10,
                        help="seconds per DP solve, excluding process startup")
    parser.add_argument("--output", help="override output CSV path")
    parser.add_argument("--resume", action="store_true",
                        help="skip configurations already present in the output")
    args = parser.parse_args(argv)
    if args.timeout <= 0 or MAX_TIME < max(SIZES):
        parser.error("timeout must be positive and MAX_TIME must cover all sizes")
    sizes = [8, 10] if args.smoke else SIZES
    machines = [2] if args.smoke else MACHINES
    seeds = [1, 2] if args.smoke else SEEDS
    repeats = 2 if args.smoke else DP_REPEATS
    default_name = "e10_dp_feasibility" + ("_smoke" if args.smoke else "") + ".csv"
    path = Path(args.output or f"benchmarks/results/{default_name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    rows, existing = load_existing(path) if args.resume else ([], set())

    configs = [(family, n, m, seed) for family in FAMILIES
               for n in sizes for m in machines for seed in seeds]
    random.Random(0).shuffle(configs)
    pending = [config for config in configs if config not in existing]
    mode = "a" if args.resume and path.exists() else "w"
    with path.open(mode, newline="") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        if mode == "w":
            writer.writeheader()
        for count, (family, n, m, seed) in enumerate(pending, 1):
            actual_seed = instance_seed(family, n, m, seed)
            jobs = make_jobs(family, n, actual_seed)
            results = {}
            order = ["DP", "LPT"]
            random.Random(f"{family}:{n}:{m}:{seed}").shuffle(order)
            for algorithm in order:
                results[algorithm] = (measure_dp(jobs.copy(), m, args.timeout, repeats)
                                      if algorithm == "DP" else measure_lpt(jobs.copy(), m))
            row = {
                "family": family, "n": n, "m": m, "seed": seed,
                "instance_seed": actual_seed, "jobs": json.dumps(jobs),
                **results["DP"], **results["LPT"],
            }
            optimum = row["optimum"]
            row["relative_error_percent"] = (
                (row["lpt_makespan"] - optimum) / optimum * 100
                if optimum is not None else None)
            writer.writerow(row)
            output.flush()
            rows.append(row)
            write_summary(path.with_name(path.stem + "_summary.csv"), rows)
            print(f"[{count}/{len(pending)}] {family} n={n} m={m} seed={seed}: "
                  f"DP={row['dp_status']}", flush=True)
    write_summary(path.with_name(path.stem + "_summary.csv"), rows)
    print(f"Saved {path} and summary CSV")


if __name__ == "__main__":
    main()
