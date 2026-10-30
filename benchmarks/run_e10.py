"""E10: DP feasibility by input structure. Run: python -m benchmarks.run_e10

Use --smoke for a short check. Full sweeps can take hours near the DP limit.
"""

import argparse
import csv
import gc
import hashlib
import json
import random
import statistics
import time
from pathlib import Path

from algorithms.lpt import solve as lpt_solve
from benchmarks.common import peak_memory_in_process
from benchmarks.generators import highly_varied, repeated_sizes, uniform
from benchmarks.run_e4 import dp_trial, timing_summary


SIZES = list(range(8, 33, 2))
MACHINES = [2, 3, 4]
SEEDS = [1, 2, 3, 4, 5]
MAX_TIME = 100
DP_REPEATS = 3
LPT_REPEATS = 5
BATCH_SECONDS = 0.02
FAMILIES = ("uniform", "repeated", "varied")


def make_jobs(family, n, seed):
    if family == "uniform":
        return uniform(n, MAX_TIME, seed)
    if family == "repeated":
        return repeated_sizes(n, seed, max_time=MAX_TIME)
    return highly_varied(n, seed, max_time=MAX_TIME)


def measure_dp(jobs, m, timeout, repeats):
    trials = []
    for _ in range(repeats):
        trial = dp_trial(jobs, m, timeout=timeout)
        trials.append(trial)
        if trial["status"] != "ok":
            break
    completed = [trial for trial in trials if trial["status"] == "ok"]
    samples = [trial["seconds"] for trial in completed]
    last = completed[-1] if completed else {}
    memory = {"status": "skipped"}
    if trials[-1]["status"] == "ok":
        memory = dp_trial(jobs, m, memory=True, timeout=timeout)
    return {
        "dp_status": trials[-1]["status"], "dp_error": trials[-1].get("error", ""),
        "dp_completed_repeats": len(samples), "optimum": last.get("makespan"),
        "dp_states": last.get("states"), "dp_cache_hits": last.get("cache_hits"),
        "dp_calls": last.get("calls"), "dp_samples_seconds": json.dumps(samples),
        **{"dp_" + key: value for key, value in timing_summary(samples).items()},
        "dp_memory_status": memory["status"],
        "dp_peak_kib": memory["peak_python_bytes"] / 1024 if memory.get("peak_python_bytes") is not None else None,
    }


def measure_lpt(jobs, m):
    lpt_solve(jobs, m)  # Untimed warm-up.
    batch = 1
    while True:
        start = time.perf_counter()
        for _ in range(batch):
            lpt_solve(jobs, m)
        if time.perf_counter() - start >= BATCH_SECONDS:
            break
        batch *= 2
    samples = []
    for _ in range(LPT_REPEATS):
        gc.collect()
        start = time.perf_counter()
        for _ in range(batch):
            makespan, assignments = lpt_solve(jobs, m)
        samples.append((time.perf_counter() - start) / batch)
    gc.collect()
    return {
        "lpt_makespan": makespan, "lpt_batch_calls": batch,
        "lpt_samples_seconds": json.dumps(samples),
        **{"lpt_" + key: value for key, value in timing_summary(samples).items()},
        "lpt_peak_kib": peak_memory_in_process(lpt_solve, jobs, m),
    }


def median_known(rows, column):
    values = [row[column] for row in rows if row[column] is not None]
    return statistics.median(values) if values else None


def write_summary(path, rows):
    fields = ["family", "n", "m", "instances", "dp_completed_instances", "dp_completion_rate",
              "dp_timeouts", "dp_errors", "dp_memory_completed_instances",
              "dp_runtime_median_completed_only", "dp_peak_kib_median_completed_only",
              "dp_states_median_completed_only", "lpt_runtime_median_all", "lpt_peak_kib_median_all",
              "known_optimum_instances", "lpt_optimal_fraction_known_only", "lpt_max_error_percent_known_only"]
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for family, n, m in sorted({(r["family"], r["n"], r["m"]) for r in rows}):
            group = [r for r in rows if (r["family"], r["n"], r["m"]) == (family, n, m)]
            completed = [r for r in group if r["dp_status"] == "ok"]
            known = [r for r in group if r["optimum"] is not None]
            writer.writerow({
                "family": family, "n": n, "m": m, "instances": len(group),
                "dp_completed_instances": len(completed), "dp_completion_rate": len(completed) / len(group),
                "dp_timeouts": sum(r["dp_status"] == "DNF" for r in group),
                "dp_errors": sum(r["dp_status"] not in ("ok", "DNF") for r in group),
                "dp_memory_completed_instances": sum(r["dp_memory_status"] == "ok" for r in completed),
                "dp_runtime_median_completed_only": median_known(completed, "dp_median_seconds"),
                "dp_peak_kib_median_completed_only": median_known(completed, "dp_peak_kib"),
                "dp_states_median_completed_only": median_known(completed, "dp_states"),
                "lpt_runtime_median_all": median_known(group, "lpt_median_seconds"),
                "lpt_peak_kib_median_all": median_known(group, "lpt_peak_kib"),
                "known_optimum_instances": len(known),
                "lpt_optimal_fraction_known_only": sum(r["lpt_makespan"] == r["optimum"] for r in known) / len(known) if known else None,
                "lpt_max_error_percent_known_only": max(r["relative_error_percent"] for r in known) if known else None,
            })


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="12 small instances; not report evidence")
    parser.add_argument("--timeout", type=float, default=10, help="seconds per DP call, excluding startup")
    parser.add_argument("--output", help="override output CSV path")
    args = parser.parse_args(argv)
    if args.timeout <= 0 or MAX_TIME < max(SIZES):
        parser.error("timeout must be positive and MAX_TIME must cover all sizes")
    sizes = [8, 10] if args.smoke else SIZES
    machines = [2] if args.smoke else MACHINES
    seeds = [1, 2] if args.smoke else SEEDS
    repeats = 2 if args.smoke else DP_REPEATS
    path = Path(args.output or ("benchmarks/results/e10_dp_feasibility" + ("_smoke" if args.smoke else "") + ".csv"))
    path.parent.mkdir(parents=True, exist_ok=True)
    configs = [(f, n, m, s) for f in FAMILIES for n in sizes for m in machines for s in seeds]
    random.Random(0).shuffle(configs)
    rows = []
    with path.open("w", newline="") as output:
        writer = None
        for count, (family, n, m, seed) in enumerate(configs, 1):
            # Independent streams across sizes; both solvers receive the same jobs.
            actual_seed = int.from_bytes(hashlib.sha256(f"{n}:{m}:{seed}".encode()).digest()[:8], "big")
            jobs = make_jobs(family, n, actual_seed)
            results = {}
            order = ["DP", "LPT"]
            random.Random(f"{family}:{n}:{m}:{seed}").shuffle(order)
            for algorithm in order:
                results[algorithm] = (measure_dp(jobs.copy(), m, args.timeout, repeats)
                                      if algorithm == "DP" else measure_lpt(jobs.copy(), m))
            row = {"family": family, "n": n, "m": m, "seed": seed, "instance_seed": actual_seed,
                   "jobs": json.dumps(jobs), **results["DP"], **results["LPT"]}
            opt = row["optimum"]
            row["relative_error_percent"] = (row["lpt_makespan"] - opt) / opt * 100 if opt is not None else None
            if writer is None:
                writer = csv.DictWriter(output, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            output.flush()
            rows.append(row)
            # Keep a summary usable even if a long sweep is interrupted.
            write_summary(path.with_name(path.stem + "_summary.csv"), rows)
            print(f"[{count}/{len(configs)}] {family} n={n} m={m} seed={seed}: DP={row['dp_status']}", flush=True)
    print(f"Saved {path} and summary CSV")


if __name__ == "__main__":
    main()
