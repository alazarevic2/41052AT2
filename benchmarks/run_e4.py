"""E4: input structure. Run: python -m benchmarks.run_e4"""

import csv
import gc
import hashlib
import json
import random
import statistics
from pathlib import Path

from algorithms.lpt import solve as lpt_solve
from benchmarks.common import peak_memory_in_process, time_in_process
from benchmarks.generators import (
    balanced, few_large_many_small, highly_varied, repeated_sizes,
    similar_sized, uniform,
)
from benchmarks.process_helpers import dp_trial


N = 12
M = 3
SEEDS = [1, 2, 3, 4, 5]
DP_REPEATS = 3
LPT_REPEATS = 5
TIMEOUT = 10
TIME_RANGE = 100
BATCH_SECONDS = 0.02
OUTPUT = "benchmarks/results/e4_input_structure.csv"
FAMILIES = {
    "uniform": lambda seed: uniform(N, TIME_RANGE, seed),
    "repeated": lambda seed: repeated_sizes(N, seed, max_time=TIME_RANGE),
    "varied": lambda seed: highly_varied(N, seed, max_time=TIME_RANGE),
    "large_small": lambda seed: few_large_many_small(N, seed),
    "similar": lambda seed: similar_sized(N, seed),
    "balanced": lambda seed: balanced(N, M, seed, target=TIME_RANGE),
}


def family_seed(family, seed):
    """Give each family a deterministic independent random stream."""
    value = hashlib.sha256(f"E4:{family}:{seed}".encode()).digest()[:8]
    return int.from_bytes(value, "big")


def timing_summary(samples):
    if not samples:
        return {"min_seconds": None, "median_seconds": None, "iqr_seconds": None}
    quartiles = (statistics.quantiles(samples, n=4, method="inclusive")
                 if len(samples) > 1 else [samples[0]] * 3)
    return {"min_seconds": min(samples),
            "median_seconds": statistics.median(samples),
            "iqr_seconds": quartiles[2] - quartiles[0]}


def measure_dp(jobs, machines=None, repeats=None, timeout=None):
    machines = M if machines is None else machines
    repeats = DP_REPEATS if repeats is None else repeats
    timeout = TIMEOUT if timeout is None else timeout
    trials = []
    for _ in range(repeats):
        trial = dp_trial(jobs, machines, timeout)
        trials.append(trial)
        if trial["status"] != "ok":
            break

    completed = [trial for trial in trials if trial["status"] == "ok"]
    samples = [trial["seconds"] for trial in completed]
    failure = next((trial for trial in trials if trial["status"] != "ok"), None)
    if len(completed) == repeats:
        status = "ok"
    elif completed:
        status = "partial"
    else:
        status = failure["status"]

    stats = memory = {"status": "skipped"}
    if status == "ok":
        stats = dp_trial(jobs, machines, timeout, collect_stats=True)
        memory = dp_trial(jobs, machines, timeout, memory=True)
    last = completed[-1] if completed else {}
    for extra in (stats, memory):
        if (extra["status"] == "ok"
                and extra["makespan"] != last.get("makespan")):
            raise RuntimeError("DP measurement passes disagree")
    return {
        "status": status,
        "error": failure.get("error", "") if failure else "",
        "solved_at_least_once": bool(completed),
        "makespan": last.get("makespan"),
        "states": stats.get("states"),
        "stats_status": stats["status"],
        "stats_error": stats.get("error", ""),
        "cache_hits": stats.get("cache_hits"),
        "calls": stats.get("calls"),
        "samples_seconds": json.dumps(samples),
        "requested_repeats": repeats,
        "completed_repeats": len(samples),
        "batch_calls": 1,
        "memory_status": memory["status"],
        "memory_error": memory.get("error", ""),
        "peak_python_bytes": memory.get("peak_python_bytes"),
        **timing_summary(samples),
    }


def measure_lpt(jobs, machines=None, repeats=None, batch_seconds=None):
    machines = M if machines is None else machines
    repeats = LPT_REPEATS if repeats is None else repeats
    batch_seconds = BATCH_SECONDS if batch_seconds is None else batch_seconds
    lpt_solve(jobs, machines)
    batch = 1

    def batch_solve(batch_jobs, batch_machines):
        for _ in range(batch):
            result = lpt_solve(batch_jobs, batch_machines)
        return result

    while True:
        _, seconds = time_in_process(batch_solve, jobs, machines, 1)
        if seconds >= batch_seconds:
            break
        batch *= 2
    samples = []
    for _ in range(repeats):
        gc.collect()
        makespan, seconds = time_in_process(batch_solve, jobs, machines, 1)
        samples.append(seconds / batch)
    gc.collect()
    peak = peak_memory_in_process(lpt_solve, jobs, machines) * 1024
    return {
        "status": "ok", "error": "", "solved_at_least_once": True,
        "makespan": makespan, "states": None, "stats_status": "not_applicable",
        "stats_error": "", "cache_hits": None, "calls": None,
        "samples_seconds": json.dumps(samples), "requested_repeats": repeats,
        "completed_repeats": len(samples), "batch_calls": batch,
        "memory_status": "ok", "memory_error": "", "peak_python_bytes": peak,
        **timing_summary(samples),
    }


def main():
    if not 1 <= M <= N or min(DP_REPEATS, LPT_REPEATS, TIMEOUT, BATCH_SECONDS) <= 0:
        raise ValueError("require 1 <= M <= N and positive repeats/time limits")
    if TIME_RANGE < max(N, 3, N // M + 1):
        raise ValueError("TIME_RANGE is too small for the configured generators")
    path = Path(OUTPUT)
    path.parent.mkdir(parents=True, exist_ok=True)
    configs = [(family, seed) for family in FAMILIES for seed in SEEDS]
    random.Random(0).shuffle(configs)
    fields = [
        "family", "n", "m", "seed", "instance_seed", "jobs", "algorithm",
        "status", "error", "solved_at_least_once", "makespan", "optimum",
        "relative_error_percent", "states", "stats_status", "stats_error",
        "cache_hits", "calls", "requested_repeats", "completed_repeats",
        "batch_calls", "samples_seconds", "min_seconds", "median_seconds",
        "iqr_seconds", "memory_status", "memory_error", "peak_python_bytes",
    ]
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for family, seed in configs:
            instance_seed = family_seed(family, seed)
            jobs = FAMILIES[family](instance_seed)
            algorithms = ["DP", "LPT"]
            random.Random(f"{family}:{seed}").shuffle(algorithms)
            results = {}
            for algorithm in algorithms:
                results[algorithm] = (measure_dp(jobs.copy()) if algorithm == "DP"
                                      else measure_lpt(jobs.copy()))
            dp_result = results["DP"]
            optimum = TIME_RANGE if family == "balanced" else dp_result["makespan"]
            for algorithm in algorithms:
                result = results[algorithm]
                error = ((result["makespan"] - optimum) / optimum * 100
                         if optimum is not None and result["makespan"] is not None else None)
                writer.writerow({
                    "family": family, "n": N, "m": M, "seed": seed,
                    "instance_seed": instance_seed, "jobs": json.dumps(jobs),
                    "algorithm": algorithm, **result, "optimum": optimum,
                    "relative_error_percent": error,
                })
            output.flush()
            print(f"{family} seed={seed}: DP={dp_result['status']}; LPT=ok", flush=True)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
