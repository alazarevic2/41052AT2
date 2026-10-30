"""
E6: cumulative DP optimisation variants

Compares uncached recursion, caching, sorted load states, equal-load
skipping, and lower-bound early stopping on the same small instances.
Records optimal-value search only; assignment reconstruction is excluded.
"""

import csv
import gc
import json
import os
import random
import statistics
import time
import tracemalloc

from benchmarks.dp_variants import VARIANTS, solve
from benchmarks.generators import balanced, repeated_sizes, uniform
from benchmarks.run_e4 import fresh_process_trial


N = 10                                      # jobs per instance
M = 3                                       # identical machines
SEEDS = [1, 2, 3, 4, 5]                    # 5 instances per family
REPEATS = 5                                 # timed searches per row
TIMEOUT = 10                                # seconds per search
MAX_TIME = 100                              # maximum generated job time
OUTPUT = "benchmarks/results/e6_dp_variants.csv"
FAMILIES = {
    "uniform": lambda seed: uniform(N, MAX_TIME, seed),
    "repeated": lambda seed: repeated_sizes(N, seed, max_time=MAX_TIME),
    "balanced": lambda seed: balanced(N, M, seed, target=MAX_TIME),
}


def _worker(connection, jobs, variant, memory):
    try:
        gc.collect()
        connection.send({"status": "ready"})
        if memory:
            tracemalloc.start()
        start = time.perf_counter()
        makespan, stats = solve(jobs, M, variant)
        seconds = time.perf_counter() - start
        peak = tracemalloc.get_traced_memory()[1] if memory else None
        if memory:
            tracemalloc.stop()
        connection.send({"status": "ok", "seconds": seconds,
                         "makespan": makespan, "peak_python_bytes": peak,
                         **stats})
    except Exception as error:
        connection.send({"status": "error",
                         "error": f"{type(error).__name__}: {error}"})
    finally:
        connection.close()


def run_once(jobs, variant, memory=False):
    return fresh_process_trial(_worker, (jobs, variant, memory), TIMEOUT)


def timing_summary(samples):
    quartiles = (statistics.quantiles(samples, n=4, method="inclusive")
                 if len(samples) > 1 else [samples[0]] * 3)
    return min(samples), statistics.median(samples), quartiles[2] - quartiles[0]


def measure(jobs, variant):
    trials = []
    for _ in range(REPEATS):
        trial = run_once(jobs, variant)
        trials.append(trial)
        if trial["status"] != "ok":
            break

    completed = [trial for trial in trials if trial["status"] == "ok"]
    samples = [trial["seconds"] for trial in completed]
    result = completed[-1] if completed else {}
    status = trials[-1]["status"]
    memory = run_once(jobs, variant, memory=True) if status == "ok" else {"status": "skipped"}
    minimum, median, iqr = timing_summary(samples) if samples else (None, None, None)
    return {
        "status": status,
        "error": trials[-1].get("error", ""),
        "makespan": result.get("makespan"),
        "calls": result.get("calls"),
        "evaluations": result.get("evaluations"),
        "cached_states": result.get("cached_states"),
        "cache_hits": result.get("cache_hits"),
        "completed_repeats": len(samples),
        "samples_seconds": json.dumps(samples),
        "min_seconds": minimum,
        "median_seconds": median,
        "iqr_seconds": iqr,
        "memory_status": memory["status"],
        "peak_python_bytes": memory.get("peak_python_bytes"),
    }


def main():
    os.makedirs("benchmarks/results", exist_ok=True)
    configs = [(family, seed) for family in FAMILIES for seed in SEEDS]
    random.Random(0).shuffle(configs)
    fields = ["family", "n", "m", "seed", "jobs", "variant", "status", "error",
              "makespan", "calls", "evaluations", "cached_states", "cache_hits",
              "completed_repeats", "samples_seconds", "min_seconds", "median_seconds",
              "iqr_seconds", "memory_status", "peak_python_bytes"]

    with open(OUTPUT, "w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for family, seed in configs:
            jobs = FAMILIES[family](seed)
            variants = VARIANTS.copy()
            random.Random(f"{family}:{seed}").shuffle(variants)
            optimum = None
            for variant in variants:
                result = measure(jobs, variant)
                if result["makespan"] is not None:
                    if optimum is None:
                        optimum = result["makespan"]
                    elif result["makespan"] != optimum:
                        raise RuntimeError(f"variants disagree for {family} seed {seed}")
                writer.writerow({"family": family, "n": N, "m": M, "seed": seed,
                                 "jobs": json.dumps(jobs), "variant": variant, **result})
                output.flush()
                print(f"{family} seed={seed} {variant}: {result['status']}", flush=True)

    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
