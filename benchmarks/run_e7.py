"""E7: DP performance vs job ordering. Run: python -m benchmarks.run_e7"""

import csv
import hashlib
import json
import random
from pathlib import Path

from benchmarks.generators import balanced, repeated_sizes, reordered, uniform
from benchmarks.process_helpers import dp_trial
from benchmarks.run_e4 import timing_summary


N = 12
M = 3
SEEDS = [1, 2, 3, 4, 5]
ORDERS = ["original", "ascending", "descending", "shuffled"]
REPEATS = 5
TIMEOUT = 10
MAX_TIME = 100
OUTPUT = "benchmarks/results/e7_job_ordering.csv"
FAMILIES = {
    "uniform": lambda seed: uniform(N, MAX_TIME, seed),
    "repeated": lambda seed: repeated_sizes(N, seed, max_time=MAX_TIME),
    "balanced": lambda seed: balanced(N, M, seed, target=MAX_TIME),
}


def family_seed(family, seed):
    value = hashlib.sha256(f"E7:{family}:{seed}".encode()).digest()[:8]
    return int.from_bytes(value, "big")


def lower_bound(jobs):
    return max(max(jobs), (sum(jobs) + M - 1) // M)


def measure(jobs):
    trials = []
    for _ in range(REPEATS):
        trial = dp_trial(jobs, M, TIMEOUT)
        trials.append(trial)
        if trial["status"] != "ok":
            break
    completed = [trial for trial in trials if trial["status"] == "ok"]
    samples = [trial["seconds"] for trial in completed]
    failure = next((trial for trial in trials if trial["status"] != "ok"), None)
    if len(completed) == REPEATS:
        status = "ok"
    elif completed:
        status = "partial"
    else:
        status = failure["status"]

    stats = memory = {"status": "skipped"}
    if status == "ok":
        stats = dp_trial(jobs, M, TIMEOUT, collect_stats=True)
        memory = dp_trial(jobs, M, TIMEOUT, memory=True)
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
        "stats_status": stats["status"],
        "stats_error": stats.get("error", ""),
        "states": stats.get("states"),
        "cache_hits": stats.get("cache_hits"),
        "calls": stats.get("calls"),
        "requested_repeats": REPEATS,
        "completed_repeats": len(samples),
        "samples_seconds": json.dumps(samples),
        "memory_status": memory["status"],
        "memory_error": memory.get("error", ""),
        "peak_python_bytes": memory.get("peak_python_bytes"),
        **timing_summary(samples),
    }


def main():
    if not 1 <= M <= N or REPEATS < 1 or TIMEOUT <= 0 or MAX_TIME < N:
        raise ValueError("invalid E7 configuration")
    path = Path(OUTPUT)
    path.parent.mkdir(parents=True, exist_ok=True)
    configs = [(family, seed) for family in FAMILIES for seed in SEEDS]
    random.Random(0).shuffle(configs)
    fields = [
        "family", "n", "m", "seed", "instance_seed", "base_jobs", "order",
        "jobs", "lower_bound", "optimum_equals_lower_bound", "status", "error",
        "solved_at_least_once", "makespan", "stats_status", "stats_error",
        "states", "cache_hits", "calls", "requested_repeats",
        "completed_repeats", "samples_seconds", "min_seconds", "median_seconds",
        "iqr_seconds", "memory_status", "memory_error", "peak_python_bytes",
    ]
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for family, seed in configs:
            instance_seed = family_seed(family, seed)
            base_jobs = FAMILIES[family](instance_seed)
            orders = ORDERS.copy()
            random.Random(f"{family}:{seed}").shuffle(orders)
            results = []
            for order in orders:
                jobs = reordered(base_jobs, order, seed=instance_seed)
                if sorted(jobs) != sorted(base_jobs):
                    raise RuntimeError(f"ordering changed the multiset for {family} seed {seed}")
                results.append((order, jobs, measure(jobs)))
            makespans = [result["makespan"] for _, _, result in results
                         if result["makespan"] is not None]
            if len(set(makespans)) > 1:
                raise RuntimeError(f"orderings disagree for {family} seed {seed}")
            optimum = makespans[0] if makespans else None
            bound = lower_bound(base_jobs)
            for order, jobs, result in results:
                writer.writerow({
                    "family": family, "n": N, "m": M, "seed": seed,
                    "instance_seed": instance_seed,
                    "base_jobs": json.dumps(base_jobs), "order": order,
                    "jobs": json.dumps(jobs), "lower_bound": bound,
                    "optimum_equals_lower_bound": (optimum == bound
                                                     if optimum is not None else None),
                    **result,
                })
            output.flush()
            statuses = ", ".join(f"{order}={result['status']}"
                                 for order, _, result in results)
            print(f"{family} seed={seed}: {statuses}", flush=True)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
