"""E5: early stopping on/off. Run: python -m benchmarks.run_e5"""

import csv
import json
import random
from pathlib import Path

from algorithms.lpt import solve as lpt_solve
from benchmarks.generators import balanced
from benchmarks.run_e4 import dp_trial, timing_summary


N = 12
M = 3
SEEDS = [1, 2, 3, 4, 5]
REPEATS = 3
TIMEOUT = 10
TARGET = 100
OUTPUT = "benchmarks/results/e5_early_stopping.csv"


def lower_bound(jobs):
    return max(max(jobs), (sum(jobs) + M - 1) // M)


def make_jobs(family, seed):
    if family == "balanced":
        return balanced(N, M, seed, target=TARGET)

    rng = random.Random(seed)
    for _ in range(1000):
        jobs = [M * rng.randint(1, 10) for _ in range(N)]
        # Every machine load is a multiple of M. A bound that is not a
        # multiple of M therefore cannot be the optimal makespan.
        if lower_bound(jobs) % M != 0:
            return jobs
    raise ValueError("could not generate an above-bound instance")


def measure(jobs, early_stop):
    trials = []
    for _ in range(REPEATS):
        trial = dp_trial(jobs, M, timeout=TIMEOUT, early_stop=early_stop)
        trials.append(trial)
        if trial["status"] != "ok":
            break
    completed = [trial for trial in trials if trial["status"] == "ok"]
    samples = [trial["seconds"] for trial in completed]
    last = completed[-1] if completed else {}
    status = trials[-1]["status"]
    memory = {"status": "skipped"}
    if status == "ok":
        memory = dp_trial(jobs, M, memory=True, timeout=TIMEOUT, early_stop=early_stop)
    return {"status": status, "error": trials[-1].get("error", ""),
            "makespan": last.get("makespan"), "states": last.get("states"),
            "cache_hits": last.get("cache_hits"), "calls": last.get("calls"),
            "completed_repeats": len(samples), "samples_seconds": json.dumps(samples),
            "memory_status": memory["status"], "peak_python_bytes": memory.get("peak_python_bytes"),
            **timing_summary(samples)}


def main():
    if not 2 <= M < N or REPEATS < 1 or TIMEOUT <= 0 or TARGET < N // M + 1:
        raise ValueError("require 2 <= M < N, positive repeats/timeout and a sufficient target")
    path = Path(OUTPUT)
    path.parent.mkdir(parents=True, exist_ok=True)
    configs = [(family, seed) for family in ("balanced", "above_bound") for seed in SEEDS]
    random.Random(0).shuffle(configs)
    fields = ["family", "n", "m", "seed", "jobs", "early_stop", "lower_bound", "lpt_makespan",
              "status", "error", "makespan", "states", "cache_hits", "calls", "completed_repeats",
              "samples_seconds", "min_seconds", "median_seconds", "iqr_seconds",
              "memory_status", "peak_python_bytes"]
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for family, seed in configs:
            jobs = make_jobs(family, seed)
            reference = lpt_solve(jobs.copy(), M)[0]
            modes = [True, False]
            random.Random(f"{family}:{seed}").shuffle(modes)
            for enabled in modes:
                result = measure(jobs, enabled)
                writer.writerow({"family": family, "n": N, "m": M, "seed": seed,
                                 "jobs": json.dumps(jobs), "early_stop": enabled,
                                 "lower_bound": lower_bound(jobs), "lpt_makespan": reference, **result})
                output.flush()
                print(f"{family} seed={seed} early_stop={enabled}: {result['status']}", flush=True)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
