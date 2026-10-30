"""E5: early stopping on/off. Run: python -m benchmarks.run_e5"""

import csv
import json
import random
from pathlib import Path

from algorithms.lpt import solve as lpt_solve
from benchmarks.generators import balanced
from benchmarks.process_helpers import dp_trial
from benchmarks.run_e4 import timing_summary


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
        if lower_bound(jobs) % M != 0:
            return jobs
    raise ValueError("could not generate an above-bound instance")


def measure(jobs, early_stop):
    trials = []
    for _ in range(REPEATS):
        trial = dp_trial(jobs, M, TIMEOUT, early_stop=early_stop)
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
        stats = dp_trial(jobs, M, TIMEOUT, collect_stats=True,
                         early_stop=early_stop)
        memory = dp_trial(jobs, M, TIMEOUT, memory=True,
                          early_stop=early_stop)
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
    if not 2 <= M < N or REPEATS < 1 or TIMEOUT <= 0 or TARGET < N // M + 1:
        raise ValueError("require 2 <= M < N, positive repeats/timeout and a sufficient target")
    path = Path(OUTPUT)
    path.parent.mkdir(parents=True, exist_ok=True)
    configs = [(family, seed) for family in ("balanced", "above_bound") for seed in SEEDS]
    random.Random(0).shuffle(configs)
    fields = [
        "family", "n", "m", "seed", "jobs", "early_stop", "lower_bound",
        "lpt_makespan", "status", "error", "solved_at_least_once", "makespan",
        "stats_status", "stats_error", "states", "cache_hits", "calls",
        "requested_repeats", "completed_repeats", "samples_seconds",
        "min_seconds", "median_seconds", "iqr_seconds", "memory_status",
        "memory_error", "peak_python_bytes",
    ]
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for family, seed in configs:
            jobs = make_jobs(family, seed)
            bound = lower_bound(jobs)
            reference = lpt_solve(jobs.copy(), M)[0]
            modes = [True, False]
            random.Random(f"{family}:{seed}").shuffle(modes)
            results = {enabled: measure(jobs, enabled) for enabled in modes}
            makespans = [result["makespan"] for result in results.values()
                         if result["makespan"] is not None]
            if len(set(makespans)) > 1:
                raise RuntimeError(f"early-stop modes disagree for {family} seed {seed}")
            if makespans:
                optimum = makespans[0]
                if family == "balanced" and optimum != bound:
                    raise RuntimeError(f"balanced instance missed its bound for seed {seed}")
                if family == "above_bound" and optimum <= bound:
                    raise RuntimeError(f"above-bound proof failed for seed {seed}")
                if reference < optimum:
                    raise RuntimeError(f"LPT beat the optimum for {family} seed {seed}")
            for enabled in modes:
                writer.writerow({
                    "family": family, "n": N, "m": M, "seed": seed,
                    "jobs": json.dumps(jobs), "early_stop": enabled,
                    "lower_bound": bound, "lpt_makespan": reference,
                    **results[enabled],
                })
            output.flush()
            print(f"{family} seed={seed}: "
                  f"on={results[True]['status']} off={results[False]['status']}", flush=True)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
