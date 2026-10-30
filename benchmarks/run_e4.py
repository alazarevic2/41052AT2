"""E4: input structure. Run: python -m benchmarks.run_e4"""

import csv
import gc
import json
import multiprocessing
import random
import statistics
import time
import tracemalloc
from pathlib import Path

from algorithms.dp import solve as dp_solve
from algorithms.lpt import solve as lpt_solve
from benchmarks.generators import (
    balanced, few_large_many_small, highly_varied, repeated_sizes,
    similar_sized, uniform,
)


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


def timing_summary(samples):
    if not samples:
        return {"min_seconds": None, "median_seconds": None, "iqr_seconds": None}
    q = statistics.quantiles(samples, n=4, method="inclusive") if len(samples) > 1 else [samples[0]] * 3
    return {"min_seconds": min(samples), "median_seconds": statistics.median(samples),
            "iqr_seconds": q[2] - q[0]}


def _dp_worker(connection, jobs, machines, memory, early_stop=True):
    try:
        gc.collect()
        stats = {}
        connection.send({"status": "ready"})
        if memory:
            tracemalloc.start()
        start = time.perf_counter()
        makespan, assignments = dp_solve(jobs, machines, stats=stats, early_stop=early_stop)
        elapsed = time.perf_counter() - start
        peak = tracemalloc.get_traced_memory()[1] if memory else None
        if memory:
            tracemalloc.stop()
        connection.send({"status": "ok", "seconds": elapsed, "makespan": makespan,
                         "peak_python_bytes": peak, **stats})
    except Exception as error:
        connection.send({"status": "error", "error": f"{type(error).__name__}: {error}"})
    finally:
        connection.close()


def fresh_process_trial(worker, args, timeout):
    """Run a ready-signalling worker with startup excluded from its time limit."""
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=worker, args=(sender, *args))
    process.start()
    sender.close()
    try:
        if not receiver.poll(30):
            return {"status": "startup_timeout"}
        ready = receiver.recv()
        if ready["status"] != "ready":
            return ready
        if not receiver.poll(timeout):
            return {"status": "DNF"}
        return receiver.recv()
    except EOFError:
        return {"status": "error", "error": "worker exited without a result"}
    finally:
        if process.is_alive():
            process.terminate()
        process.join()
        process.close()
        receiver.close()


def dp_trial(jobs, machines, memory=False, timeout=TIMEOUT, early_stop=True):
    """One solve per child. Startup is excluded from timing and the solve cap."""
    return fresh_process_trial(_dp_worker, (jobs, machines, memory, early_stop), timeout)


def measure_dp(jobs):
    trials = []
    for _ in range(DP_REPEATS):
        trial = dp_trial(jobs, M, timeout=TIMEOUT)
        trials.append(trial)
        if trial["status"] != "ok":
            break
    samples = [trial["seconds"] for trial in trials if trial["status"] == "ok"]
    completed = [trial for trial in trials if trial["status"] == "ok"]
    status = trials[-1]["status"]
    # A traced-call timeout is separate from a timed-call timeout.
    memory = dp_trial(jobs, M, memory=True, timeout=TIMEOUT) if status == "ok" else {"status": "skipped"}
    last = completed[-1] if completed else {}
    return {"status": status, "error": trials[-1].get("error", ""),
            "makespan": last.get("makespan"), "states": last.get("states"),
            "cache_hits": last.get("cache_hits"), "calls": last.get("calls"),
            "samples_seconds": json.dumps(samples), "completed_repeats": len(samples),
            "batch_calls": 1, "memory_status": memory["status"],
            "peak_python_bytes": memory.get("peak_python_bytes"), **timing_summary(samples)}


def measure_lpt(jobs):
    # Warm-up and calibration are not included in reported timing samples.
    lpt_solve(jobs, M)
    batch = 1
    while True:
        start = time.perf_counter()
        for _ in range(batch):
            lpt_solve(jobs, M)
        if time.perf_counter() - start >= BATCH_SECONDS:
            break
        batch *= 2
    samples = []
    for _ in range(LPT_REPEATS):
        gc.collect()
        start = time.perf_counter()
        for _ in range(batch):
            makespan, assignments = lpt_solve(jobs, M)
        samples.append((time.perf_counter() - start) / batch)
    gc.collect()
    tracemalloc.start()
    try:
        result = lpt_solve(jobs, M)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    return {"status": "ok", "error": "", "makespan": makespan, "states": None,
            "cache_hits": None, "calls": None, "samples_seconds": json.dumps(samples),
            "completed_repeats": len(samples), "batch_calls": batch,
            "memory_status": "ok", "peak_python_bytes": peak, **timing_summary(samples)}


def main():
    if not 1 <= M <= N or min(DP_REPEATS, LPT_REPEATS, TIMEOUT, BATCH_SECONDS) <= 0:
        raise ValueError("require 1 <= M <= N and positive repeats/time limits")
    if TIME_RANGE < max(N, 3, N // M + 1):
        raise ValueError("TIME_RANGE is too small for the configured generators")
    path = Path(OUTPUT)
    path.parent.mkdir(parents=True, exist_ok=True)
    configs = [(family, seed) for family in FAMILIES for seed in SEEDS]
    random.Random(0).shuffle(configs)
    fields = ["family", "n", "m", "seed", "jobs", "algorithm", "status", "error",
              "makespan", "optimum", "relative_error_percent", "states", "cache_hits", "calls",
              "completed_repeats", "batch_calls", "samples_seconds", "min_seconds",
              "median_seconds", "iqr_seconds", "memory_status", "peak_python_bytes"]
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for family, seed in configs:
            jobs = FAMILIES[family](seed)
            # Shuffle solver order too, without changing the input either receives.
            algorithms = ["DP", "LPT"]
            random.Random(f"{family}:{seed}").shuffle(algorithms)
            results = {}
            for algorithm in algorithms:
                results[algorithm] = measure_dp(jobs.copy()) if algorithm == "DP" else measure_lpt(jobs.copy())
            dp_result = results["DP"]
            optimum = TIME_RANGE if family == "balanced" else dp_result["makespan"]
            for algorithm in algorithms:
                result = results[algorithm]
                error = ((result["makespan"] - optimum) / optimum * 100
                         if optimum is not None and result["makespan"] is not None else None)
                writer.writerow({"family": family, "n": N, "m": M, "seed": seed,
                                 "jobs": json.dumps(jobs), "algorithm": algorithm, **result,
                                 "optimum": optimum, "relative_error_percent": error})
            output.flush()
            print(f"{family} seed={seed}: DP={dp_result['status']}; LPT=ok", flush=True)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
