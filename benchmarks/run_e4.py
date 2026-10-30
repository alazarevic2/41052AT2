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
from benchmarks.common import (
    dp_states_with_timeout, peak_memory_in_process, run_with_timeout,
    time_in_process,
)
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


def measure_dp(jobs, machines=None, repeats=None, timeout=None):
    machines = M if machines is None else machines
    repeats = DP_REPEATS if repeats is None else repeats
    timeout = TIMEOUT if timeout is None else timeout
    trials = []
    for _ in range(repeats):
        trial = run_with_timeout("dp", jobs, machines, timeout)
        trials.append(trial)
        if trial is None:
            break
    completed = [trial for trial in trials if trial is not None]
    samples = [trial[1] for trial in completed]
    status = "ok" if trials[-1] is not None else "DNF"
    stats = memory = None
    if status == "ok":
        # Shared helpers collect statistics and memory in separate solves.
        stats = dp_states_with_timeout(jobs, machines, timeout)
        memory = run_with_timeout("dp", jobs, machines, timeout, measure_memory=True)
    stats_status = ("ok" if stats is not None else "DNF") if status == "ok" else "skipped"
    memory_status = ("ok" if memory is not None else "DNF") if status == "ok" else "skipped"
    stats = stats or {}
    return {"status": status, "error": "" if status == "ok" else "no result before timeout",
            "makespan": completed[-1][0] if completed else None, "states": stats.get("states"),
            "stats_status": stats_status, "cache_hits": stats.get("cache_hits"), "calls": stats.get("calls"),
            "samples_seconds": json.dumps(samples), "completed_repeats": len(samples),
            "batch_calls": 1, "memory_status": memory_status,
            "peak_python_bytes": memory[2] * 1024 if memory is not None else None,
            **timing_summary(samples)}


def measure_lpt(jobs, machines=None, repeats=None, batch_seconds=None):
    machines = M if machines is None else machines
    repeats = LPT_REPEATS if repeats is None else repeats
    batch_seconds = BATCH_SECONDS if batch_seconds is None else batch_seconds
    # Warm-up and calibration are not included in reported timing samples.
    lpt_solve(jobs, machines)
    batch = 1

    def batch_solve(jobs, machines):
        for _ in range(batch):
            result = lpt_solve(jobs, machines)
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
    return {"status": "ok", "error": "", "makespan": makespan, "states": None,
            "stats_status": "not_applicable", "cache_hits": None, "calls": None, "samples_seconds": json.dumps(samples),
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
              "makespan", "optimum", "relative_error_percent", "states", "stats_status", "cache_hits", "calls",
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
