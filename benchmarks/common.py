"""
Shared timing helpers for the experiment scripts

The DP can run for a very long time on larger inputs, so DP runs happen
in a separate process that is stopped after a time limit. The time is
measured inside that process around the solve call only, so starting
the process is not included in the result.
"""

import multiprocessing
import queue
import time
import tracemalloc

from algorithms.dp import solve as dp_solve
from algorithms.lpt import solve as lpt_solve

SOLVERS = {"dp": dp_solve, "lpt": lpt_solve}


def _worker(solver_name, jobs, m, measure_memory, results):
    """Runs in the child process: solve once and report the measurements."""
    solve = SOLVERS[solver_name]
    if measure_memory:
        tracemalloc.start()
    start = time.perf_counter()
    makespan, _ = solve(jobs, m)
    seconds = time.perf_counter() - start
    peak_kib = None
    if measure_memory:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_kib = peak / 1024
    results.put((makespan, seconds, peak_kib))


def run_with_timeout(solver_name, jobs, m, timeout, measure_memory=False):
    """
    Run one solver once in a separate process.
    Returns (makespan, seconds, peak_kib), or None if it did not finish
    within `timeout` seconds. peak_kib is None unless measure_memory is True.
    """
    ctx = multiprocessing.get_context("spawn")   # same behaviour on Windows and Mac
    results = ctx.Queue()
    process = ctx.Process(target=_worker,
                          args=(solver_name, jobs, m, measure_memory, results))
    process.start()
    try:
        result = results.get(timeout=timeout)
    except queue.Empty:
        result = None                           # did not finish in time
        process.terminate()
    process.join()
    return result


def time_in_process(solve, jobs, m, repeats):
    """Time a fast solver directly (no separate process). Returns (makespan, median seconds)."""
    times = []
    makespan = None
    for _ in range(repeats):
        start = time.perf_counter()
        makespan, _ = solve(jobs, m)
        times.append(time.perf_counter() - start)
    times.sort()
    return makespan, times[len(times) // 2]


def peak_memory_in_process(solve, jobs, m):
    """Peak memory (KiB) of one run of a fast solver, measured with tracemalloc."""
    tracemalloc.start()
    solve(jobs, m)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1024


def _states_worker(jobs, m, results):
    """Runs in the child process: solve once with the DP and report its cache stats."""
    stats = {}
    dp_solve(jobs, m, stats=stats)
    results.put(stats)


def dp_states_with_timeout(jobs, m, timeout):
    """
    Run the DP once in a separate process and return its cache statistics
    (states, cache_hits, calls), or None if it did not finish in time.
    """
    ctx = multiprocessing.get_context("spawn")
    results = ctx.Queue()
    process = ctx.Process(target=_states_worker, args=(jobs, m, results))
    process.start()
    try:
        stats = results.get(timeout=timeout)
    except queue.Empty:
        stats = None
        process.terminate()
    process.join()
    return stats
