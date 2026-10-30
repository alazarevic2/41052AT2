"""Shared experiments: pytest tests/test_performance.py -s -q
Author: Thomas Page
Author: Anna Lazarevic
Times full solve calls with fresh caches; measures peak memory separately.
Setup and validation are excluded from measurements.
"""

import gc
import tracemalloc
from statistics import median
from time import perf_counter

import pytest

from algorithms.dp import solve as dp_solve
from algorithms.lpt import solve as lpt_solve
from benchmarks.dp_generators import known_optimum_jobs
from benchmarks.generators import (
    balanced, few_large_many_small, highly_varied, lpt_worst_case,
    repeated_sizes, reordered, scaled, similar_sized, uniform,
)


SEED = 67
REPEATS = 5
FAMILIES = {
    "uniform": lambda n, m: uniform(n, 100, seed=SEED),
    "repeated": lambda n, m: repeated_sizes(n, seed=SEED),
    "varied": lambda n, m: highly_varied(n, seed=SEED),
    "large_small": lambda n, m: few_large_many_small(n, seed=SEED),
    "similar": lambda n, m: similar_sized(n, seed=SEED),
    "balanced": lambda n, m: balanced(n, m, seed=SEED, target=100),
}


def measure(solver, jobs, machines):
    times = []
    makespans = []
    for _ in range(REPEATS):
        gc.collect()
        input_jobs = jobs.copy()
        start = perf_counter()
        makespan, assignments = solver(input_jobs, machines)
        times.append((perf_counter() - start) * 1000)

        # Validate both solvers' schedules without including checks in timings.
        assert len(assignments) == machines
        assert sorted(job for machine in assignments for job in machine) == sorted(jobs)
        assert max(map(sum, assignments)) == makespan
        assert input_jobs == jobs
        makespans.append(makespan)

    assert len(set(makespans)) == 1
    return median(times), makespans[0]


def measure_peak_memory(solver, jobs, machines, expected):
    gc.collect()
    input_jobs = jobs.copy()
    tracemalloc.start()
    try:
        makespan, assignments = solver(input_jobs, machines)
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert makespan == expected
    return peak_bytes / 1024


@pytest.fixture(scope="module")
def table_header():
    print(f"\nSeed={SEED}; median of {REPEATS} runs; times in milliseconds.")
    print("Memory: peak Python allocations in KiB from one separate call per solver.")
    print(f"{'Family':<13} {'Jobs':>4} {'M':>3} {'DP ms':>11} {'LPT ms':>11} "
          f"{'DP KiB':>11} {'LPT KiB':>11} "
          f"{'OPT':>6} {'LPT':>6} {'Ratio':>7}")


def compare(family, jobs, machines):
    dp_ms, optimum = measure(dp_solve, jobs, machines)
    lpt_ms, approximate = measure(lpt_solve, jobs, machines)
    dp_kib = measure_peak_memory(dp_solve, jobs, machines, optimum)
    lpt_kib = measure_peak_memory(lpt_solve, jobs, machines, approximate)
    assert approximate >= optimum
    ratio = approximate / optimum
    print(f"{family:<13} {len(jobs):>4} {machines:>3} {dp_ms:>11.4f} "
          f"{lpt_ms:>11.4f} {dp_kib:>11.2f} {lpt_kib:>11.2f} "
          f"{optimum:>6} {approximate:>6} {ratio:>7.3f}")
    return optimum, approximate


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("machines", [2, 3])
@pytest.mark.parametrize("size", [6, 10, 12])
def test_shared_performance(family, machines, size, table_header):
    jobs = FAMILIES[family](size, machines)
    optimum, _ = compare(family, jobs, machines)
    if family == "balanced":
        assert optimum == 100


@pytest.mark.parametrize("machines", [2, 3, 4])
@pytest.mark.parametrize("family, generator", [
    ("LPT_worst", lpt_worst_case),
    ("DP_known", known_optimum_jobs),
])
def test_known_optimum_performance(family, generator, machines, table_header):
    optimum, approximate = compare(family, generator(machines), machines)
    assert optimum == 3 * machines
    assert approximate == 4 * machines - 1


@pytest.mark.parametrize("factor", [10, 100])
@pytest.mark.parametrize("machines", [2, 3])
@pytest.mark.parametrize("size", [6, 10, 12])
def test_scaled_performance(factor, machines, size, table_header):
    jobs = uniform(size, 100, seed=SEED)
    # Reference calls are outside the measurements.
    baseline_optimum, _ = dp_solve(jobs, machines)
    baseline_lpt, _ = lpt_solve(jobs, machines)
    optimum, approximate = compare(f"scaled_x{factor}", scaled(jobs, factor), machines)
    assert optimum == baseline_optimum * factor
    assert approximate == baseline_lpt * factor


@pytest.mark.parametrize("order", ["original", "ascending", "descending", "shuffled"])
@pytest.mark.parametrize("machines", [2, 3])
@pytest.mark.parametrize("size", [6, 10, 12])
def test_reordered_performance(order, machines, size, table_header):
    jobs = uniform(size, 100, seed=SEED)
    baseline_optimum, _ = dp_solve(jobs, machines)
    baseline_lpt, _ = lpt_solve(jobs, machines)
    ordered_jobs = reordered(jobs, order, seed=SEED)
    optimum, approximate = compare(order, ordered_jobs, machines)
    assert optimum == baseline_optimum
    assert approximate == baseline_lpt
