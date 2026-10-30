"""
E9: LPT worst case
Author: Anna Lazarevic

Runs both algorithms on Graham's worst-case family for m = 2 to 10
machines. Checks that LPT's makespan divided by the optimum equals
exactly 4/3 - 1/(3m), and records both runtimes
"""

import csv
import os
import random
import statistics
import time
from fractions import Fraction

from algorithms.dp import solve as dp_solve
from algorithms.lpt import solve as lpt_solve
from benchmarks.generators import lpt_worst_case

MACHINES = range(2, 11)                 # m = 2 to 10
LPT_REPEATS = 11                        # LPT is fast, so more repeats
DP_REPEATS = 3                          # DP is slow for large m
ORDER_SEED = 0                          # fixed seed for shuffling the run order
OUTPUT = "benchmarks/results/e9_lpt_worst_case.csv"


def time_runs(solver, jobs, m, repeats):
    """Run solver `repeats` times; return (makespan, median seconds)."""
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        makespan, _ = solver(jobs, m)
        times.append(time.perf_counter() - start)
    return makespan, statistics.median(times)


def main():
    os.makedirs("benchmarks/results", exist_ok=True)

    # Shuffle the run order so no setting always runs first
    order = list(MACHINES)
    random.Random(ORDER_SEED).shuffle(order)

    rows = []
    for m in order:
        jobs = lpt_worst_case(m)

        lpt_solve(jobs, m)   # warm-up run, not timed

        opt, dp_seconds = time_runs(dp_solve, jobs, m, DP_REPEATS)
        lpt, lpt_seconds = time_runs(lpt_solve, jobs, m, LPT_REPEATS)

        # Exact fractions so the comparison has no rounding errors
        ratio = Fraction(lpt, opt)
        bound = Fraction(4, 3) - Fraction(1, 3 * m)

        rows.append([m, len(jobs), opt, lpt, float(ratio), float(bound),
                     ratio == bound, dp_seconds, lpt_seconds])
        print(f"m={m:>2}  OPT={opt:>3}  LPT={lpt:>3}  ratio={float(ratio):.4f}  "
              f"bound={float(bound):.4f}  matches={ratio == bound}  "
              f"DP={dp_seconds:.4f}s  LPT={lpt_seconds * 1000:.4f}ms")

    rows.sort()   # save in order of m

    with open(OUTPUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["m", "n", "opt", "lpt_makespan", "ratio", "bound",
                         "ratio_equals_bound", "dp_median_seconds",
                         "lpt_median_seconds"])
        writer.writerows(rows)


if __name__ == "__main__":
    main()
