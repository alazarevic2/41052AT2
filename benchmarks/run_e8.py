"""
E8: LPT runtime at scale
Author: Anna Lazarevic

Times LPT on uniform random instances with n from 100 to 100,000
and saves the median runtime of each instance to a CSV file
"""

import csv
import os
import statistics
import time

from algorithms.lpt import solve
from benchmarks.generators import uniform

M = 10                                      # number of machines
SIZES = [100, 1_000, 10_000, 100_000]       # values of n to test
SEEDS = [1, 2, 3, 4, 5]                     # 5 different instances per size
REPEATS = 5                                 # timed runs per instance
MAX_TIMES = [100, 10_000, 1_000_000]        # job times drawn from 1 to each of these
OUTPUT = "benchmarks/results/e8_lpt_scale.csv"


def time_once(jobs, m):
    start = time.perf_counter()
    solve(jobs, m)
    end = time.perf_counter()
    return end - start

def main():
    os.makedirs("benchmarks/results", exist_ok=True)

    with open(OUTPUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n", "m", "max_time", "seed", "median_seconds"])

        for max_time in MAX_TIMES:
            for n in SIZES:
                for seed in SEEDS:
                    jobs = uniform(n, max_time, seed)

                    solve(jobs, M)   # warm-up run, not timed

                    times = [time_once(jobs, M) for _ in range(REPEATS)]
                    
                    median = statistics.median(times)
                    writer.writerow([n, M, max_time, seed, median])
                    print(f"max={max_time:>9} n={n:>7} seed={seed}  {median * 1000:.3f} ms")

if __name__ == "__main__":
    main()