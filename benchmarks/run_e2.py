"""
E2: Job magnitude
Author: Anna Lazarevic

Part A: fixed n = 12 jobs on m = 3 machines, with job times drawn from
1 up to 10, 100, 1,000 and 10,000. Tests whether larger job times slow
the DP (more distinct load states) but leave LPT unaffected.

Part B: the same base instances (job times 1 to 100) multiplied by
1, 2, 5 and 10. Tests whether scaling every job by the same factor
leaves the DP's state count unchanged.

Run order is shuffled, and DP runs are stopped after 10 seconds.
"""

import csv
import os
import random

from algorithms.lpt import solve as lpt_solve
from benchmarks.common import (dp_states_with_timeout, run_with_timeout,
                               time_in_process)
from benchmarks.generators import scaled, uniform

N = 12
M = 3
SEEDS = range(10)
MAX_TIMES = [10, 100, 1_000, 10_000]     # Part A
FACTORS = [1, 2, 5, 10]                  # Part B
BASE_MAX_TIME = 100                      # base instances for Part B
TIMEOUT = 10
DP_REPEATS = 3
SLOW_RUN = 1.0
LPT_REPEATS = 11
ORDER_SEED = 0
OUTPUT = "benchmarks/results/e2_job_magnitude.csv"


def time_dp(jobs, m):
    """Returns (makespan, median seconds) or None if any run timed out."""
    times = []
    makespan = None
    for _ in range(DP_REPEATS):
        result = run_with_timeout("dp", jobs, m, TIMEOUT)
        if result is None:
            return None
        makespan, seconds, _ = result
        times.append(seconds)
        if seconds > SLOW_RUN:
            break
    times.sort()
    return makespan, times[len(times) // 2]


def main():
    os.makedirs("benchmarks/results", exist_ok=True)

    # Each config: (part, max_time or factor, seed)
    configs = ([("A", mt, s) for mt in MAX_TIMES for s in SEEDS] +
               [("B", fa, s) for fa in FACTORS for s in SEEDS])
    random.Random(ORDER_SEED).shuffle(configs)

    with open(OUTPUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["part", "max_time", "factor", "n", "m", "seed",
                         "dp_finished", "opt", "lpt_makespan",
                         "dp_median_seconds", "lpt_median_seconds",
                         "dp_states", "dp_cache_hits", "dp_calls"])

        for count, (part, value, seed) in enumerate(configs, start=1):
            if part == "A":
                max_time, factor = value, 1
                jobs = uniform(N, max_time, seed)
            else:
                max_time, factor = BASE_MAX_TIME * value, value
                jobs = scaled(uniform(N, BASE_MAX_TIME, seed), factor)

            lpt_solve(jobs, M)   # warm-up run, not timed
            lpt, lpt_seconds = time_in_process(lpt_solve, jobs, M, LPT_REPEATS)

            dp = time_dp(jobs, M)
            stats = dp_states_with_timeout(jobs, M, TIMEOUT) if dp else None

            if dp is None:
                row = [part, max_time, factor, N, M, seed, False, "", lpt,
                       "", lpt_seconds, "", "", ""]
                status = "DNF"
            else:
                opt, dp_seconds = dp
                states = stats["states"] if stats else ""
                hits = stats["cache_hits"] if stats else ""
                calls = stats["calls"] if stats else ""
                row = [part, max_time, factor, N, M, seed, True, opt, lpt,
                       dp_seconds, lpt_seconds, states, hits, calls]
                status = f"{dp_seconds:.4f}s, {states} states"

            writer.writerow(row)
            f.flush()
            print(f"[{count:>2}/{len(configs)}] part {part}  max={max_time:>6}  "
                  f"x{factor:<2} seed={seed}  DP={status}")


if __name__ == "__main__":
    main()
