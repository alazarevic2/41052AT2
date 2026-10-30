"""
E1: Scaling with n and m

Runs both algorithms on uniform random instances (job times 1 to 100)
for m = 2, 3, 4 machines and n = 4 to 40 jobs, 10 instances each
Records runtime, peak memory and makespans. DP runs are stopped after
10 seconds and recorded as "did not finish"

Run order is shuffled so no setting always runs first. Rows are written
as they finish, so partial results are kept if the run is interrupted
"""

import csv
import os
import random

from algorithms.lpt import solve as lpt_solve
from benchmarks.common import (peak_memory_in_process, run_with_timeout,
                               time_in_process)
from benchmarks.generators import uniform

MACHINES = [2, 3, 4]
SIZES = range(4, 41, 4)         # n = 4, 8, ..., 40
SEEDS = range(10)               # 10 instances per (n, m)
MAX_TIME = 100                  # job times drawn from 1 to 100
TIMEOUT = 10                    # seconds before a DP run is stopped
DP_REPEATS = 3                  # timed DP runs per instance...
SLOW_RUN = 1.0                  # ...unless the first takes over 1 s
LPT_REPEATS = 11
ORDER_SEED = 0
OUTPUT = "benchmarks/results/e1_scaling.csv"


def lower_bound(jobs, m):
    """Larger of the longest job and the average load (rounded up)"""
    return max(max(jobs, default=0), (sum(jobs) + m - 1) // m)


def time_dp(jobs, m):
    """Returns (makespan, median seconds) or None if any run timed out"""
    times = []
    makespan = None
    for _ in range(DP_REPEATS):
        result = run_with_timeout("dp", jobs, m, TIMEOUT)
        if result is None:
            return None
        makespan, seconds, _ = result
        times.append(seconds)
        if seconds > SLOW_RUN:      # slow runs vary little, so one is enough
            break
    times.sort()
    return makespan, times[len(times) // 2]


def main():
    os.makedirs("benchmarks/results", exist_ok=True)

    configs = [(m, n, s) for m in MACHINES for n in SIZES for s in SEEDS]
    random.Random(ORDER_SEED).shuffle(configs)

    with open(OUTPUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["m", "n", "seed", "dp_finished", "opt",
                         "lpt_makespan", "lower_bound",
                         "opt_equals_lower_bound", "dp_median_seconds",
                         "lpt_median_seconds", "dp_peak_kib", "lpt_peak_kib"])

        for count, (m, n, seed) in enumerate(configs, start=1):
            jobs = uniform(n, MAX_TIME, seed)
            bound = lower_bound(jobs, m)

            # LPT: fast, so timed directly in this process
            lpt_solve(jobs, m)   # warm-up run, not timed
            lpt, lpt_seconds = time_in_process(lpt_solve, jobs, m, LPT_REPEATS)
            lpt_kib = peak_memory_in_process(lpt_solve, jobs, m)

            # DP: separate process with a time limit
            dp = time_dp(jobs, m)
            if dp is None:
                row = [m, n, seed, False, "", lpt, bound, "", "",
                       lpt_seconds, "", lpt_kib]
                status = "DNF"
            else:
                opt, dp_seconds = dp
                memory = run_with_timeout("dp", jobs, m, TIMEOUT,
                                          measure_memory=True)
                dp_kib = memory[2] if memory else ""
                row = [m, n, seed, True, opt, lpt, bound, opt == bound,
                       dp_seconds, lpt_seconds, dp_kib, lpt_kib]
                status = f"{dp_seconds:.4f}s"

            writer.writerow(row)
            f.flush()   # keep results so far if the run is interrupted
            print(f"[{count:>3}/{len(configs)}] m={m} n={n:>2} seed={seed}  "
                  f"DP={status}")


if __name__ == "__main__":
    main()
