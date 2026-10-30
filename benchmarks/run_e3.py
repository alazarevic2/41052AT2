"""
E3: LPT solution quality vs jobs per machine

For m = 2 and 3 machines and 1 to 8 jobs per machine, runs LPT and the
DP on 100 random instances per setting (job times 1 to 50)
Records LPT's relative error compared with the optimum found by the DP

This experiment measures solution quality only, not time, so run order
does not affect the results and is not shuffled
"""

import csv
import os
import statistics

from algorithms.dp import solve as dp_solve
from algorithms.lpt import solve as lpt_solve
from benchmarks.generators import uniform

MACHINES = [2, 3]
JOBS_PER_MACHINE = range(1, 9)              # n/m from 1 to 8
SEEDS = range(100)                          # 100 instances per setting
MAX_TIME = 50                               # job times drawn from 1 to 50
OUTPUT = "benchmarks/results/e3_lpt_quality.csv"


def relative_error(lpt, opt):
    """LPT's makespan above the optimum, as a percentage."""
    if opt == 0:
        return 0.0
    return (lpt - opt) / opt * 100


def main():
    os.makedirs("benchmarks/results", exist_ok=True)

    with open(OUTPUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["m", "n", "jobs_per_machine", "seed", "opt",
                         "lpt_makespan", "lpt_optimal", "relative_error_pct"])

        for m in MACHINES:
            for k in JOBS_PER_MACHINE:
                n = m * k
                errors = []
                for seed in SEEDS:
                    jobs = uniform(n, MAX_TIME, seed)
                    opt, _ = dp_solve(jobs, m)
                    lpt, _ = lpt_solve(jobs, m)
                    error = relative_error(lpt, opt)
                    errors.append(error)
                    writer.writerow([m, n, k, seed, opt, lpt, lpt == opt,
                                     round(error, 4)])

                optimal_pct = sum(e == 0 for e in errors) / len(errors) * 100
                print(f"m={m}  n={n:>2}  n/m={k}  "
                      f"optimal={optimal_pct:5.1f}%  "
                      f"avg error={statistics.mean(errors):5.2f}%  "
                      f"max error={max(errors):5.2f}%")


if __name__ == "__main__":
    main()
