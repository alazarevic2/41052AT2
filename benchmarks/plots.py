"""
Figures for the report, made from the experiment CSVs
Author: Anna Lazarevic

Figures 6-8, 10 and 11 plot results from Thomas Page's experiment
scripts (E4-E7 and E10).

Run from the repository root after the experiments:
    py -m benchmarks.plots
Saves PNG files to benchmarks/figures/.
"""

import csv
import os
import statistics
from collections import defaultdict
from fractions import Fraction

import matplotlib
matplotlib.use("Agg")           # save files without opening windows
import matplotlib.pyplot as plt

RESULTS = "benchmarks/results"
FIGURES = "benchmarks/figures"


def read(name):
    with open(os.path.join(RESULTS, name), newline="") as f:
        return list(csv.DictReader(f))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, name), dpi=200)
    plt.close(fig)
    print("saved", name)


def figure1_e1_runtime():
    """DP and LPT runtime vs n, one line per m, with timeouts marked."""
    rows = read("e1_scaling.csv")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colours = {2: "tab:blue", 3: "tab:orange", 4: "tab:green"}

    for m in sorted({int(r["m"]) for r in rows}):
        by_n = defaultdict(list)
        for r in rows:
            if int(r["m"]) == m:
                by_n[int(r["n"])].append(r)
        ns = sorted(by_n)

        dp_ns, dp_medians, dnf_ns = [], [], []
        for n in ns:
            finished = [float(r["dp_median_seconds"]) for r in by_n[n]
                        if r["dp_finished"] == "True"]
            if finished:
                dp_ns.append(n)
                dp_medians.append(statistics.median(finished))
            if any(r["dp_finished"] == "False" for r in by_n[n]):
                dnf_ns.append(n)
        lpt_medians = [statistics.median(float(r["lpt_median_seconds"])
                                         for r in by_n[n]) for n in ns]

        ax.plot(dp_ns, dp_medians, "o-", color=colours[m], label=f"DP, m = {m}")
        ax.plot(ns, lpt_medians, "s--", color=colours[m], alpha=0.6,
                label=f"LPT, m = {m}")
        if dnf_ns:
            ax.scatter(dnf_ns, [10] * len(dnf_ns), marker="x", s=60,
                       color=colours[m], label=f"DP timeout (10 s), m = {m}")

    ax.set_yscale("log")
    ax.set_xlabel("Number of jobs n")
    ax.set_ylabel("Median runtime (seconds, log scale)")
    ax.set_title("E1: Runtime vs number of jobs (median of 10 instances)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, which="major", alpha=0.3)
    save(fig, "fig1_e1_runtime.png")


def figure2_e3_quality():
    """LPT % optimal and average error vs jobs per machine."""
    rows = read("e3_lpt_quality.csv")
    fig, (left, right) = plt.subplots(1, 2, figsize=(9, 4))

    for m in sorted({int(r["m"]) for r in rows}):
        by_k = defaultdict(list)
        for r in rows:
            if int(r["m"]) == m:
                by_k[int(r["jobs_per_machine"])].append(r)
        ks = sorted(by_k)
        optimal = [100 * sum(r["lpt_optimal"] == "True" for r in by_k[k])
                   / len(by_k[k]) for k in ks]
        errors = [statistics.mean(float(r["relative_error_pct"])
                                  for r in by_k[k]) for k in ks]
        left.plot(ks, optimal, "o-", label=f"m = {m}")
        right.plot(ks, errors, "o-", label=f"m = {m}")

    left.set_title("Instances where LPT is optimal")
    left.set_xlabel("Jobs per machine (n / m)")
    left.set_ylabel("% of instances")
    left.set_ylim(0, 105)
    right.set_title("LPT's average error")
    right.set_xlabel("Jobs per machine (n / m)")
    right.set_ylabel("Average error above optimum (%)")
    for ax in (left, right):
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle("E3: LPT solution quality (100 random instances per point)")
    save(fig, "fig2_e3_quality.png")


def figure3_e2_states():
    """DP state count and cache hit rate vs maximum job time."""
    rows = [r for r in read("e2_job_magnitude.csv") if r["part"] == "A"]
    by_max = defaultdict(list)
    for r in rows:
        by_max[int(r["max_time"])].append(r)
    maxes = sorted(by_max)
    states = [statistics.median(int(r["dp_states"]) for r in by_max[t])
              for t in maxes]
    hit_rate = [100 * sum(int(r["dp_cache_hits"]) for r in by_max[t])
                / sum(int(r["dp_calls"]) for r in by_max[t]) for t in maxes]

    fig, left = plt.subplots(figsize=(7, 4.5))
    left.plot(maxes, states, "o-", color="tab:blue", label="DP states (median)")
    left.set_xscale("log")
    left.set_yscale("log")
    left.set_xlabel("Maximum job time (log scale)")
    left.set_ylabel("DP cached states (log scale)", color="tab:blue")

    right = left.twinx()
    right.plot(maxes, hit_rate, "s--", color="tab:red", label="Cache hit rate")
    right.set_ylabel("Cache hit rate (%)", color="tab:red")
    right.set_ylim(0, 50)

    left.set_title("E2: Effect of job magnitude on the DP (n = 12, m = 3)")
    left.grid(True, alpha=0.3)
    save(fig, "fig3_e2_states.png")


def figure4_e2_scaling():
    """State count per instance across scaling factors."""
    rows = [r for r in read("e2_job_magnitude.csv") if r["part"] == "B"]
    by_seed = defaultdict(dict)
    for r in rows:
        by_seed[int(r["seed"])][int(r["factor"])] = int(r["dp_states"])
    factors = sorted({int(r["factor"]) for r in rows})

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for seed in sorted(by_seed):
        counts = [by_seed[seed][f] for f in factors]
        unchanged = len(set(counts)) == 1
        ax.plot(factors, counts, marker="o",
                linestyle="-" if unchanged else "--",
                label=f"seed {seed}" + (" (unchanged)" if unchanged else ""))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(factors)
    ax.set_xticklabels([f"×{f}" for f in factors])
    ax.set_xlabel("Scaling factor applied to every job")
    ax.set_ylabel("DP cached states (log scale)")
    ax.set_title("E2: Scaling the same instances changes the DP's state count")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    save(fig, "fig4_e2_scaling.png")


def figure5_e9_worst_case():
    """LPT ratio vs the 4/3 - 1/(3m) bound, and DP runtime vs m."""
    rows = sorted(read("e9_lpt_worst_case.csv"), key=lambda r: int(r["m"]))
    ms = [int(r["m"]) for r in rows]
    ratios = [float(r["ratio"]) for r in rows]
    bound_ms = [m / 10 for m in range(15, 106)]
    bounds = [float(Fraction(4, 3)) - 1 / (3 * m) for m in bound_ms]

    fig, (left, right) = plt.subplots(1, 2, figsize=(9, 4))
    left.plot(bound_ms, bounds, "-", color="grey", label="4/3 − 1/(3m)")
    left.plot(ms, ratios, "o", color="tab:blue", label="LPT ÷ optimum")
    left.axhline(4 / 3, linestyle=":", color="black", label="4/3")
    left.set_xlabel("Number of machines m")
    left.set_ylabel("LPT makespan ÷ optimal makespan")
    left.set_title("LPT on its worst-case family")
    left.legend()

    right.plot(ms, [float(r["dp_median_seconds"]) for r in rows], "o-",
               label="DP")
    right.plot(ms, [float(r["lpt_median_seconds"]) for r in rows], "s--",
               label="LPT")
    right.set_yscale("log")
    right.set_xlabel("Number of machines m (n = 2m + 1 jobs)")
    right.set_ylabel("Median runtime (seconds, log scale)")
    right.set_title("Runtime on the same instances")
    right.legend()

    for ax in (left, right):
        ax.grid(True, alpha=0.3)
    fig.suptitle("E9: Graham's worst-case family")
    save(fig, "fig5_e9_worst_case.png")


def figure6_e4_structure():
    """E4 (Thomas): DP runtime, DP states and LPT error for each input family."""
    rows = read("e4_input_structure.csv")
    families = ["uniform", "repeated", "varied", "large_small", "similar", "balanced"]
    labels = ["uniform", "repeated", "varied", "few large +\nmany small", "similar", "balanced"]
    dp = {f: [r for r in rows if r["family"] == f and r["algorithm"] == "DP"] for f in families}
    lpt = {f: [r for r in rows if r["family"] == f and r["algorithm"] == "LPT"] for f in families}
    xs = range(len(families))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    panels = [
        (axes[0], dp, "median_seconds", "DP runtime (seconds, log scale)", True),
        (axes[1], dp, "states", "DP cached states (log scale)", True),
        (axes[2], lpt, "relative_error_percent", "LPT error above optimum (%)", False),
    ]
    for ax, data, column, ylabel, log in panels:
        for x, f in zip(xs, families):
            values = [float(r[column]) for r in data[f]]
            ax.scatter([x] * len(values), values, alpha=0.6, color="tab:blue")
            ax.scatter([x], [statistics.median(values)], marker="_", s=400,
                       color="black")
        ax.set_xticks(list(xs))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        if log:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
    axes[0].set_title("DP runtime")
    axes[1].set_title("DP states")
    axes[2].set_title("LPT error")
    fig.suptitle("E4: Input structure (n = 12, m = 3; dots = instances, bars = medians)")
    save(fig, "fig6_e4_structure.png")


def figure7_e5_early_stopping():
    """E5 (Thomas): DP runtime and states with early stopping on and off."""
    rows = read("e5_early_stopping.csv")
    families = ["balanced", "above_bound"]
    labels = ["optimum = lower bound\n(balanced)", "optimum > lower bound"]
    width = 0.35

    fig, (left, right) = plt.subplots(1, 2, figsize=(9, 4))
    for i, (setting, name) in enumerate([("False", "early stopping off"),
                                         ("True", "early stopping on")]):
        times, states = [], []
        for f in families:
            rs = [r for r in rows if r["family"] == f and r["early_stop"] == setting]
            times.append(statistics.median(float(r["median_seconds"]) for r in rs))
            states.append(statistics.median(int(r["states"]) for r in rs))
        xs = [x + (i - 0.5) * width for x in range(len(families))]
        left.bar(xs, times, width, label=name)
        right.bar(xs, states, width, label=name)

    for ax, ylabel in ((left, "Median DP runtime (seconds)"),
                       (right, "Median DP cached states")):
        ax.set_xticks(range(len(families)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("E5: Effect of early stopping (n = 12, m = 3)")
    save(fig, "fig7_e5_early_stopping.png")


def figure8_e6_variants():
    """E6 (Thomas): runtime and work as DP features are added one by one."""
    rows = read("e6_dp_variants.csv")
    variants = ["recursion", "cache", "sorted_loads", "skip_equal_loads", "early_stop"]
    labels = ["no cache", "+ cache", "+ sorted\nloads", "+ skip equal\nloads",
              "+ early\nstopping"]
    families = ["uniform", "repeated", "balanced"]

    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    for f in families:
        times, evals = [], []
        for v in variants:
            rs = [r for r in rows if r["family"] == f and r["variant"] == v]
            times.append(statistics.median(float(r["median_seconds"]) for r in rs))
            evals.append(statistics.median(int(r["evaluations"]) for r in rs))
        left.plot(range(len(variants)), times, "o-", label=f)
        right.plot(range(len(variants)), evals, "o-", label=f)

    for ax, ylabel in ((left, "Median DP runtime (seconds, log scale)"),
                       (right, "States evaluated (log scale)")):
        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("E6: DP features added cumulatively (n = 10, m = 3)")
    save(fig, "fig8_e6_variants.png")


def figure9_e8_lpt_scale():
    """E8: LPT runtime vs n, one line per job time range."""
    rows = read("e8_lpt_scale.csv")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for max_time in sorted({int(r["max_time"]) for r in rows}):
        by_n = defaultdict(list)
        for r in rows:
            if int(r["max_time"]) == max_time:
                by_n[int(r["n"])].append(float(r["median_seconds"]))
        ns = sorted(by_n)
        ax.plot(ns, [statistics.median(by_n[n]) for n in ns], "o-",
                label=f"job times 1-{max_time:,}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of jobs n (log scale)")
    ax.set_ylabel("Median LPT runtime (seconds, log scale)")
    ax.set_title("E8: LPT runtime at scale (m = 10, median of 5 instances)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    save(fig, "fig9_e8_lpt_scale.png")


def figure10_e7_job_order():
    """E7 (Thomas): DP runtime and states for each job order."""
    rows = read("e7_job_ordering.csv")
    orders = ["original", "ascending", "descending", "shuffled"]
    families = ["uniform", "repeated", "balanced"]

    # Check the optimal makespan is identical across orders for every instance
    makespans = defaultdict(set)
    for r in rows:
        makespans[(r["family"], r["seed"])].add(r["makespan"])
    unchanged = all(len(v) == 1 for v in makespans.values())

    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    for f in families:
        times, states = [], []
        for o in orders:
            rs = [r for r in rows if r["family"] == f and r["order"] == o]
            times.append(statistics.median(float(r["median_seconds"]) for r in rs))
            states.append(statistics.median(int(r["states"]) for r in rs))
        left.plot(range(len(orders)), times, "o-", label=f)
        right.plot(range(len(orders)), states, "o-", label=f)

    for ax, ylabel in ((left, "Median DP runtime (seconds, log scale)"),
                       (right, "Median DP cached states (log scale)")):
        ax.set_xticks(range(len(orders)))
        ax.set_xticklabels(orders)
        ax.set_xlabel("Order the jobs are given to the DP")
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    note = ("optimal makespan identical across orders in every instance"
            if unchanged else "WARNING: makespan differs between orders")
    fig.suptitle(f"E7: Effect of job order on the DP (n = 12, m = 3; {note})",
                 fontsize=10)
    save(fig, "fig10_e7_job_order.png")


def figure11_e10_feasibility():
    """E10 (Thomas): DP completion rate and runtime by family, n and m."""
    rows = read("e10_dp_feasibility_summary.csv")
    families = ["uniform", "repeated", "varied"]
    machines = sorted({int(r["m"]) for r in rows})

    fig, axes = plt.subplots(2, len(machines), figsize=(12, 6.5), sharex=True)
    for col, m in enumerate(machines):
        top, bottom = axes[0][col], axes[1][col]
        for f in families:
            rs = sorted((r for r in rows if r["family"] == f and int(r["m"]) == m),
                        key=lambda r: int(r["n"]))
            ns = [int(r["n"]) for r in rs]
            top.plot(ns, [100 * float(r["dp_solve_rate"]) for r in rs], "o-",
                     label=f)
            solved = [(int(r["n"]), float(r["dp_runtime_median_solved_only"]))
                      for r in rs if r["dp_runtime_median_solved_only"]]
            bottom.plot([n for n, _ in solved], [t for _, t in solved], "o-",
                        label=f)
        top.set_title(f"m = {m}")
        top.set_ylim(0, 105)
        bottom.set_yscale("log")
        bottom.set_xlabel("Number of jobs n")
        for ax in (top, bottom):
            ax.grid(True, alpha=0.3)
    axes[0][0].set_ylabel("DP finished within 10 s (%)")
    axes[1][0].set_ylabel("Median DP runtime, finished\nruns (seconds, log scale)")
    axes[0][0].legend(fontsize=8)
    fig.suptitle("E10: DP feasibility by input family (5 instances per point)")
    save(fig, "fig11_e10_feasibility.png")


def main():
    os.makedirs(FIGURES, exist_ok=True)
    figure1_e1_runtime()
    figure2_e3_quality()
    figure3_e2_states()
    figure4_e2_scaling()
    figure5_e9_worst_case()
    figure6_e4_structure()
    figure7_e5_early_stopping()
    figure8_e6_variants()
    figure9_e8_lpt_scale()
    figure10_e7_job_order()
    figure11_e10_feasibility()


if __name__ == "__main__":
    main()
