"""Cumulative DP variants used by experiment E6."""

from functools import lru_cache


VARIANTS = [
    "recursion",
    "cache",
    "sorted_loads",
    "skip_equal_loads",
    "early_stop",
]


def solve(jobs, m, variant, *, collect_stats=True):
    """Return the optimal makespan and search counts for one E6 variant."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant: {variant}")
    if m <= 0:
        raise ValueError("machines must be greater than 0")
    if any(job < 0 for job in jobs):
        raise ValueError("job processing times cannot be negative")

    use_cache = variant != "recursion"
    sort_loads = variant in VARIANTS[2:]
    skip_equal = variant in VARIANTS[3:]
    early_stop = variant == "early_stop"
    lower_bound = max(max(jobs, default=0), (sum(jobs) + m - 1) // m)

    def search(job_index, loads):
        if job_index == len(jobs):
            return max(loads)

        best = float("inf")
        job = jobs[job_index]
        for machine in range(m):
            if skip_equal and machine > 0 and loads[machine] == loads[machine - 1]:
                continue
            new_loads = list(loads)
            new_loads[machine] += job
            if sort_loads:
                new_loads.sort()
            best = min(best, dp(job_index + 1, tuple(new_loads)))
            if early_stop and best == lower_bound:
                break
        return best

    dp = lru_cache(maxsize=None)(search) if use_cache else search
    makespan = dp(0, (0,) * m)
    if not collect_stats:
        return makespan, {}

    if use_cache:
        cache = dp.cache_info()
        cache_hits = cache.hits
        cached_states = cache.currsize
        evaluations = cache.misses
        calls = evaluations + cache_hits
    else:
        # Plain recursion explores the full m-ary tree, including its root.
        calls = len(jobs) + 1 if m == 1 else (m ** (len(jobs) + 1) - 1) // (m - 1)
        evaluations = calls
        cache_hits = 0
        cached_states = 0

    stats = {
        "calls": calls,
        "evaluations": evaluations,
        "cached_states": cached_states,
        "cache_hits": cache_hits,
    }
    return makespan, stats
