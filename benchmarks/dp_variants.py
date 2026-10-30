"""Cumulative DP variants used by experiment E6."""

from functools import lru_cache


VARIANTS = [
    "recursion",
    "cache",
    "sorted_loads",
    "skip_equal_loads",
    "early_stop",
]


def solve(jobs, m, variant):
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

    calls = 0
    evaluations = 0

    def recurrence(job_index, loads, visit):
        nonlocal evaluations
        evaluations += 1
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
            best = min(best, visit(job_index + 1, tuple(new_loads)))
            if early_stop and best == lower_bound:
                break
        return best

    if use_cache:
        @lru_cache(maxsize=None)
        def cached(job_index, loads):
            return recurrence(job_index, loads, visit)

        def visit(job_index, loads):
            nonlocal calls
            calls += 1
            return cached(job_index, loads)

        makespan = visit(0, (0,) * m)
        cache = cached.cache_info()
        cache_hits = cache.hits
        cached_states = cache.currsize
    else:
        def visit(job_index, loads):
            nonlocal calls
            calls += 1
            return recurrence(job_index, loads, visit)

        makespan = visit(0, (0,) * m)
        cache_hits = 0
        cached_states = 0

    stats = {
        "calls": calls,
        "evaluations": evaluations,
        "cached_states": cached_states,
        "cache_hits": cache_hits,
    }
    return makespan, stats
