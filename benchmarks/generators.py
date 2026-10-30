"""
Input generators for makespan scheduling experiments
Author: Anna Lazarevic
Author: Thomas Page

Every random generator takes a seed, so the same seed always produces
the same instance and experiments can be reproduced exactly.

"""

import random


def uniform(n, max_time, seed, min_time=1):
    """
    n job times drawn uniformly from min_time to max_time (inclusive).
    Neutral baseline for runtime and solution quality.
    """
    rng = random.Random(seed)
    return [rng.randint(min_time, max_time) for _ in range(n)]


def repeated_sizes(n, seed, num_distinct=3, max_time=100):
    """
    n job times drawn from only num_distinct different values.
    Tests whether repeated sizes increase DP state reuse.
    """
    rng = random.Random(seed)
    sizes = rng.sample(range(1, max_time + 1), num_distinct)
    return [rng.choice(sizes) for _ in range(n)]


def highly_varied(n, seed, max_time=None):
    """
    n job times that are all different, from a wide range.
    Minimises DP state reuse.
    """
    if max_time is None:
        max_time = 100 * n
    if max_time < n:
        raise ValueError("max_time must be at least n for all times to differ")
    rng = random.Random(seed)
    return rng.sample(range(1, max_time + 1), n)


def few_large_many_small(n, seed, num_large=None,
                         large_range=(50, 100), small_range=(1, 10)):
    """
    A few large jobs plus many small ones.
    Tests whether small jobs help LPT balance the loads.
    By default, about 10% of the jobs are large (at least one).
    """
    if num_large is None:
        num_large = max(1, n // 10)
    rng = random.Random(seed)
    large = [rng.randint(*large_range) for _ in range(num_large)]
    small = [rng.randint(*small_range) for _ in range(n - num_large)]
    jobs = large + small
    rng.shuffle(jobs)
    return jobs


def similar_sized(n, seed, low=45, high=55):
    """
    n job times from a narrow range, so all jobs are similar in size.
    Leaves LPT little room to correct early imbalances.
    """
    rng = random.Random(seed)
    return [rng.randint(low, high) for _ in range(n)]


def balanced(n, m, seed, target=100):
    """
    n jobs that can be split perfectly across m machines, each with
    load exactly `target`, so the optimal makespan is guaranteed to be
    `target`, which equals the DP's lower bound.

    Each machine's target is cut into random positive pieces, then all
    pieces are shuffled together so the perfect split is hidden.
    Tests the DP's early stopping.
    """
    if n < m:
        raise ValueError("need at least one job per machine (n >= m)")
    base, extra = divmod(n, m)
    if target < base + 1:
        raise ValueError("target too small to split into that many pieces")

    rng = random.Random(seed)
    jobs = []
    for i in range(m):
        k = base + 1 if i < extra else base     # pieces for this machine
        cuts = sorted(rng.sample(range(1, target), k - 1))
        points = [0] + cuts + [target]
        jobs += [points[j + 1] - points[j] for j in range(k)]
    rng.shuffle(jobs)
    return jobs


def lpt_worst_case(m):
    """
    Graham's worst-case family for LPT on m machines

    Two jobs each of sizes 2m-1, 2m-2, ..., m+1, plus three jobs of size m
    Optimal makespan is 3m, but LPT gives 4m-1, so the ratio is
    exactly 4/3 - 1/(3m), the worst LPT can ever do

    Examples: m=2 gives [3, 3, 2, 2, 2], m=3 gives [5, 5, 4, 4, 3, 3, 3]
    """
    jobs = []
    for size in range(2 * m - 1, m, -1):
        jobs += [size, size]
    jobs += [m, m, m]
    return jobs


# Helpers for specific experiments
def scaled(jobs, factor):
    """Multiply every job time by factor (for E2)."""
    return [job * factor for job in jobs]


def reordered(jobs, order, seed=None):
    """
    Return the same jobs in a different order (for E7).
    order: "original", "ascending", "descending" or "shuffled".
    """
    if order == "original":
        return list(jobs)
    if order == "ascending":
        return sorted(jobs)
    if order == "descending":
        return sorted(jobs, reverse=True)
    if order == "shuffled":
        shuffled = list(jobs)
        random.Random(seed).shuffle(shuffled)
        return shuffled
    raise ValueError(f"unknown order: {order}")