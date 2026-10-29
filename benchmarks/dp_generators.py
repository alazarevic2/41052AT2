"""Input generators for exact DP experiments."""


def known_optimum_jobs(m):
    """Generate jobs with optimal makespan 3m, for m >= 2 machines.

    This is the same mathematical family used for the LPT comparison:
    two jobs of each size from m+1 to 2m-1, plus three jobs of size m.
    """
    jobs = []
    for size in range(2 * m - 1, m, -1):
        jobs += [size, size]
    jobs += [m, m, m]
    return jobs
