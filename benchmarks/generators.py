"""
Input generators for makespan scheduling experiments
Author: Anna Lazarevic
"""


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