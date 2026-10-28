from functools import lru_cache


def exact_makespan(jobs, machines):
    """
    Basic exact DP for identical-machine makespan scheduling.
    Author: Thomas Page
    Tries assigning each job to every machine and memoises states
    that have already been solved.
    """

    @lru_cache(maxsize=None)
    def dp(job_index, loads):
        # All jobs have been assigned
        if job_index == len(jobs):
            return max(loads)

        best = float("inf")
        job = jobs[job_index]

        # Try putting the current job on every machine
        for machine in range(machines):
            new_loads = list(loads)
            new_loads[machine] += job

            result = dp(
                job_index + 1,
                tuple(new_loads)
                )

            best = min(best, result)

        return best

    initial_loads = tuple([0] * machines)

    return dp(0, initial_loads)


