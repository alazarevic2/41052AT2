from functools import lru_cache


def exact_makespan(jobs, machines):
    """
    Basic exact DP for identical-machine makespan scheduling.
    Author: Thomas Page
    Tries assigning each job to every machine and memoises states
    that have already been solved.
    """
    if machines <= 0:
        raise ValueError("machines must be greater than 0")

    if any(job < 0 for job in jobs):
        raise ValueError("job processing times cannot be negative")

    # Integer job times: round the average load up without using floats.
    average_bound = (sum(jobs) + machines - 1) // machines
    lower_bound = max(max(jobs, default=0), average_bound)

    @lru_cache(maxsize=None)
    def dp(job_index, loads):
        # All jobs have been assigned
        if job_index == len(jobs):
            return max(loads)

        best = float("inf")
        job = jobs[job_index]

        # Try one machine for each distinct load (loads are sorted).
        for machine in range(machines):
            if machine > 0 and loads[machine] == loads[machine - 1]:
                continue

            new_loads = list(loads)
            new_loads[machine] += job

            result = dp(
                job_index + 1,
                tuple(sorted(new_loads))
                )

            best = min(best, result)

            # No schedule can beat this bound, so this state is solved.
            if best == lower_bound:
                return best

        return best

    initial_loads = tuple([0] * machines)

    return dp(0, initial_loads)


