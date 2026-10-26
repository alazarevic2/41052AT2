"""
LPT (Longest Processing Time first) for makespan scheduling
Author: Anna Lazarevic

Sorts jobs from longest to shortest, then assigns each job to the
machine with the smallest current load.
"""

def solve(jobs: list[int], m: int) -> tuple[int, list[list[int]]]:

    # jobs: a list of job times, e.g. [4, 7, 2, 5, 3, 6]
    # m: the number of machines
  
    # Each machine starts with 0 work and no jobs
    loads = [0] * m                         # e.g. [0, 0, 0]
    machines = [[] for _ in range(m)]     # e.g. [[], [], []]

    # Sort the jobs from longest to shortest
    sorted_jobs = sorted(jobs, reverse=True)

    for job in sorted_jobs:
        # Find the index of the machine with the smallest load
        i = loads.index(min(loads))
    
        # Give this job to machine i (update its load and add the job to its list)
        loads[i] += job
        machines[i].append(job)
        
    makespan = max(loads)
    return makespan, machines
