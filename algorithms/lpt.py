"""
LPT (Longest Processing Time first) for makespan scheduling
Author: Anna Lazarevic

Sorts jobs from longest to shortest, then assigns each job to the
machine with the smallest current load.
"""
import heapq

def solve(jobs: list[int], m: int) -> tuple[int, list[list[int]]]:

    # jobs: a list of job times, e.g. [4, 7, 2, 5, 3, 6]
    # m: the number of machines
    
    if m < 1:
        raise ValueError("m must be at least 1")

    if any(job < 0 for job in jobs):
        raise ValueError("job times must be non-negative")
        
    
    machines = [[] for _ in range(m)]     # e.g. [[], [], []]

    # Heap of (load, machine index); every machine starts with load 0
    heap = [(0, i) for i in range(m)]
    heapq.heapify(heap)

    for job in sorted(jobs, reverse=True):
        # Take the least loaded machine
        load, i = heapq.heappop(heap)
        machines[i].append(job)
        heapq.heappush(heap, (load + job, i))

        
    makespan = max(load for load, i in heap)
    return makespan, machines
