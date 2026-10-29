from itertools import product
from random import Random

import pytest

from algorithms.dp import exact_makespan


def test_balanced_two_machines():
    assert exact_makespan([2, 3, 4, 5], 2) == 7


def test_one_machine():
    assert exact_makespan([2, 3, 4, 5], 1) == 14


def test_single_job():
    assert exact_makespan([10], 2) == 10


def test_three_machines():
    assert exact_makespan([3, 3, 3, 3, 3, 3], 3) == 6


@pytest.mark.parametrize("machines", [0, -1])
def test_invalid_machine_count(machines):
    with pytest.raises(ValueError, match="machines must be greater than 0"):
        exact_makespan([2, 3, 4], machines)


def test_negative_job_time():
    with pytest.raises(ValueError, match="job processing times cannot be negative"):
        exact_makespan([2, -3, 4], 2)


def test_zero_job_time():
    assert exact_makespan([0, 2, 3], 2) == 3


def test_empty_jobs():
    assert exact_makespan([], 2) == 0


def test_more_machines_than_jobs():
    assert exact_makespan([4, 7], 5) == 7


def test_longest_job_determines_makespan():
    assert exact_makespan([10, 2, 3], 2) == 10


@pytest.mark.parametrize("jobs, machines, expected", [
    ([2, 2, 3], 2, 4),  # Rounded average is achievable.
    ([3, 3, 3], 2, 6),  # Rounded average is 5, but no schedule achieves it.
    ([0, 0, 0], 2, 0),
])
def test_lower_bound_cases(jobs, machines, expected):
    assert exact_makespan(jobs, machines) == expected


def test_matches_exhaustive_assignments():
    # Independent reference: enumerate every labelled-machine assignment.
    rng = Random(41052)
    for _ in range(30):
        jobs = [rng.randint(0, 8) for _ in range(rng.randint(0, 6))]
        machines = rng.randint(1, 3)
        expected = float("inf")
        for assignment in product(range(machines), repeat=len(jobs)):
            loads = [0] * machines
            for job, machine in zip(jobs, assignment):
                loads[machine] += job
            expected = min(expected, max(loads))

        assert exact_makespan(jobs, machines) == expected, (jobs, machines)
