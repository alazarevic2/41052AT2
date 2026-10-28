import pytest

from algorithms.lpt import solve

def check_valid(jobs, makespan, machines):
    # Every job appears exactly once across all machines
    all_assigned = []
    for machine in machines:
        all_assigned.extend(machine)
    assert sorted(all_assigned) == sorted(jobs)

    # Makespan equals the largest machine total
    assert makespan == max(sum(machine) for machine in machines)


def test_1():
    jobs = [4, 7, 2, 5, 3, 6]
    makespan, machines = solve(jobs, 3)
    assert makespan == 9
    check_valid(jobs, makespan, machines)


def test_2():
    jobs = [5, 5, 4, 4, 3, 3, 3]
    makespan, machines = solve(jobs, 3)
    assert makespan == 11
    check_valid(jobs, makespan, machines)


def test_3():
    jobs = [3, 3, 2, 2, 2]
    makespan, machines = solve(jobs, 2)
    assert makespan == 7
    check_valid(jobs, makespan, machines)


# Edge cases
def test_one_machine_only():
    jobs = [4, 7, 2]
    makespan, machines = solve(jobs, 1)
    assert makespan == 13
    check_valid(jobs, makespan, machines)


def test_more_machines_than_jobs():
    jobs = [5, 3]
    makespan, machines = solve(jobs, 4)
    assert makespan == 5
    assert machines == [[5], [3], [], []]
    check_valid(jobs, makespan, machines)


def test_jobs_same_size():
    jobs = [2, 2, 2, 2]
    makespan, machines = solve(jobs, 2)
    assert makespan == 4
    check_valid(jobs, makespan, machines)


def test_no_jobs():
    jobs = []
    makespan, machines = solve(jobs, 3)
    assert makespan == 0
    check_valid(jobs, makespan, machines)


# Invalid input
def test_no_machines():
    with pytest.raises(ValueError):
        solve([4, 7], 0)


def test_negative():
    with pytest.raises(ValueError):
        solve([4, -3, 2], 2)