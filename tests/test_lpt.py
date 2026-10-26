from algorithms.lpt import solve
import pytest

def test_1():
    makespan, machines = solve([4, 7, 2, 5, 3, 6], 3)
    assert makespan == 9

def test_2():
    makespan, machines = solve([5, 5, 4, 4, 3, 3, 3], 3)
    assert makespan == 11

def test_3():
    makespan, machines = solve([3, 3, 2, 2, 2], 2)
    assert makespan == 7

# Testing edge cases
def test_one_machine_only():
    makespan, machines = solve([4, 7, 2], 1)
    assert makespan == 13

def test_more_machines_than_jobs():
    makespan, machines = solve([5, 3], 4)
    assert makespan == 5
    assert machines == [[5], [3], [] , []]

def test_job_same_size():
    makespan, machines = solve([2, 2, 2, 2], 2)
    assert makespan == 4

def test_no_jobs():
    makespan, machines = solve([], 3)
    assert makespan == 0

from algorithms.lpt import solve

def test_1():
    makespan, machines = solve([4, 7, 2, 5, 3, 6], 3)
    assert makespan == 9

def test_2():
    makespan, machines = solve([5, 5, 4, 4, 3, 3, 3], 3)
    assert makespan == 11

def test_3():
    makespan, machines = solve([3, 3, 2, 2, 2], 2)
    assert makespan == 7

# Testing edge cases
def test_one_machine_only():
    makespan, machines = solve([4, 7, 2], 1)
    assert makespan == 13

def test_more_machines_than_jobs():
    makespan, machines = solve([5, 3], 4)
    assert makespan == 5

def test_job_same_size():
    makespan, machines = solve([2, 2, 2, 2], 2)
    assert makespan == 4

def test_no_jobs():
    makespan, machines = solve([], 3)
    assert makespan == 0

def test_no_machines():
    with pytest.raises(ValueError):
        solve([4, 7], 0)

def test_negative():
    with pytest.raises(ValueError):
        solve([4, -3, 2], 2)