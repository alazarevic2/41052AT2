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
  