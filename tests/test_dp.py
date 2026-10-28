from algorithms.dp import exact_makespan


def test_balanced_two_machines():
    assert exact_makespan([2, 3, 4, 5], 2) == 7


def test_one_machine():
    assert exact_makespan([2, 3, 4, 5], 1) == 14


def test_single_job():
    assert exact_makespan([10], 2) == 10


def test_three_machines():
    assert exact_makespan([3, 3, 3, 3, 3, 3], 3) == 6
