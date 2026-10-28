from algorithms.lpt import solve
from benchmarks.generators import lpt_worst_case


def test_worst_case_matches_hand_examples():
    assert lpt_worst_case(2) == [3, 3, 2, 2, 2]
    assert lpt_worst_case(3) == [5, 5, 4, 4, 3, 3, 3]


def test_lpt_hits_worst_case_bound():
    for m in range(2, 11):
        makespan, machines = solve(lpt_worst_case(m), m)
        assert makespan == 4 * m - 1