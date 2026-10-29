from algorithms.dp import exact_makespan
from benchmarks.dp_generators import known_optimum_jobs


def test_known_optimum_matches_hand_examples():
    assert known_optimum_jobs(2) == [3, 3, 2, 2, 2]
    assert known_optimum_jobs(3) == [5, 5, 4, 4, 3, 3, 3]


def test_dp_finds_known_optimum():
    # Keep exact-search cases small.
    for m in range(2, 5):
        makespan = exact_makespan(known_optimum_jobs(m), m)
        assert makespan == 3 * m
