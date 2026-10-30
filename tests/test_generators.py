from algorithms.lpt import solve
from benchmarks.generators import (
    lpt_worst_case, uniform, repeated_sizes, highly_varied, few_large_many_small,
    similar_sized, balanced, scaled, reordered,
)

def test_worst_case_matches_hand_examples():
    assert lpt_worst_case(2) == [3, 3, 2, 2, 2]
    assert lpt_worst_case(3) == [5, 5, 4, 4, 3, 3, 3]


def test_lpt_hits_worst_case_bound():
    for m in range(2, 11):
        makespan, machines = solve(lpt_worst_case(m), m)
        assert makespan == 4 * m - 1


def test_same_seed_gives_same_instance():
    assert uniform(20, 100, seed=1) == uniform(20, 100, seed=1)


def test_repeated_sizes_uses_few_values():
    assert len(set(repeated_sizes(50, seed=1, num_distinct=3))) <= 3


def test_highly_varied_all_different():
    jobs = highly_varied(30, seed=1)
    assert len(set(jobs)) == 30


def test_few_large_many_small_counts():
    jobs = few_large_many_small(20, seed=1)
    assert len(jobs) == 20
    assert sum(1 for job in jobs if job >= 50) == 2


def test_similar_sized_in_range():
    assert all(45 <= job <= 55 for job in similar_sized(30, seed=1))


def test_balanced_has_perfect_split():
    jobs = balanced(10, 3, seed=1, target=100)
    assert len(jobs) == 10
    assert sum(jobs) == 3 * 100
    assert all(job > 0 for job in jobs)


def test_scaled_and_reordered():
    assert scaled([1, 2, 3], 5) == [5, 10, 15]
    assert reordered([2, 3, 1], "descending") == [3, 2, 1]