# 41052AT2

41052 - Advanced Algorithms Programming Assignment 2

## Makespan Scheduling: Exact DP vs LPT Approximation

**Problem:** Given n jobs with processing times and m identical machines, assign each job to a machine to minimise the makespan (the maximum machine load).

## Algorithms

| Algorithm | Type | File | Primary author |
|---|---|---|---|
| Dynamic Programming | Exact | `algorithms/dp.py` | Thomas Page |
| LPT (Longest Processing Time first) | 4/3-approximation | `algorithms/lpt.py` | Anna Lazarevic |

## Project Structure
```text
41052AT2/
├── README.md
├── requirements.txt
├── .gitignore
├── algorithms/          # One module per algorithm
│   ├── __init__.py
│   ├── dp.py            # Exact DP (Thomas Page)
│   └── lpt.py           # LPT approximation (Anna Lazarevic)
├── tests/               # Unit tests for each algorithm
│   ├── test_dp.py
│   └── test_lpt.py
└── benchmarks/          # Shared benchmarking framework (input generators, runner, plots) 
```

## Shared Interface

Every algorithm module must provide:

```python
solve(jobs: list[int], m: int) -> tuple[int, list[list[int]]]
```

- `jobs`: a list of non-negative integer processing times
- `m`: the number of machines (must be at least 1)
- Returns `(makespan, machines)`, where `machines[i]` is the list of job times given to machine i

This lets the benchmarking framework run every algorithm on the same inputs.

## Setup
```text
pip install -r requirements.txt
```

## Running Tests

From the repository root:
```text
python -m pytest
```

## Shared Performance Experiments

Run both algorithms on identical generated inputs and show the result table:

```text
pytest tests/test_performance.py -s -q
```

The table reports median runtime over five fresh calls, both makespans, and
the ratio LPT / OPT (1.0 means LPT is optimal). Random inputs use seed 41052.
Cases cover six input families with 6, 10, and 12 jobs on 2 and 3 machines,
plus the known LPT worst-case family on 2 to 4 machines.
The separate DP known-optimum generator is also tested on 2 to 4 machines
(it produces the same mathematical input family). Scaling experiments multiply
uniform job times by 10 and 100. Ordering experiments use original, ascending,
descending, and shuffled uniform inputs. Both use the same job counts, machine
counts, and seed as the family comparisons. Checks verify that scaling multiplies
both makespans by the factor and reordering preserves both makespans.
Times include schedule reconstruction but exclude input preparation and result
checks. Garbage collection runs before each measurement. Timings vary between
machines and runs; the tests validate results without imposing speed thresholds.
