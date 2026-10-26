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
- Returns `(makespan, machine)`, where `machine[i]` is the list of job times given to machine i

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