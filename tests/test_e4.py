"""E4 measurement and output checks; no timing thresholds."""

import csv
import json
import time

from algorithms.dp import solve
from benchmarks import run_e4


def stalled_worker(connection, *args):
    connection.send({"status": "ready"})
    time.sleep(5)


def test_dp_stats_preserve_schedule_and_fresh_cache():
    first, second = {}, {}
    jobs = [3, 3, 3]
    expected = solve(jobs, 2)
    assert solve(jobs, 2, stats=first) == expected
    assert solve(jobs, 2, stats=second) == expected
    assert first == second
    assert first["states"] > 0
    assert first["calls"] == first["states"] + first["cache_hits"]


def test_dp_trial_reports_counts_and_memory():
    timing = run_e4.dp_trial([2, 3, 4, 5], 2)
    memory = run_e4.dp_trial([2, 3, 4, 5], 2, memory=True)
    assert timing["status"] == memory["status"] == "ok"
    assert timing["makespan"] == memory["makespan"] == 7
    assert timing["states"] > 0
    assert timing["peak_python_bytes"] is None
    assert memory["peak_python_bytes"] > 0


def test_dp_timeout_is_recorded(monkeypatch):
    monkeypatch.setattr(run_e4, "_dp_worker", stalled_worker)
    assert run_e4.dp_trial([3], 1, timeout=0.05)["status"] == "DNF"


def test_lpt_uses_batches_and_separate_memory(monkeypatch):
    monkeypatch.setattr(run_e4, "M", 2)
    monkeypatch.setattr(run_e4, "LPT_REPEATS", 3)
    monkeypatch.setattr(run_e4, "BATCH_SECONDS", 0.001)
    result = run_e4.measure_lpt([2, 3, 4, 5])
    assert result["makespan"] == 7
    assert result["batch_calls"] > 1
    assert len(json.loads(result["samples_seconds"])) == 3
    assert result["min_seconds"] <= result["median_seconds"]
    assert result["iqr_seconds"] >= 0
    assert result["peak_python_bytes"] > 0


def test_csv_preserves_matched_inputs_and_dnf(tmp_path, monkeypatch):
    output = tmp_path / "e4.csv"
    monkeypatch.setattr(run_e4, "OUTPUT", str(output))
    monkeypatch.setattr(run_e4, "N", 6)
    monkeypatch.setattr(run_e4, "M", 2)
    monkeypatch.setattr(run_e4, "SEEDS", [1, 2])
    monkeypatch.setattr(run_e4, "FAMILIES", {
        name: run_e4.FAMILIES[name] for name in ("uniform", "balanced")})
    received = {"DP": [], "LPT": []}

    def dp(jobs):
        received["DP"].append(jobs.copy())
        return {"status": "DNF", "makespan": None}

    def lpt(jobs):
        received["LPT"].append(jobs.copy())
        return {"status": "ok", "makespan": 100}

    monkeypatch.setattr(run_e4, "measure_dp", dp)
    monkeypatch.setattr(run_e4, "measure_lpt", lpt)
    run_e4.main()
    assert received["DP"] == received["LPT"]
    rows = list(csv.DictReader(output.open()))
    assert len(rows) == 8
    for row in rows:
        assert len(json.loads(row["jobs"])) == 6
        if row["family"] == "uniform":
            assert row["optimum"] == row["relative_error_percent"] == ""
        else:
            assert row["optimum"] == "100"
    metadata = json.loads(output.with_suffix(".json").read_text())
    assert metadata["parameters"]["SEEDS"] == [1, 2]
    assert metadata["python"]
