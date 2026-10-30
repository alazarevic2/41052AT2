"""Ready-signalled process helpers for the exact-DP experiments."""

import gc
import multiprocessing
import time
import tracemalloc

from algorithms.dp import solve as dp_solve


def _dp_worker(connection, jobs, machines, memory, collect_stats, early_stop):
    try:
        gc.collect()
        connection.send({"status": "ready"})
        if memory:
            tracemalloc.start()
        stats = {} if collect_stats else None
        start = time.perf_counter()
        makespan, _ = dp_solve(jobs, machines, stats=stats,
                               early_stop=early_stop)
        seconds = time.perf_counter() - start
        peak = tracemalloc.get_traced_memory()[1] if memory else None
        connection.send({"status": "ok", "makespan": makespan,
                         "seconds": seconds, "peak_python_bytes": peak,
                         **(stats or {})})
    except Exception as error:
        connection.send({"status": "error",
                         "error": f"{type(error).__name__}: {error}"})
    finally:
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        connection.close()


def fresh_process_trial(worker, args, timeout, startup_timeout=30):
    """Run a ready-signalling worker with startup excluded from `timeout`."""
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=worker, args=(sender, *args))
    started = False
    try:
        process.start()
        started = True
        sender.close()
        if not receiver.poll(startup_timeout):
            return {"status": "startup_timeout",
                    "error": "worker did not become ready before startup timeout"}
        ready = receiver.recv()
        if ready.get("status") != "ready":
            return ready
        if not receiver.poll(timeout):
            return {"status": "DNF", "error": "no result before solve timeout"}
        return receiver.recv()
    except EOFError:
        return {"status": "error", "error": "worker exited without a result"}
    except Exception as error:
        return {"status": "error", "error": f"{type(error).__name__}: {error}"}
    finally:
        if started:
            if process.is_alive():
                process.terminate()
            process.join()
            process.close()
        receiver.close()
        sender.close()


def dp_trial(jobs, machines, timeout, *, memory=False, collect_stats=False,
             early_stop=True):
    """Run one production-DP call in a fresh process."""
    return fresh_process_trial(
        _dp_worker,
        (jobs, machines, memory, collect_stats, early_stop),
        timeout,
    )
