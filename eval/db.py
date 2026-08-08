"""SQLite execution with a hard timeout — thread-safe.

The eval scores many examples in a ThreadPool. We must NOT fork subprocesses from
worker threads (fork in a multithreaded process inherits locked mutexes → the
child can deadlock). Instead we run the query on a fresh connection in the
calling thread and arm a `threading.Timer` that calls `connection.interrupt()`
on timeout — `interrupt()` is explicitly thread-safe and aborts a running query
with OperationalError. No subprocesses, no fork.
"""
from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass


@dataclass
class ExecResult:
    ok: bool                 # executed without error AND within timeout
    rows: list | None        # result rows (None on error)
    error: str | None        # error / "timeout" string when ok is False


def execute_sql(db_path: str, sql: str, timeout: float = 30.0) -> ExecResult:
    sql = (sql or "").strip().rstrip(";").strip()
    if not sql:
        return ExecResult(False, None, "empty SQL")

    # check_same_thread=False so the Timer thread may call interrupt()
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
    con.text_factory = lambda b: b.decode("utf-8", "replace")
    timed_out = {"v": False}

    def _kill():
        timed_out["v"] = True
        try:
            con.interrupt()
        except Exception:  # noqa: BLE001
            pass

    timer = threading.Timer(timeout, _kill)
    timer.start()
    try:
        rows = con.execute(sql).fetchall()
        return ExecResult(True, rows, None)
    except Exception as e:  # noqa: BLE001 - any error means invalid SQL (or interrupt)
        if timed_out["v"]:
            return ExecResult(False, None, "timeout")
        return ExecResult(False, None, f"{type(e).__name__}: {e}")
    finally:
        timer.cancel()
        con.close()
