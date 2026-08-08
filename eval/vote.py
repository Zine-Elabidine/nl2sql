"""Execution-based self-consistency: turn N sampled SQL candidates into one pick.

Standard test-time scaling for text-to-SQL: sample N queries, EXECUTE all of
them, cluster candidates by their result set, and keep a representative of the
largest cluster (majority vote on *answers*, not on query text — two different
queries that return the same rows agree). This converts the base model's pass@k
strength into a single committed answer.

Clustering is over VALID executions only: a candidate that errors or times out
casts no vote (a degenerate always-failing query can't win). If nothing runs, we
fall back to the first candidate so the example still gets scored (as wrong).
"""
from __future__ import annotations

from db import execute_sql


def _key(rows):
    """Hashable, order-insensitive key for a result set (mirrors compare._normalize)."""
    try:
        return frozenset(tuple(r) for r in rows)
    except TypeError:
        # extremely rare unhashable cell type — fall back to a stable repr
        return tuple(sorted(repr(r) for r in rows))


def select_by_execution(db_path: str, candidates: list[str], timeout: float = 30.0):
    """Execute candidates, cluster by result set, return (chosen_sql, meta).

    Largest cluster of agreeing VALID results wins; ties break to the earliest
    candidate (deterministic). meta carries diagnostics for the JSONL.
    """
    clusters: dict = {}  # key -> [first_idx, count, sql]
    for i, sql in enumerate(candidates):
        res = execute_sql(db_path, sql, timeout)
        if not res.ok:
            continue
        k = _key(res.rows)
        if k not in clusters:
            clusters[k] = [i, 0, sql]
        clusters[k][1] += 1

    n_valid = sum(v[1] for v in clusters.values())
    if not clusters:
        return candidates[0], {
            "n_candidates": len(candidates), "n_valid": 0,
            "cluster_size": 0, "n_clusters": 0, "agreement": 0.0,
        }
    # max count, tie-break earliest occurrence
    best = min(clusters.values(), key=lambda v: (-v[1], v[0]))
    meta = {
        "n_candidates": len(candidates), "n_valid": n_valid,
        "cluster_size": best[1], "n_clusters": len(clusters),
        "agreement": round(best[1] / len(candidates), 3),
    }
    return best[2], meta
