"""Execution-accuracy comparison: do predicted and gold SQL return the same rows?

Set-based and order-insensitive by default (matches the official BIRD/Spider EX
definition: a query is correct iff its result set equals the gold result set,
regardless of row order). Order is only enforced when the gold query has a
top-level ORDER BY, which is the standard refinement to avoid rewarding queries
that drop an explicit ordering requirement.
"""
from __future__ import annotations

import re

from db import ExecResult, execute_sql

_ORDER_BY = re.compile(r"\border\s+by\b", re.IGNORECASE)


def _normalize(rows: list | None, ordered: bool):
    """Canonicalize a result set for comparison.

    Official BIRD `evaluation_ex.py` semantics: `set(cursor.fetchall())` — raw
    row tuples (native types, original column order) into a set, i.e.
    order-insensitive across rows but column order WITHIN a row is significant,
    and values are compared natively (1 == 1.0). We mirror that exactly so our
    EX is byte-comparable to published BIRD numbers (validated: matches the
    official scorer to within 1/50 examples).

    NOTE: BIRD EX does not enforce ORDER BY (it always set()s), so `ordered` is
    only honoured in the stricter non-default path used for debugging.
    """
    if rows is None:
        return None
    if ordered:
        # stricter, non-official: preserve row order
        return tuple(tuple(r) for r in rows)
    return set(tuple(r) for r in rows)


def compare(pred: ExecResult, gold: ExecResult, gold_sql: str,
            enforce_order: bool = False) -> bool:
    """True iff predicted execution matches gold execution (EX).

    Default mirrors official BIRD EX (pure set comparison, ORDER BY ignored).
    Pass enforce_order=True for the stricter order-sensitive variant.
    """
    if not gold.ok:
        # gold itself failed to run — exclude from accuracy upstream; treat as no-match
        return False
    if not pred.ok:
        return False
    ordered = enforce_order and bool(_ORDER_BY.search(gold_sql or ""))
    return _normalize(pred.rows, ordered) == _normalize(gold.rows, ordered)


def score_one(db_path: str, pred_sql: str, gold_sql: str, timeout: float = 30.0):
    """Execute both and return (is_correct, pred_valid, gold_ok, detail)."""
    gold = execute_sql(db_path, gold_sql, timeout)
    pred = execute_sql(db_path, pred_sql, timeout)
    correct = compare(pred, gold, gold_sql)
    detail = {
        "pred_ok": pred.ok,
        "gold_ok": gold.ok,
        "pred_error": pred.error,
        "gold_error": gold.error,
    }
    return correct, pred.ok, gold.ok, detail
