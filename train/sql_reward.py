"""Reward functions for GRPO (the core of Arm 3).

Reuses the validated eval harness: extract SQL from the completion (same extractor
as eval), execute against the example's SQLite DB, compare result rows to gold with
the official-BIRD-faithful set(tuple(row)) semantics.

Two rewards live here:

`execution_reward` — the ORIGINAL simple reward. KEPT FOR REFERENCE ONLY.
    +1.0 exact result match / +0.1 runs-but-wrong / 0.0 invalid.
    The +0.1 "executes" floor is HACKABLE: a trivial always-running query farms it,
    the policy collapses to a 22-token degenerate query, every rollout gets 0.1,
    reward_std->0, gradient->0 (observed run #1, see PROGRESS.md). DO NOT USE.

`composite_reward` — the SOLID reward (default). Partial credit comes from signals
that CORRELATE with correctness and can't be trivially maxed (literature: Reasoning-
SQL 2503.23157 reports reward hacking from naive relaxed-match partials):

    +1.0                         exact result-set match (DOMINATES — the real goal)
    else, sum of CAPPED partials (total <= MAX_PARTIAL, well below 1.0):
      + W_OVERLAP * jaccard(pred_rows, gold_rows)   result-set overlap; an empty/
                                                    trivial query scores ~0 -> no farm
      + W_SCHEMA  * schema_link(pred, gold)         frac of GOLD identifiers used;
                                                    can't max without ~the right query
      + W_SYNTAX                                    tiny, only if pred executes at all
      - LEN_PENALTY                                 if pred is degenerately short
    0.0                          no SQL extracted

Pair with grpo.py: small KL anchor (beta~0.02-0.04) + temperature>=1.0 + num_gen>=4.

TRL passes dataset columns (db_path, gold_sql) through **kwargs aligned to the
completions batch.
"""
from __future__ import annotations

import os
import re
import sys

_EVAL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval")
if _EVAL not in sys.path:
    sys.path.insert(0, _EVAL)

import prompts as pt           # noqa: E402
from db import execute_sql      # noqa: E402
from compare import compare     # noqa: E402

CORRECT = 1.0
RUNS = 0.1                      # legacy floor (execution_reward only)
TIMEOUT = 30.0

# composite_reward weights (partials are CAPPED so CORRECT always dominates)
W_OVERLAP = 0.20               # result-set Jaccard
W_SCHEMA = 0.10               # gold-identifier coverage
W_SYNTAX = 0.05               # executes-but-wrong (tiny; can't live on it)
MAX_PARTIAL = 0.30            # hard cap on total partial credit (< CORRECT)
MIN_TOKENS = 5               # below this, pred is degenerate
LEN_PENALTY = 0.10           # subtracted from partials for degenerate-short preds

# SQL keywords excluded from "identifier" extraction for schema-linking
_SQL_KW = {
    "select", "from", "where", "group", "by", "order", "having", "limit", "offset",
    "join", "inner", "left", "right", "outer", "full", "on", "as", "and", "or",
    "not", "in", "is", "null", "like", "between", "exists", "count", "sum", "avg",
    "min", "max", "distinct", "case", "when", "then", "else", "end", "asc", "desc",
    "union", "all", "with", "cast", "round", "abs", "coalesce", "substr", "length",
    "upper", "lower", "strftime", "date", "time", "datetime", "true", "false",
}
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _completion_text(c):
    return c[0]["content"] if isinstance(c, list) else c


def _identifiers(sql: str) -> set[str]:
    """Lowercased identifier-like tokens (table/column names), minus SQL keywords."""
    return {t.lower() for t in _IDENT.findall(sql or "")} - _SQL_KW


def _schema_link(pred_sql: str, gold_sql: str) -> float:
    """Fraction of GOLD identifiers that appear in the prediction (0..1)."""
    gold_ids = _identifiers(gold_sql)
    if not gold_ids:
        return 0.0
    pred_ids = _identifiers(pred_sql)
    return len(gold_ids & pred_ids) / len(gold_ids)


def _row_jaccard(pred_rows, gold_rows) -> float:
    """Set Jaccard over result rows (native tuples). Empty pred -> 0."""
    if not pred_rows or not gold_rows:
        return 0.0
    ps = {tuple(r) for r in pred_rows}
    gs = {tuple(r) for r in gold_rows}
    inter = len(ps & gs)
    union = len(ps | gs)
    return inter / union if union else 0.0


def execution_reward(completions, db_path, gold_sql, **kwargs):
    """LEGACY simple reward (hackable 0.1 floor). Kept for reference; do not use."""
    rewards = []
    for comp, dbp, gold in zip(completions, db_path, gold_sql):
        sql = pt.extract(_completion_text(comp))
        if not sql:
            rewards.append(0.0)
            continue
        g = execute_sql(dbp, gold, TIMEOUT)
        p = execute_sql(dbp, sql, TIMEOUT)
        if compare(p, g, gold):
            rewards.append(CORRECT)
        elif p.ok:
            rewards.append(RUNS)
        else:
            rewards.append(0.0)
    return rewards


def composite_reward(completions, db_path, gold_sql, **kwargs):
    """SOLID anti-hacking reward (default). See module docstring."""
    rewards = []
    for comp, dbp, gold in zip(completions, db_path, gold_sql):
        sql = pt.extract(_completion_text(comp))
        if not sql:
            rewards.append(0.0)
            continue
        g = execute_sql(dbp, gold, TIMEOUT)
        p = execute_sql(dbp, sql, TIMEOUT)
        if compare(p, g, gold):
            rewards.append(CORRECT)          # exact match dominates
            continue
        # --- partial credit (capped, correlated-with-correctness signals only) ---
        partial = W_SCHEMA * _schema_link(sql, gold)
        if p.ok:
            partial += W_SYNTAX
            if g.ok:
                partial += W_OVERLAP * _row_jaccard(p.rows, g.rows)
        # degenerate-short penalty (kills the 22-token collapse incentive)
        if len(sql.split()) < MIN_TOKENS:
            partial -= LEN_PENALTY
        rewards.append(max(0.0, min(MAX_PARTIAL, partial)))
    return rewards
