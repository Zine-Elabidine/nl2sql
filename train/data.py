"""BIRD-train data loading + prompt formatting for Arm-3 training.

Reuses the eval/ primitives (Example shape, prompt templates, schema DDL) so the
training prompts match exactly what we grade on at eval time. Builds:
  - SFT rows:  {"text": <full chat incl. gold SQL completion>}
  - GRPO rows: {"prompt": <chat messages>, "db_path", "gold_sql"} (reward needs the
               last two as columns -> they arrive in the reward func via **kwargs)
"""
from __future__ import annotations

import glob
import json
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass

# make eval/ importable
_EVAL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval")
if _EVAL not in sys.path:
    sys.path.insert(0, _EVAL)

import prompts as pt          # noqa: E402
from schema import schema_ddl  # noqa: E402


@dataclass
class TrainExample:
    db_id: str
    question: str
    gold_sql: str
    db_path: str
    evidence: str = ""


def _find(root, *names):
    for n in names:
        hits = glob.glob(os.path.join(root, "**", n), recursive=True)
        if hits:
            return sorted(hits, key=len)[0]
    raise FileNotFoundError(f"none of {names} under {root}")


def _db_dir(root):
    for c in ["train_databases", "database", "databases"]:
        for h in glob.glob(os.path.join(root, "**", c), recursive=True):
            if os.path.isdir(h) and glob.glob(os.path.join(h, "*", "*.sqlite")):
                return h
    raise FileNotFoundError(f"no train sqlite db dir under {root}")


def load_bird_train(root: str) -> list[TrainExample]:
    train_json = _find(root, "train.json")
    dbdir = _db_dir(root)
    with open(train_json) as f:
        data = json.load(f)
    out = []
    for r in data:
        db_id = r["db_id"]
        sql = r.get("SQL") or r.get("query", "")
        dbp = os.path.join(dbdir, db_id, f"{db_id}.sqlite")
        if not sql or not os.path.exists(dbp):
            continue
        out.append(TrainExample(db_id, r["question"], sql, dbp,
                                r.get("evidence", "") or ""))
    return out


def stratified_subset(exs: list[TrainExample], n: int, seed: int = 3407) -> list[TrainExample]:
    """Pick ~n examples spread across db_id (keeps schema diversity for the
    execution reward). Proportional per-DB, round-robin to fill the remainder."""
    if n <= 0 or n >= len(exs):
        return exs
    rng = random.Random(seed)
    by_db: dict[str, list[TrainExample]] = defaultdict(list)
    for e in exs:
        by_db[e.db_id].append(e)
    for v in by_db.values():
        rng.shuffle(v)
    dbs = sorted(by_db)
    picked: list[TrainExample] = []
    # round-robin one-at-a-time across DBs until we hit n
    idx = {db: 0 for db in dbs}
    while len(picked) < n:
        progressed = False
        for db in dbs:
            if idx[db] < len(by_db[db]):
                picked.append(by_db[db][idx[db]])
                idx[db] += 1
                progressed = True
                if len(picked) >= n:
                    break
        if not progressed:
            break
    rng.shuffle(picked)
    return picked


# --- prompt building (reuse eval template so train==eval framing) ---
def _ex_for_template(te: TrainExample):
    # eval prompt templates read .question/.evidence; schema_ddl reads db_path
    class _E:  # lightweight shim with the attrs the templates touch
        question = te.question
        evidence = te.evidence
    return _E()


def build_prompt(te: TrainExample, template: str = "generic", sample_rows: int = 0):
    ddl = schema_ddl(te.db_path, sample_rows)
    return pt.TEMPLATES[template](_ex_for_template(te), ddl)


def sft_row(te: TrainExample, tok, template: str = "generic", sample_rows: int = 0):
    msgs = build_prompt(te, template, sample_rows)
    completion = f"```sql\n{te.gold_sql.strip()}\n```"
    msgs = msgs + [{"role": "assistant", "content": completion}]
    return {"text": tok.apply_chat_template(msgs, tokenize=False)}


def grpo_row(te: TrainExample, template: str = "generic", sample_rows: int = 0):
    return {
        "prompt": build_prompt(te, template, sample_rows),
        "db_path": te.db_path,
        "gold_sql": te.gold_sql,
    }
