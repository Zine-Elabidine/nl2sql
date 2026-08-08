"""Uniform loaders for BIRD dev and Spider dev.

Both are normalized to a list of `Example`. Folder layout is auto-discovered
because the official zips nest things slightly differently across releases.
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass


@dataclass
class Example:
    id: str
    db_id: str
    question: str
    gold_sql: str
    db_path: str          # absolute path to the .sqlite file
    evidence: str = ""    # BIRD external knowledge; "" for Spider
    difficulty: str = ""  # BIRD: simple/moderate/challenging


def _find(root: str, *names: str) -> str:
    """Return the first existing path matching any of `names` (glob) under root."""
    for n in names:
        hits = glob.glob(os.path.join(root, "**", n), recursive=True)
        if hits:
            return sorted(hits, key=len)[0]
    raise FileNotFoundError(f"none of {names} found under {root}")


def _db_dir(root: str, kind: str) -> str:
    """Locate the directory holding per-db subfolders with .sqlite files."""
    candidates = ["dev_databases", "database", "databases"]
    for c in candidates:
        hits = glob.glob(os.path.join(root, "**", c), recursive=True)
        for h in hits:
            if os.path.isdir(h) and glob.glob(os.path.join(h, "*", "*.sqlite")):
                return h
    raise FileNotFoundError(f"could not locate {kind} sqlite db dir under {root}")


def load_bird(root: str) -> list[Example]:
    dev_json = _find(root, "dev.json", "mini_dev_sqlite.json")
    dbdir = _db_dir(root, "bird")
    with open(dev_json) as f:
        data = json.load(f)
    out = []
    for i, r in enumerate(data):
        db_id = r["db_id"]
        out.append(Example(
            id=str(r.get("question_id", i)),
            db_id=db_id,
            question=r["question"],
            gold_sql=r.get("SQL") or r.get("query", ""),
            db_path=os.path.join(dbdir, db_id, f"{db_id}.sqlite"),
            evidence=r.get("evidence", "") or "",
            difficulty=r.get("difficulty", "") or "",
        ))
    return out


def load_spider(root: str) -> list[Example]:
    dev_json = _find(root, "dev.json")
    dbdir = _db_dir(root, "spider")
    with open(dev_json) as f:
        data = json.load(f)
    out = []
    for i, r in enumerate(data):
        db_id = r["db_id"]
        out.append(Example(
            id=str(i),
            db_id=db_id,
            question=r["question"],
            gold_sql=r.get("query") or r.get("SQL", ""),
            db_path=os.path.join(dbdir, db_id, f"{db_id}.sqlite"),
        ))
    return out


LOADERS = {"bird": load_bird, "spider": load_spider}
