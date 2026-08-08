"""Render a database's schema as CREATE TABLE DDL for the prompt.

Pulls the stored DDL straight from sqlite_master (what the models in this field
are trained to consume), with a small per-table sample-row option off by default.
Cached per db_path since many examples share a database.
"""
from __future__ import annotations

import functools
import sqlite3


@functools.lru_cache(maxsize=256)
def schema_ddl(db_path: str, sample_rows: int = 0) -> str:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.text_factory = lambda b: b.decode("utf-8", "replace")
    parts: list[str] = []
    tables = con.execute(
        "SELECT name, sql FROM sqlite_master "
        "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    for name, sql in tables:
        if not sql:
            continue
        parts.append(sql.strip() + ";")
        if sample_rows > 0:
            try:
                rows = con.execute(f'SELECT * FROM "{name}" LIMIT {sample_rows}').fetchall()
                cols = [d[0] for d in con.execute(f'SELECT * FROM "{name}" LIMIT 0').description]
                if rows:
                    parts.append(f"/* {sample_rows} example rows from {name}:")
                    parts.append("   " + " | ".join(cols))
                    for r in rows:
                        parts.append("   " + " | ".join(str(c) for c in r))
                    parts.append("*/")
            except Exception:  # noqa: BLE001
                pass
    con.close()
    return "\n".join(parts)
