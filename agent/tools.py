"""SQLite schema-discovery + execution tools for the Arm 4 agentic loop.

Three read-only tools, exposed to the Claude Agent SDK as an in-process MCP
server (no subprocess). The DB is per-question, so we build a fresh server bound
to one db_path per call (`build_server`). All tools open the DB read-only
(`mode=ro`) — the agent can explore and validate but never mutate.

Error policy (critical): a handler NEVER raises. A raised exception ends the
whole `query()` call; instead we catch and return `is_error=True` so the model
sees the failure as data and self-corrects (the whole point of the loop). This
mirrors the docs' guidance on litellm-path error handling.
"""
from __future__ import annotations

import sqlite3
from typing import Any

from claude_agent_sdk import tool, create_sdk_mcp_server, ToolAnnotations

# how many rows a tool echoes back into the context (keep prompts bounded)
_SAMPLE_ROWS = 3
_MAX_RESULT_ROWS = 20


def _connect(db_path: str) -> sqlite3.Connection:
    """Read-only connection (mirrors eval/db.py). immutable would also work but
    mode=ro matches the scorer's connection semantics on a writable FS."""
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)


def build_server(db_path: str) -> tuple[Any, dict]:
    """Build an in-process MCP server bound to `db_path`.

    Returns (server, state). `state["submitted"]` is populated the moment the
    agent calls submit_query — that string IS the final answer, captured straight
    from the tool-call argument. No text parsing anywhere.
    """
    state: dict = {"submitted": None}

    @tool("list_tables", "List all table names in the database.", {},
          annotations=ToolAnnotations(readOnlyHint=True))
    async def list_tables(args: dict[str, Any]) -> dict[str, Any]:
        try:
            with _connect(db_path) as c:
                rows = c.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
                    "ORDER BY name").fetchall()
            names = ", ".join(r[0] for r in rows) or "(no tables)"
            return {"content": [{"type": "text", "text": names}]}
        except Exception as e:
            return {"content": [{"type": "text", "text": f"error: {e}"}],
                    "is_error": True}

    @tool("describe_table",
          "Show the CREATE statement (columns, types, keys) and a few sample "
          "rows for one table. Call this before writing SQL against a table.",
          {"table": str}, annotations=ToolAnnotations(readOnlyHint=True))
    async def describe_table(args: dict[str, Any]) -> dict[str, Any]:
        table = args["table"]
        try:
            with _connect(db_path) as c:
                ddl = c.execute(
                    "SELECT sql FROM sqlite_master WHERE name=?",
                    (table,)).fetchone()
                if not ddl:
                    return {"content": [{"type": "text",
                            "text": f"no such table: {table}"}],
                            "is_error": True}
                sample = c.execute(
                    f'SELECT * FROM "{table}" LIMIT {_SAMPLE_ROWS}').fetchall()
            text = f"{ddl[0]}\n\n-- {_SAMPLE_ROWS} sample rows:\n{sample}"
            return {"content": [{"type": "text", "text": text}]}
        except Exception as e:
            return {"content": [{"type": "text", "text": f"error: {e}"}],
                    "is_error": True}

    @tool("execute_sql",
          "Run a read-only SQL query against the database and return up to "
          f"{_MAX_RESULT_ROWS} result rows. Use this to validate a query before "
          "giving your final answer; on an error, read it and fix the query.",
          {"sql": str})
    async def execute_sql(args: dict[str, Any]) -> dict[str, Any]:
        sql = args["sql"]
        try:
            with _connect(db_path) as c:
                rows = c.execute(sql).fetchall()
            shown = rows[:_MAX_RESULT_ROWS]
            more = "" if len(rows) <= _MAX_RESULT_ROWS \
                else f"\n... ({len(rows)} rows total)"
            return {"content": [{"type": "text",
                    "text": f"{len(rows)} rows: {shown}{more}"}]}
        except Exception as e:
            # feed the SQL error straight back so the model can repair
            return {"content": [{"type": "text", "text": f"SQL error: {e}"}],
                    "is_error": True}

    @tool("submit_query",
          "Submit your FINAL SQL query as the answer to the question. Call this "
          "exactly once, after you have validated the query with execute_sql. "
          "The query you pass here is recorded as your answer; do not call any "
          "more tools afterward.",
          {"sql": str})
    async def submit_query(args: dict[str, Any]) -> dict[str, Any]:
        state["submitted"] = args["sql"]
        return {"content": [{"type": "text",
                "text": "Final query recorded. You are done."}]}

    server = create_sdk_mcp_server(
        name="sqlite", version="1.0.0",
        tools=[list_tables, describe_table, execute_sql, submit_query])
    return server, state


# tool names as Claude sees them (mcp__<server>__<tool>)
ALLOWED_TOOLS = [
    "mcp__sqlite__list_tables",
    "mcp__sqlite__describe_table",
    "mcp__sqlite__execute_sql",
    "mcp__sqlite__submit_query",
]
