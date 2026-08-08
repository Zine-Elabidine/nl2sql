"""Prompt templates, one per model family.

Each template takes an Example + rendered schema DDL and returns a list of chat
messages ([{role, content}, ...]). Kept pluggable because Arctic, SLM-SQL and a
plain Qwen-Coder expect slightly different framing; EX is sensitive to this so we
keep them explicit rather than one-size-fits-all.

A template also exposes `extract(text)` to pull the final SQL out of the model's
(possibly reasoning-wrapped) output.
"""
from __future__ import annotations

import re


def _strip_sql(text: str) -> str:
    """Best-effort extraction of the final SQL statement from model output."""
    if not text:
        return ""
    # 1) fenced ```sql ... ``` block (take the last one)
    fences = re.findall(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fences:
        return fences[-1].strip()
    # 2) reasoning models: take whatever follows the last </think>
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    # 3) from the last SELECT/WITH onward
    m = list(re.finditer(r"\b(WITH|SELECT)\b", text, re.IGNORECASE))
    if m:
        return text[m[-1].start():].strip().rstrip("`").strip()
    return text.strip()


_GENERIC_SYS = (
    "You are an expert data analyst. Given a SQLite database schema and a "
    "question, write a single valid SQLite query that answers it. Return only "
    "the SQL inside a ```sql code block."
)


def _user_block(ex, ddl: str) -> str:
    parts = [f"Database schema (SQLite):\n{ddl}", f"\nQuestion: {ex.question}"]
    if ex.evidence:
        parts.append(f"\nExternal knowledge: {ex.evidence}")
    parts.append("\nWrite the SQLite query.")
    return "\n".join(parts)


def generic(ex, ddl: str):
    return [
        {"role": "system", "content": _GENERIC_SYS},
        {"role": "user", "content": _user_block(ex, ddl)},
    ]


def arctic(ex, ddl: str):
    # Arctic-Text2SQL-R1 is trained to reason then emit SQL; generic framing works,
    # extraction handles the reasoning prefix.
    return generic(ex, ddl)


_OMNISQL_TMPL = """Task Overview:
You are a data science expert. Below, you are provided with a database schema and a natural language question. Your task is to understand the schema and generate a valid SQL query to answer the question.

Database Engine:
SQLite

Database Schema:
{db_details}
This schema describes the database's structure, including tables, columns, primary keys, foreign keys, and any relevant relationships or constraints.

Question:
{question}

Instructions:
- Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
- The generated query should return all of the information asked in the question without any missing or extra information.
- Before generating the final SQL query, please think through the steps of how to write the query.

Output Format:
In your answer, please enclose the generated SQL query in a code block:
```sql
-- Your SQL query
```

Take a deep breath and think step by step to find the correct SQL query."""


def omnisql(ex, ddl: str):
    """OmniSQL / SynSQL prompt — what SLM-SQL was trained on.

    Single user message; external knowledge (BIRD evidence) is folded into the
    question; schema (ddl) should include sample rows (run with --sample-rows>0).
    """
    question = ex.question
    if ex.evidence:
        question = f"{ex.question}\n\nExternal knowledge: {ex.evidence}"
    content = _OMNISQL_TMPL.format(db_details=ddl, question=question)
    return [{"role": "user", "content": content}]


def slm_sql(ex, ddl: str):
    # SLM-SQL was trained on SynSQL-2.5M → use the OmniSQL prompt.
    return omnisql(ex, ddl)


TEMPLATES = {"generic": generic, "arctic": arctic, "slm_sql": slm_sql,
             "omnisql": omnisql}


def extract(text: str) -> str:
    return _strip_sql(text)
