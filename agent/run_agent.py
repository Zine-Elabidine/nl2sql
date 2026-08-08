"""Arm 4 — agentic NL→SQL over BIRD dev, scored on the same EX harness.

Runs a small Claude Agent SDK loop per question: the model discovers the schema
with list_tables/describe_table, drafts SQL, validates it with execute_sql,
repairs on error, then emits final SQL in a ```sql block. We extract that with
eval/prompts.extract and score it with eval/compare.score_one — so the EX number
is directly comparable to Arms 1-3 (single-pass) run through run_eval.py.

The model is served locally: vLLM (OpenAI) → litellm proxy (Anthropic front) →
this SDK. See WIRING.md / litellm_config.yaml. No Anthropic API is used.

Usage:
  # smoke (verify the model actually calls tools, not one-shots):
  python agent/run_agent.py --root data/dev --limit 5 --verbose

  # slice:
  python agent/run_agent.py --root data/dev --limit 32 --concurrency 4

  # full:
  python agent/run_agent.py --root data/dev --tag arm4_coder7b
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

# import the existing harness (eval/ is a sibling of agent/)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))
import datasets as ds          # noqa: E402
from compare import score_one   # noqa: E402

from claude_agent_sdk import (  # noqa: E402
    query, ClaudeAgentOptions,
    AssistantMessage, ToolUseBlock, ResultMessage,
)
from tools import build_server, ALLOWED_TOOLS  # noqa: E402


SYSTEM = (
    "You are an expert text-to-SQL agent working over a single SQLite database. "
    "You MUST ground every query in the real schema — never guess table or "
    "column names.\n\n"
    "Procedure:\n"
    "1. Call list_tables to see what exists.\n"
    "2. Call describe_table on each table you might need (read columns, types, "
    "keys, and the sample rows).\n"
    "3. Draft one SQLite query that answers the question.\n"
    "4. Call execute_sql to run it. If it errors or the rows look wrong, read "
    "the message and fix the query, then run it again.\n"
    "5. When the query runs and the result answers the question, call "
    "submit_query with that SQL as your final answer. Call submit_query exactly "
    "once and then stop.\n\n"
    "Rules: SQLite dialect only; return exactly the columns the question asks "
    "for, no more; prefer the simplest correct query. You MUST finish by calling "
    "submit_query — a question is only answered once you have submitted."
)


def _build_prompt(ex) -> str:
    q = f"Question: {ex.question}"
    if ex.evidence:
        q += f"\nExternal knowledge: {ex.evidence}"
    return q


async def solve(ex, *, model: str, base_url: str, max_turns: int,
                verbose: bool) -> tuple[str, dict]:
    """Run the agentic loop for one example. Returns (final_sql, meta).

    The final SQL is captured directly from the submit_query tool-call argument
    (state["submitted"]) — no text parsing. If the agent never submits, the
    answer is empty and scores wrong.
    """
    server, state = build_server(ex.db_path)
    options = ClaudeAgentOptions(
        system_prompt=SYSTEM,
        model=model,
        mcp_servers={"sqlite": server},
        allowed_tools=ALLOWED_TOOLS,
        tools=[],                      # no built-ins; only our 3 tools exist
        permission_mode="default",
        max_turns=max_turns,
        setting_sources=[],            # ignore any local CLI settings
        env={
            "ANTHROPIC_BASE_URL": base_url,
            "ANTHROPIC_API_KEY": "sk-local",   # opaque; vLLM needs no auth
            "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
            # The CLI otherwise requests max_tokens=32000, which exceeds our
            # vLLM max_model_len (8192) → 400. Cap output; SQL answers are short
            # and schema is discovered via tools, so prompts stay small.
            "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "2048",
        },
    )

    n_tool_calls = 0
    subtype = "unknown"
    async for msg in query(prompt=_build_prompt(ex), options=options):
        if isinstance(msg, AssistantMessage):
            for b in msg.content:
                if isinstance(b, ToolUseBlock):
                    n_tool_calls += 1
                    if verbose:
                        print(f"    [tool] {b.name}({b.input})")
        elif isinstance(msg, ResultMessage):
            subtype = msg.subtype

    final_sql = state["submitted"] or ""      # captured from submit_query arg
    meta = {"n_tool_calls": n_tool_calls, "result_subtype": subtype,
            "used_tools": n_tool_calls > 0, "submitted": bool(state["submitted"])}
    return final_sql, meta


async def run(args):
    examples = ds.LOADERS["bird"](args.root)
    if args.limit:
        examples = examples[: args.limit]
    print(f"[{args.tag}] {len(examples)} examples | model={args.model} "
          f"| base_url={args.base_url} | concurrency={args.concurrency}")

    missing = [e for e in examples if not os.path.exists(e.db_path)]
    if missing:
        print(f"WARNING: {len(missing)} db files missing, e.g. {missing[0].db_path}")

    preds: list[str] = [""] * len(examples)
    metas: list[dict] = [{}] * len(examples)
    sem = asyncio.Semaphore(args.concurrency)
    done = 0
    t0 = time.time()

    async def _one(k: int):
        nonlocal done
        async with sem:
            try:
                sql, meta = await solve(
                    examples[k], model=args.model, base_url=args.base_url,
                    max_turns=args.max_turns, verbose=args.verbose)
            except Exception as e:            # keep the batch alive
                sql, meta = "", {"n_tool_calls": 0, "result_subtype": "crash",
                                 "used_tools": False, "error": str(e)}
            preds[k], metas[k] = sql, meta
            done += 1
            print(f"  solved {done}/{len(examples)}", end="\r", flush=True)

    await asyncio.gather(*(_one(k) for k in range(len(examples))))
    agent_s = time.time() - t0
    print(f"\n[{args.tag}] agent loop done in {agent_s:.1f}s")

    # ---- score (reuse the harness; execution is sync, run in a thread pool) ----
    from concurrent.futures import ThreadPoolExecutor
    results = [None] * len(examples)

    def _score(k):
        e = examples[k]
        correct, pred_ok, gold_ok, detail = score_one(
            e.db_path, preds[k], e.gold_sql, args.exec_timeout)
        return k, correct, pred_ok, gold_ok, detail

    with ThreadPoolExecutor(max_workers=args.exec_workers) as ex:
        for k, correct, pred_ok, gold_ok, detail in ex.map(
                _score, range(len(examples))):
            results[k] = (correct, pred_ok, gold_ok, detail)

    # ---- aggregate (mirror run_eval.py summary shape + agentic extras) ----
    n = len(examples)
    n_gold_ok = sum(1 for r in results if r[2])
    n_correct = sum(1 for r in results if r[0])
    n_valid = sum(1 for r in results if r[1])
    n_used_tools = sum(1 for m in metas if m.get("used_tools"))
    n_submitted = sum(1 for m in metas if m.get("submitted"))
    mean_tool_calls = sum(m.get("n_tool_calls", 0) for m in metas) / (n or 1)
    ex_denom = n_gold_ok or 1

    summary = {
        "tag": args.tag, "dataset": "bird", "backend": "agent",
        "model": args.model, "n": n, "gold_runnable": n_gold_ok,
        "EX": round(n_correct / ex_denom, 4),
        "EX_over_all": round(n_correct / n, 4),
        "valid_sql_rate": round(n_valid / n, 4),
        "tool_use_rate": round(n_used_tools / n, 4),
        "submit_rate": round(n_submitted / n, 4),
        "mean_tool_calls": round(mean_tool_calls, 2),
        "agent_seconds": round(agent_s, 1),
    }
    # per-difficulty
    diffs = {}
    for k, e in enumerate(examples):
        if not e.difficulty:
            continue
        d = diffs.setdefault(e.difficulty, [0, 0])
        if results[k][2]:
            d[1] += 1
            if results[k][0]:
                d[0] += 1
    if diffs:
        summary["EX_by_difficulty"] = {
            d: round(c / (t or 1), 4) for d, (c, t) in diffs.items()}

    print(json.dumps(summary, indent=2))

    out = args.out or f"../results/{args.tag}_bird.jsonl"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write(json.dumps({"__summary__": summary}) + "\n")
        for k, e in enumerate(examples):
            correct, pred_ok, gold_ok, detail = results[k]
            f.write(json.dumps({
                "id": e.id, "db_id": e.db_id, "difficulty": e.difficulty,
                "question": e.question, "pred_sql": preds[k], "gold_sql": e.gold_sql,
                "correct": correct, "pred_ok": pred_ok, "gold_ok": gold_ok,
                **detail, "agent": metas[k],
            }) + "\n")
    print(f"[{args.tag}] wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="BIRD dev root dir")
    ap.add_argument("--model", default="sql-agent",
                    help="litellm model_name (see litellm_config.yaml)")
    ap.add_argument("--base-url", default="http://localhost:4000",
                    help="litellm proxy (Anthropic-compatible) base URL")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--concurrency", type=int, default=4,
                    help="in-flight questions (bounded by the single vLLM backend)")
    ap.add_argument("--max-turns", type=int, default=12,
                    help="cap on agentic turns per question")
    ap.add_argument("--exec-timeout", type=float, default=30.0)
    ap.add_argument("--exec-workers", type=int, default=8)
    ap.add_argument("--verbose", action="store_true", help="print each tool call")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="arm4")
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
