# Arm 4 — wiring a Claude Agent SDK loop to *our* model (not Anthropic)

How to drive the Claude Agent SDK against arbitrary (non-Anthropic) models, and
how we use that for Arm 4 to run a small agentic loop on BIRD with
**Arctic-7B / Qwen2.5-Coder-7B** served locally by vLLM.

## The trick, in one line

The Claude Agent SDK (and the `claude` CLI it spawns) talks **only** the
Anthropic Messages wire format (`POST /v1/messages`). You never hand it an
OpenAI/vLLM endpoint directly. Instead you point it at a **proxy that speaks
Anthropic on the front and translates to whatever the model speaks on the back**,
via two env vars:

```
ANTHROPIC_BASE_URL = <proxy>        # SDK posts /v1/messages here instead of api.anthropic.com
ANTHROPIC_API_KEY  = <token>        # opaque to the SDK; the proxy interprets it
```

That's the whole integration surface. No Anthropic key is ever used; the SDK
doesn't know or care what's behind the base URL. Our local vLLM needs no auth,
so the key can be any non-empty string.

## The translation layer

`litellm` does the Anthropic ⇄ OpenAI conversion (tool schemas, streaming SSE,
thinking-block handling). One detail matters when the backend is self-hosted:

```python
# An OpenAI-family provider pointed at a *custom* base_url (self-hosted vLLM)
# must hit /chat/completions, not OpenAI's /responses API. hosted_vllm forces it.
litellm_model = f"hosted_vllm/{model_name}"   # e.g. hosted_vllm/Arctic-Text2SQL-R1-7B
```

Two gotchas worth knowing up front:
1. **Strip `thinking` blocks** from assistant history before sending to
   OpenAI/vLLM gateways — they 400 on an unknown content variant.
2. **Always stream** on the litellm path, so a pre-content failure surfaces as a
   real HTTP non-200 instead of a 200 SSE carrying a swallowed error.

## What we build for Arm 4 (minimum viable replica)

We don't need the whole proxy — no `ref:` credentials, no multi-tenant, no
traces. We need one process that is Anthropic-in / vLLM-out. Two options:

### Option A — LiteLLM proxy in front of vLLM (least code) ✅ recommended

`litellm` ships a proxy server that already exposes an **Anthropic `/v1/messages`
passthrough** and translates to a `hosted_vllm` backend, as a CLI. Sketch:

```bash
# 1. serve our SQL model (OpenAI-compatible) with vLLM
vllm serve Snowflake/Arctic-Text2SQL-R1-7B \
    --quantization fp8 --gpu-memory-utilization 0.85 \
    --enforce-eager --max-model-len 8192 --port 8001

# 2. litellm proxy: Anthropic front  →  hosted_vllm back
#    config.yaml:
#      model_list:
#        - model_name: sql-agent
#          litellm_params:
#            model: hosted_vllm/Snowflake/Arctic-Text2SQL-R1-7B
#            api_base: http://localhost:8001/v1
#            api_key: none
litellm --config config.yaml --port 4000

# 3. point the Claude Agent SDK at it
export ANTHROPIC_BASE_URL=http://localhost:4000
export ANTHROPIC_API_KEY=sk-anything      # ignored by vLLM
```

Then the agent code is just the SDK with our 3 SQLite tools (`list_tables`,
`describe_table`, `execute_sql`) as in-process MCP / SDK tools, driving the loop;
final SQL scored by the existing EX harness (`eval/compare.py`, `eval/db.py`).

### Option B — tiny FastAPI shim (if litellm's passthrough misbehaves)

Write the ~30 lines that matter yourself: a `POST /v1/messages` that calls
`litellm.anthropic.messages.acreate(model="hosted_vllm/<model>",
api_base="http://localhost:8001/v1", ...)`, streaming, with the thinking-strip.
Same idea, fully under our control. Only fall back to this if A's built-in
adapter drops tool_calls or streaming.

## Open risk (unchanged from the design discussion)

Arctic-7B is a **single-shot SQL/reasoning** model — it may ignore the tool
schema and just emit SQL instead of calling `execute_sql`. If tool-calling is
unreliable, switch the backend `model_name` to **Qwen2.5-Coder-7B-Instruct**
(reliable Hermes-style tool calls in vLLM) and keep everything else identical —
the wiring above is model-agnostic by construction. Serve Qwen with
`--enable-auto-tool-choice --tool-call-parser hermes`.

## TL;DR

Claude SDK → `ANTHROPIC_BASE_URL` → (litellm proxy: Anthropic⇄`hosted_vllm`) →
local vLLM (our model). A stock litellm proxy covers the whole last mile.

---

# Claude Agent SDK — official API (from docs.claude.com, verified 2026-07)

`pip install claude-agent-sdk` — Python package `claude_agent_sdk`.

## Two entry points

- **`query(prompt, options)`** → async generator of `Message`. One-off, fresh
  session, no interrupts. Good enough for Arm 4 (one BIRD question per call).
- **`ClaudeSDKClient(options)`** → persistent multi-turn session, `.connect()`,
  `.query()`, `.receive_response()`, `.interrupt()`. Use only if we want to
  reuse warm context across questions (we don't — each question is independent).

We want **`query()`**, called once per BIRD example.

## Custom tools = in-process MCP server (no subprocess)

Four parts per tool via the **`@tool(name, description, input_schema, annotations=None)`**
decorator. `input_schema` is either a simple `{"name": type}` dict (SDK converts
to JSON Schema; **every key is required**) or a full JSON Schema dict (needed for
enums / optional / ranges). Handler is `async def(args: dict) -> dict` and MUST
return `{"content": [ {"type": "text", "text": ...} ], "is_error": bool?}`.

Key rule for our loop: **return `"is_error": True` instead of raising** — a raised
exception kills the whole `query()`; an `is_error` result feeds the error back to
the model so it self-corrects. That's exactly what we want when `execute_sql`
hits a SQL error.

Bundle tools with **`create_sdk_mcp_server(name, version, tools=[...])`** → pass
as `mcp_servers={"<server>": server}`. Tool names Claude sees are
**`mcp__<server>__<tool>`**; list them in `allowed_tools` to skip the permission
prompt. Mark read-only tools `annotations=ToolAnnotations(readOnlyHint=True)` so
they can batch in parallel.

## Reading the stream

`async for message in query(...)`:
- `AssistantMessage.content` → list of blocks (`TextBlock.text`,
  `ToolUseBlock{name,input}`, `ThinkingBlock`).
- `ResultMessage` = final; `.subtype` ("success" | "error_max_turns" | …),
  `.result` (final text), `.is_error`.

`ClaudeAgentOptions` fields we use: `system_prompt`, `model`, `mcp_servers`,
`allowed_tools`, `tools=[]` (drop ALL built-ins so the model can only use ours),
`permission_mode="default"`, `max_turns` (cap the self-correct loop),
`env` (our `ANTHROPIC_BASE_URL`/`ANTHROPIC_API_KEY`), `cwd`, `setting_sources=[]`
(don't load any local CLABUDE settings).

## Arm 4 skeleton (maps to our EX harness)

```python
import asyncio, sqlite3
from typing import Any
from claude_agent_sdk import (
    tool, create_sdk_mcp_server, query,
    ClaudeAgentOptions, ToolAnnotations,
    AssistantMessage, ToolUseBlock, ResultMessage, TextBlock,
)

# db_path is per-question; bind it via a closure/contextvar per call.
def build_server(db_path: str):
    def _ro():
        return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)

    @tool("list_tables", "List all table names in the database", {},
          annotations=ToolAnnotations(readOnlyHint=True))
    async def list_tables(args):
        rows = _ro().execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        return {"content": [{"type": "text", "text": ", ".join(r[0] for r in rows)}]}

    @tool("describe_table", "Show CREATE statement + sample rows for a table",
          {"table": str}, annotations=ToolAnnotations(readOnlyHint=True))
    async def describe_table(args):
        c = _ro()
        try:
            ddl = c.execute("SELECT sql FROM sqlite_master WHERE name=?",
                            (args["table"],)).fetchone()
            if not ddl:
                return {"content": [{"type": "text",
                        "text": f"no such table {args['table']}"}], "is_error": True}
            sample = c.execute(f'SELECT * FROM "{args["table"]}" LIMIT 3').fetchall()
            return {"content": [{"type": "text", "text": f"{ddl[0]}\nsample: {sample}"}]}
        except Exception as e:
            return {"content": [{"type": "text", "text": str(e)}], "is_error": True}

    @tool("execute_sql", "Run a read-only SQL query and return up to 20 rows",
          {"sql": str})
    async def execute_sql(args):
        try:
            rows = _ro().execute(args["sql"]).fetchall()
            return {"content": [{"type": "text", "text": repr(rows[:20])}]}
        except Exception as e:               # feed the error back, don't raise
            return {"content": [{"type": "text",
                    "text": f"SQL error: {e}"}], "is_error": True}

    return create_sdk_mcp_server(name="sqlite", version="1.0.0",
                                 tools=[list_tables, describe_table, execute_sql])

SYSTEM = ("You are a text-to-SQL agent over a SQLite database. Discover the "
          "schema with list_tables/describe_table, draft a query, validate it "
          "with execute_sql, fix any error, then output the FINAL SQL alone in "
          "a ```sql fenced block.")

async def solve(question: str, db_path: str) -> str:
    server = build_server(db_path)
    opts = ClaudeAgentOptions(
        system_prompt=SYSTEM,
        model="sql-agent",                       # litellm model_name → our vLLM
        mcp_servers={"sqlite": server},
        allowed_tools=["mcp__sqlite__list_tables",
                       "mcp__sqlite__describe_table",
                       "mcp__sqlite__execute_sql"],
        tools=[],                                # no built-ins; only our 3 tools
        permission_mode="default",
        max_turns=12,
        setting_sources=[],
        env={"ANTHROPIC_BASE_URL": "http://localhost:4000",
             "ANTHROPIC_API_KEY": "sk-anything"},
    )
    final = ""
    async for msg in query(prompt=f"Question: {question}", options=opts):
        if isinstance(msg, AssistantMessage):
            for b in msg.content:
                if isinstance(b, TextBlock):
                    final = b.text            # keep last assistant text
        elif isinstance(msg, ResultMessage):
            final = msg.result or final
    return final   # → feed to eval/prompts.extract() → eval/compare.score_one()
```

Then wrap `solve()` over BIRD dev, extract SQL with the existing
`eval/prompts.extract`, and score with `eval/compare.score_one` — same EX number,
comparable to the other arms. Run serially or with a small asyncio gather
(bounded — the single 16GB vLLM backend is the bottleneck).

## Gotchas confirmed from docs

- Python `@tool` forwards only `content` + `is_error` (no `structuredContent`) —
  fine for us, we only return text.
- Simple-dict schema keys are ALL required; for an optional arg omit it from the
  schema and read `args.get(...)`.
- `tools=[]` removes all built-ins (Read/Bash/etc.) so the model can't wander;
  only our MCP tools remain.
- Raising in a handler ends `query()`; always catch → `is_error`.

Sources:
- https://code.claude.com/docs/en/agent-sdk/custom-tools
- https://code.claude.com/docs/en/agent-sdk/python
