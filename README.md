# nl2sql — an Execution-Accuracy harness and a NL→SQL model bake-off

Can a 7B model be fine-tuned into a better text-to-SQL model than it already is?
I built the measurement first, then ran the experiment on **the full BIRD dev
set — all 1,534 questions across 11 databases**, every arm scored the same way.

The answer turned out to be no, and what *did* work was cheaper.

## Results — Execution Accuracy on BIRD dev (n = 1,534)

| # | model / protocol | EX |
|---|---|---|
| 1 | Arctic-Text2SQL-R1-7B + self-consistency (n=8) | **63.7 %** |
| 2 | Arctic-Text2SQL-R1-7B, greedy | 62.0 % |
| 3 | Qwen2.5-Coder-7B + self-consistency (n=8) | 56.6 % |
| 4 | Qwen2.5-Coder-7B, fp8, greedy | 53.7 % |
| 5 | Qwen2.5-Coder-7B, bf16, greedy *(baseline)* | 51.2 % |
| 6 | Qwen2.5-Coder-7B + **LoRA SFT → GRPO** (200 steps) | 51.0 % |
| 7 | Qwen2.5-Coder-7B + **LoRA SFT** | 49.7 % |
| 8 | SLM-SQL-1.5B | 37.1 % |

### What the numbers say

**Fine-tuning did not beat the base model.** SFT *lost* 1.5 points (51.2 → 49.7)
and GRPO only clawed back to parity (51.0). The cause was diagnosable, not a
hyperparameter accident: the training data let the model memorise schema-specific
answers rather than learn the mapping, and the loss cannot tell those apart —
`Street` vs `MailStreet` is about one token of loss and a completely wrong result
set. Written up in [`FINDINGS.md`](FINDINGS.md).

**Decoding beat training, for free.** Self-consistency at n=8 bought **+5.4
points** on the same weights (51.2 → 56.6) — more than SFT and GRPO combined,
with no training run. It helped the stronger model too, by less (+1.7).

**Published headline numbers are protocols, not models.** SLM-SQL reports 67.3 %;
single-pass it scores 43.3 %. The gap is sample-N + vote + merge. Compare
like-for-like or you are comparing a model to a system.

## Why trust these numbers

- **Execution Accuracy**, not string match: both queries are executed and their
  **result sets** compared, order-insensitively, in a timeout-guarded subprocess.
  A differently-worded query that returns the right rows counts as correct.
- **Harness self-test:** replaying BIRD's own gold SQL through the pipeline
  scores **99.9 %** — the handful of misses are gold queries that fail to execute,
  which bounds the harness's own error at ~0.1 %.
- **Full dev set every time**, never a 50-example slice. Early probes on the
  first 50 questions ranked configurations *wrongly* — they all sit in one hard
  database.
- One machine, one harness, identical prompts and extraction per model family.

## Write-ups

- [`FINDINGS.md`](FINDINGS.md) — portable lessons (why fine-tuning can make a
  model worse; how to read a benchmark number).
- [`TECHNIQUES.md`](TECHNIQUES.md) — the mechanisms: GRPO, self-consistency, LoRA.
- [`PROGRESS.md`](PROGRESS.md) — the run log, including what broke.

## Layout

```
eval/        the EX harness — CPU-only, no ML stack needed to run it
  db.py          execute SQL in a timeout-guarded subprocess
  compare.py     set-based, order-insensitive result comparison (EX)
  datasets.py    BIRD / Spider dev loaders → uniform Example
  schema.py      render CREATE TABLE DDL for the prompt
  prompts.py     per-model chat templates + SQL extraction
  predictors.py  backends: gold / echo / vllm (heavy deps lazy-imported)
  vote.py        self-consistency voting over n samples
  run_eval.py    CLI: predict → extract → execute → score → JSONL
train/       Unsloth SFT → GRPO scripts
agent/       an agentic arm: Claude Agent SDK driven against a local vLLM model
results/     per-run JSONL, one row per question, with both SQL strings
```

## Quick start

The harness itself runs on CPU and needs no ML stack.

1. Get **BIRD dev** (1,534 questions + the SQLite databases) from
   [bird-bench.github.io](https://bird-bench.github.io/) and unpack it into
   `data/dev/` — `data/` is gitignored, nothing large lives in this repo.
2. Self-test the harness against BIRD's own gold SQL. It must report EX ≈ 1.0;
   anything lower means the harness, not the model, is wrong:

```bash
python eval/run_eval.py --dataset bird --root data/dev \
       --backend gold --limit 50 --tag selftest
```

3. A real run serves a model with vLLM and needs the GPU stack in
   `requirements-gpu.txt`:

```bash
python eval/run_eval.py --dataset bird --root data/dev \
       --backend vllm --model Snowflake/Arctic-Text2SQL-R1-7B \
       --template arctic --tag arctic_full
```

Add `--n-samples 8 --temperature 0.8` to reproduce a self-consistency row.

## Hardware

Everything above was produced on a single **RTX 5060 Ti 16 GB** (Blackwell,
sm_120), models served with vLLM. Fitting a 7B model, fp8 weights, and n=8
sampling into 16 GB is most of why the run configuration looks the way it does;
[`BLACKWELL_BUGS.md`](BLACKWELL_BUGS.md) records what sm_120 broke on the way.

## License

MIT
