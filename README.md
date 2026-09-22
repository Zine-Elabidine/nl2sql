# nl2sql — EX harness + model bake-off

Goal: adopt strong open NL→SQL models **and** fine-tune our own, graded by
**Execution Accuracy (EX)** on BIRD dev (Spider as floor).
Findings and mechanisms are written up in `FINDINGS.md` and `TECHNIQUES.md`.

## Layout
- `eval/`   — the EX harness (CPU-only; no ML stack needed to run it)
  - `db.py`        execute SQL in a timeout-guarded subprocess
  - `compare.py`   set-based, order-insensitive result-set comparison (EX)
  - `datasets.py`  BIRD / Spider dev loaders → uniform `Example`
  - `schema.py`    render CREATE TABLE DDL for the prompt
  - `prompts.py`   per-model chat templates + SQL extraction
  - `predictors.py` backends: gold / echo / vllm (heavy deps lazy-imported)
  - `run_eval.py`  CLI: predict → extract → execute → score → JSONL + summary
- `data/`    benchmark data (BIRD dev.zip extracted here)
- `train/`   Unsloth SFT→GRPO scripts (Arm 3)
- `results/` per-run JSONL + summaries
- `configs/` run configs

## Env
- `uv` venv at `.venv` (Python 3.12). Harness deps already installed.
- GPU stack: `requirements-gpu.txt` (install once the GPU is free).

## Bake-off arms
1. SLM-SQL (Qwen2.5-Coder-1.5B) — adopt
2. Arctic-Text2SQL-R1-7B — adopt
3. Train our own — Unsloth SFT→GRPO, A/B Qwen2.5-Coder-3B vs Qwen3-4B
4. DeepEye-SQL (3B-MoE agentic) — eval-only reference

## Quick start
```bash
P=~/nl2sql/.venv/bin/python
cd ~/nl2sql/eval
# harness self-test (must report EX≈1.0):
$P run_eval.py --dataset bird --root ../data/dev --backend gold --limit 50 --tag selftest
```
