# NL→SQL Bake-off — Progress Log

**Last updated:** 2026-06-26
**Owner:** Zine · **Machine:** local RTX 5060 Ti 16GB (Blackwell sm_120), CUDA driver 13.2
**Project dir:** `~/nl2sql/` · **Notes:** `FINDINGS.md`, `TECHNIQUES.md`

---

## 1. Goal & Plan

Adopt strong open NL→SQL models **and** fine-tune our own, then compare single-model vs
agentic — everything graded by **Execution Accuracy (EX)** on **BIRD dev**. Four "arms":

1. **SLM-SQL 1.5B** (`cycloneboy/SLM-SQL-1.5B`) — adopt
2. **Arctic-7B** (`Snowflake/Arctic-Text2SQL-R1-7B`) — adopt
3. **Train our own** — Unsloth SFT→GRPO, A/B Qwen2.5-Coder-3B vs Qwen3-4B
4. **DeepEye-SQL** (3B-MoE agentic) — eval-only reference

GRPO training **dataset** choice (BIRD train vs Spider vs synthetic) is **parked** until Arm 3.

---

## 2. Status snapshot

| Item | State |
|---|---|
| Environment (uv + Py3.12 + CUDA-13 stack) | ✅ working |
| EX eval harness | ✅ built & validated (gold = EX 1.0) |
| BIRD dev data (1534 ex, 11 DBs) | ✅ downloaded & extracted |
| Models cached (SLM-SQL 1.5B, Arctic-7B) | ✅ |
| Arm 1 — SLM-SQL eval | 🔄 in progress (prompt-format tuning) |
| Arm 2 — Arctic-7B | ⬜ next |
| Arm 3 — train our own | ⬜ |
| Arm 4 — DeepEye-SQL | ⬜ |
| Spider floor set | ⬜ deferred (optional) |

---

## 3. Environment — final working stack

- **Python 3.12.13** via `uv` venv at `~/nl2sql/.venv` (system Python is **3.14**, too new for the ML stack).
- **torch 2.11.0+cu130**, torchvision 0.26.0, torchaudio 2.11.0 — all from the **cu130** index.
- **vLLM 0.23.0**, transformers 5.12.1, trl 1.7.0, peft, bitsandbytes.
- CPU-only harness deps: huggingface_hub, datasets, sqlparse, pandas, tqdm.

**Required runtime env for vLLM on this box** (baked into `eval/predictors.py`):
```
VLLM_USE_FLASHINFER_SAMPLER=0     # FlashInfer can't read sm_120 → "requires sm75+"
VLLM_ATTENTION_BACKEND=FLASH_ATTN # FlashAttention detects sm_120 fine
VLLM_CACHE_ROOT=~/nl2sql/.vllm_cache  # default ~/.cache/vllm is root-owned
```
Plus LLM kwargs: `enforce_eager=True`, `max_num_seqs=16`, `gpu_memory_utilization=0.85`.

---

## 4. Obstacles & remediations (chronological)

### 4.1 GPU occupied by a root vLLM container
- **Symptom:** 12.7GB/16GB used by `VLLM::EngineCore` (root); `kill` → "Operation not permitted".
- **Cause:** a Docker Compose service `vllm_local` (`vllm/vllm-openai:cu130-nightly`, restart `unless-stopped`) at `~/vllm-local/`. Killing the PID just made Docker relaunch it.
- **Fix:** `docker stop vllm_local`. With `unless-stopped`, a manually-stopped container does **not** auto-restart on reboot. Confirmed it stays `Exited` after the later reboot.

### 4.2 System Python 3.14 too new
- **Cause:** torch/vLLM/unsloth wheels top out at 3.12/3.13.
- **Fix:** install `uv`; create a 3.12 venv (`uv venv --python 3.12`).

### 4.3 BIRD dev download crawling (~7 MB/min)
- **Cause:** the official OSS endpoint (`oss-cn-beijing`) throttles **per connection**; a fresh connection burst-tested at 2.8 MB/s.
- **Fix:** 8-way **segmented download** (curl byte-ranges) + concat → 99s instead of ~45 min. (No aria2c available.)
- BIRD dev layout: `dev_20240627/dev.json` (1534), `dev_databases/<db>/<db>.sqlite` (nested zip), 11 DBs.

### 4.4 Harness fork-from-threads deadlock (the important bug)
- **Symptom:** gold self-test gave EX 0.979 (should be 1.0) with **non-deterministic timeouts** — same SQL would time out for gold but pass for pred, or vice versa.
- **Cause:** `db.py` ran each query in a **forked subprocess** for timeout control, called from **ThreadPoolExecutor** workers. `fork()` in a multithreaded process inherits locked mutexes → child deadlocks until the 30s timeout kills it.
- **Fix:** dropped subprocess/fork entirely; now use SQLite's thread-safe `connection.interrupt()` driven by a `threading.Timer`. Result: **gold = EX 1.0**, and eval ~**20× faster** (61.5s → 2.9s on 200).
- Residual: 2/1534 golds are genuinely pathological (>120s even standalone) → correctly excluded, same as the official evaluator.

### 4.5 CUDA version mismatch hell (torch ↔ vLLM)
- **Chain of failures:**
  1. First installed torch **cu128**; vLLM 0.23.0 needs **CUDA 13** → `ImportError: libcudart.so.13`.
  2. torchvision/torchaudio got pulled from PyPI built for a different CUDA → `operator torchvision::nms does not exist`, then `libcudart.so.13` from torchaudio.
  3. Switching to plain cu130 gave torch **2.12.1**, but vLLM 0.23.0 **pins torch==2.11.0** → `_C.abi3.so: undefined symbol` (ABI mismatch).
- **Root rule learned:** vLLM 0.23.0 ⇒ **torch==2.11.0 built for cu130**. That exact wheel exists.
- **Fix:** `uv pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url .../cu130`. Stack consistent.
- Also: removed torchvision/torchaudio early as not needed for text-only — but vLLM pins them, so reinstalled the matching cu130 builds.

### 4.6 Two concurrent installs corrupting the venv
- **Cause:** a background install whose torch step failed kept its `vllm` step running, while a retry also ran — both writing the same venv.
- **Fix:** `pkill` all `uv pip install`, then one clean sequential install.

### 4.7 Disk saturation (PC near-full, then a freeze)
- **Symptom:** `/` at **97% (8.8G free)**; installs timing out.
- **Investigation:** uv cache 16G (mostly hardlinked into venv, freed only ~4G); HF cache 38G (Arctic 15G + SLM-SQL 3.4G + **non-ours Qwen3.5-9B-AWQ 12G + Qwen3.5-4B 8.8G**).
- **Real culprit:** **Docker build cache 68.6GB (44.7GB reclaimable)**.
- **Fix:** `docker builder prune -f` → freed 44.7GB → disk down to **77% (55G free)**. Did **not** delete the user's Qwen models.

### 4.8 vLLM cache PermissionError
- **Cause:** `~/.cache/vllm` owned by **root** (created by the old container) → vLLM can't write torch-compile cache.
- **Fix:** `VLLM_CACHE_ROOT=~/nl2sql/.vllm_cache` (can't chown root's dir without sudo).

### 4.9 FlashInfer doesn't support Blackwell
- **Symptom:** `RuntimeError: FlashInfer requires GPUs with sm75 or higher` + benign `Failed to get device capability: SM 12.x requires CUDA >= 12.9`.
- **Cause:** bundled FlashInfer build can't read sm_120.
- **Fix:** `VLLM_USE_FLASHINFER_SAMPLER=0` + `VLLM_ATTENTION_BACKEND=FLASH_ATTN` (FlashAttention works on sm_120).

### 4.10 PC freeze during the first full run
- **Cause (likely):** vLLM CUDA-graph capture burst (~51 batch sizes up to 512) on a 16GB box — confirmed against the user's own `~/vllm-local/NOTES.md`.
- **Fix:** `enforce_eager=True` (no graph capture) + `max_num_seqs=16` cap + `gpu_memory_utilization=0.85`. Now bounded to ~13.6GB; verified stable (4GB at load, no runaway).
- `vllm_local` stayed `Exited` after the reboot (restart policy correctly inert).

### 4.11 SLM-SQL prompt-format mismatch (current focus)
- **Generic CREATE-TABLE prompt, greedy:** EX **37.1%**, valid-SQL **84%** — model sometimes hallucinates columns; mostly emits SQL directly.
- **OmniSQL CoT prompt, max_tokens 2048:** EX **26%**, valid **58%** — *worse*, because the model **rambles 7600+ chars of step-by-step reasoning and gets truncated before the SQL block**.
- **Key finding from their repo** (`data/bird_dev/1.5b/...`): the file is `sampling_think_sql_merge_pred_major_voting_sqls.sql` and `metric.json` shows **mj_metric all = 67.28%** — i.e. their headline is **sampling + think (CoT) + majority-voting + merge**, NOT single greedy. (Their inference code is unreleased.)
- **Implications:**
  - SLM-SQL **is** a thinking model → CoT prompt is correct, but needs **max_tokens ≥ 4096** so reasoning finishes.
  - Their 67% is a **self-consistency** number; a fair **single greedy** EX will be lower (expect ~50s).
- **Next:** re-run greedy with CoT + max_tokens 4096 + sample-rows 3; report single-greedy EX as the bake-off number and note the voting ceiling separately. (This re-run was the pending step at compaction.)

---

## 5. The EX harness (`~/nl2sql/eval/`)

- `db.py` — `execute_sql(db, sql, timeout)`; timeout via `threading.Timer` + `connection.interrupt()` (thread-safe, no fork).
- `compare.py` — set-based, order-insensitive result-set match; order enforced only when gold has top-level `ORDER BY`. Cells stringified (SQLite dynamic typing).
- `datasets.py` — BIRD/Spider dev loaders → uniform `Example(id, db_id, question, gold_sql, db_path, evidence, difficulty)`; auto-discovers folder layout.
- `schema.py` — CREATE TABLE DDL from `sqlite_master` (+ optional sample rows), lru-cached per db.
- `prompts.py` — templates `generic`, `omnisql` (OmniSQL/SynSQL CoT, evidence folded into question), `slm_sql`→omnisql, `arctic`; `extract()` pulls SQL from fenced block / after `</think>` / last SELECT|WITH.
- `predictors.py` — `gold` (self-test), `echo` (wiring test), `vllm` (Blackwell env + enforce_eager + max_num_seqs baked in).
- `run_eval.py` — CLI: predict → extract → execute → score → JSONL + summary (EX over gold-runnable, EX_over_all, valid_sql_rate, EX_by_difficulty). Flags: `--limit --sample-rows --max-model-len --max-tokens --max-num-seqs --gpu-mem-frac --cuda-graphs`.

**Validation:** gold backend = EX 1.0 across all 11 DBs (1532/1534); echo backend ≈ EX 0.

---

## 6. Results so far (BIRD dev, single greedy unless noted)

| Run | Model | Prompt | EX | valid-SQL | notes |
|---|---|---|---|---|---|
| gold_full | (gold self-test) | — | 1.000 | 0.999 | harness validation |
| slmsql_full | SLM-SQL 1.5B | generic | 0.371 | 0.840 | direct output; column hallucinations |
| slmsql_omni50 | SLM-SQL 1.5B | OmniSQL CoT, 2048 tok | 0.260 | 0.580 | truncated mid-reasoning |
| (pending) | SLM-SQL 1.5B | OmniSQL CoT, 4096 tok | — | — | next step |

**Reference (their paper, majority-voting, NOT greedy):** SLM-SQL 1.5B = 67.3% dev / 70.5% test.

---

## 7. Key facts / gotchas to remember

- **BIRD:** dev (1534, public) is our yardstick; test is private (submission only); train (9.4k, 69 DBs) only needed for GRPO execution reward (parked).
- **EX = single-greedy** in our harness; published SOTA numbers often include **self-consistency/voting** — compare like-for-like.
- **Blackwell + vLLM:** torch 2.11.0+**cu130**, FlashAttention (not FlashInfer), eager mode, cap `max_num_seqs`.
- **Models load fully local** from `~/.cache/huggingface` — no network at inference.
- **`~/vllm-local/`** is a separate local vLLM container project (CUDA-graph/GDN gotchas). Stopped; do not auto-restart.

---

## 8. Immediate next steps

1. Re-run SLM-SQL greedy: OmniSQL CoT + `--max-tokens 4096 --sample-rows 3` → record single-greedy EX (full 1534).
2. Arm 2: Arctic-7B (its own prompt; bf16, fits 16GB) → EX.
3. (Optional) reproduce SLM-SQL voting ceiling for context.
4. Arm 3 scaffolding (Unsloth SFT→GRPO) — pick GRPO dataset then.
5. Arm 4: DeepEye-SQL agentic eval.

---

## Arm 1 RESOLVED (2026-06-26) — SLM-SQL-1.5B is not a single-pass model

Diagnosed the low EX. Root cause is NOT parsing and NOT truncation — it is
**greedy-decoding degeneration + the headline being a voting number.**

Raw-output inspection: the model writes good "### Step 1..6" reasoning, then at
the reasoning→SQL transition collapses into `</details></details>...` /
`</div></div>...` loops and never reaches the ```sql block. Example that escapes
the loop produces clean correct SQL.

Probe results (BIRD dev, 50 ex, OmniSQL CoT prompt, --sample-rows 3, max-tokens 4096):
| decoding | EX | valid-SQL |
|---|---|---|
| generic prompt, greedy | 0.371 | 0.84 |
| OmniSQL CoT, greedy | 0.26 | 0.58 |
| OmniSQL CoT, +rep-penalty 1.05 | 0.14 | 0.28 |  (rep-penalty hurts SQL too)
| OmniSQL CoT, sample T=0.8 top_p 0.95 | 0.28 | 0.64 |

Among the 43 NON-degenerate outputs at T=0.8: valid 32/43, correct 14/43 → EX 0.33.
So loops explain only ~7/50; the model is genuinely ~30% EX single-pass.

CONCLUSIONS:
- SLM-SQL's 67.3% = sampling + think + **majority-voting + merge** over many
  candidates. Single greedy/sample ≈ 28-37% EX. To hit 67% it needs 8+ samples +
  voting + correction → operationally a mini-pipeline, not a cheap one-shot model.
  This is a material input to the ship decision (a "1.5B" that needs voting is not
  as cheap as it looks).
- CoT prompt < generic prompt at single-pass (0.26 vs 0.371) because CoT gives
  room to degenerate; CoT only pays off WITH voting.

Harness changes: predictors.VLLMPredictor + run_eval now expose --temperature,
--top-p, --repetition-penalty (were hardcoded greedy). Verified working.

Optional follow-up (parked): add k-sample majority-voting to reproduce the 67%
ceiling and validate our harness against their published number.

NEXT: Arm 2 — Arctic-Text2SQL-R1-7B (clean GRPO execution-reward model, designed
for single greedy; should not have this failure mode). This is the real
single-model quality reference.

---

## Scorer validated against official BIRD eval (2026-06-26)

Question raised: do our EX numbers match the official BIRD definition?
Answer: YES. Re-scored saved SLM-SQL predictions with the official
`evaluation_ex.py` logic verbatim (`set(cursor.fetchall())`, func_timeout 30s,
plain sqlite connect):

| run | our EX (old scorer) | official EX | valid |
|---|---|---|---|
| SLM-SQL CoT greedy | 0.260 | 0.280 | 0.58 |
| SLM-SQL CoT sample | 0.280 | 0.280 | 0.64 |

Diff = +0.02 / +0.00 → our scorer was faithful; the 1-example gap on greedy was
our str()-stringify under-counting int-vs-float (gold 1.0 vs pred 1).

=> The 30% vs 67% gap is ENTIRELY inference protocol (single-pass vs
sampling+voting+merge), NOT our eval pipeline.

ACTION TAKEN: rewrote compare._normalize to be official-faithful by default —
pure `set(tuple(row))`, native types, no stringify, no cell-sort; ORDER BY no
longer enforced by default (BIRD EX ignores it) — enforce_order=True kept as a
stricter debug path. Re-validated: gold self-test still EX 1.000 (100 ex).

Why we didn't just use the official script: it's only a ~5-line batch scorer
over a pre-made predict.json — it does no generation, prompting, schema
rendering, SQL extraction, or diagnostics (valid-SQL rate, EX-by-difficulty,
per-example JSONL). We'd build 90% of the harness anyway; correct move = mirror
its exact comparison inside our harness (now done & validated).

---

## Arm 2 RESULT (2026-06-26) — Arctic-Text2SQL-R1-7B, FP8, FULL 1534

First headline-comparable number (full BIRD dev, official-faithful scorer):

| metric | value |
|---|---|
| EX | 0.6208 (62.1%) |
| valid_sql_rate | 0.9941 |
| EX simple/moderate/challenging | 0.679 / 0.540 / 0.507 |
| gold_runnable | 1532/1534 |
| gen_seconds | 2128 (~35 min), exec 67s |

Config: --template arctic --sample-rows 3 --max-model-len 8192 --max-tokens 2048
--max-num-seqs 8 --gpu-mem-frac 0.92 --quantization fp8. tag=arctic_full.

vs published 68.9 dev (bf16) → we are ~6.8 pts low. FP8 explains only ~1-2;
valid-SQL is 99.4% so it's NOT parsing/degeneration — likely PROMPT/SCHEMA-FORMAT
mismatch (our `arctic` template is bare generic; Arctic trained on OmniSQL/SynSQL
prompt + specific schema serialization). Two follow-ups to close the gap:
  (a) bf16 run at --max-model-len 4096 (isolate the FP8 cost),
  (b) re-run with --template omnisql (Arctic's training-time prompt).

Note: FP8 EX is arguably the MORE relevant number for our deployment (we'd serve
FP8 on the 5060Ti anyway). bf16 = literature-comparable.

---

## Arm 1 FULL-DEV numbers (2026-06-26) — SLM-SQL-1.5B single-pass

Ran full 1534 (probes were biased — first 50 are all california_schools, a hard DB):

| config | EX | valid-SQL | simple/mod/chall |
|---|---|---|---|
| OmniSQL CoT + sample T=0.8 top_p0.95 (tag slmsql_full_samp) | 0.4328 | 0.782 | .502/.343/.278 |
| generic prompt + greedy (tag slmsql_full_generic) | 0.4060 | 0.842 | .485/.305/.222 |

=> SLM-SQL single-pass honest benchmark = **43.3%** (intended omnisql+sampling
config wins on full dev — opposite of the probe). 67.3% headline stays a
sampling+voting+merge number. ~22% of sample-run outputs still degenerate
(valid 0.782), capping single-pass.

LESSON REINFORCED: never trust the --limit 50 probe as a result; first 50 of
BIRD dev = one hard DB. Probes are smoke tests only; report full 1534.

## SCOREBOARD so far (full BIRD dev, official-faithful EX)
| arm | EX | notes |
|---|---|---|
| SLM-SQL-1.5B single-pass | 43.3% | needs voting for its 67.3% headline |
| Arctic-7B (FP8) | 62.1% | ~7pt under bf16 68.9 (FP8 + prompt/schema gap) |
| our own fine-tune | TBD | Arm 3 |
| DeepEye agentic | TBD | Arm 4 |

---

## SESSION CLOSE — 2026-06-26 (RESUME HERE NEXT TIME)

State: clean. GPU free. No background runs. All results in ~/nl2sql/results/*.jsonl.

DONE this session:
- Harness scorer made official-BIRD-faithful (set(tuple(row)), native types);
  validated vs evaluation_ex.py to within 1/50; gold self-test EX 1.000.
- Predictor/run_eval now expose --temperature --top-p --repetition-penalty
  --quantization (fp8). All committed in eval/.
- Arm 1 (SLM-SQL-1.5B) DONE: single-pass full-dev EX = 43.3% (omnisql+sample,
  best) / 40.6% (generic+greedy). 67.3% headline = sampling+voting+merge.
- Arm 2 (Arctic-7B FP8) DONE: full-dev EX = 62.1%, valid 99.4%.

SCOREBOARD (full BIRD dev, official EX): SLM-SQL 43.3% | Arctic-7B(FP8) 62.1%.

NEXT (in priority order):
1. Arm 3 — train our own: Unsloth SFT->GRPO, A/B Qwen2.5-Coder-3B vs Qwen3-4B.
   FIRST decide GRPO dataset (BIRD train 9.4k = "hardcore", was parked). Target:
   beat/match Arctic 62% at smaller size. Eval every checkpoint by EX not loss.
2. Refinement (optional, do before declaring Arctic final): re-run Arctic with
   --template omnisql to close the ~7pt gap to 68.9; if it helps, it also helps
   our own model's eval prompt.
3. Refinement (optional): reproduce SLM-SQL voting ceiling (k-sample + majority
   vote) to validate our harness against their published 67.3%.
4. Arm 4 — DeepEye-SQL agentic eval.
5. (Optional) Arctic bf16 @ max-model-len 4096 to isolate FP8 cost; Spider floor.

Useful run templates:
  # adopt-model full dev:
  python run_eval.py --dataset bird --root ~/nl2sql/data/dev_20240627 \
    --backend vllm --model <hf_id> --template <arctic|slm_sql|omnisql|generic> \
    --sample-rows 3 --max-model-len 8192 --max-tokens 2048 --max-num-seqs 8 \
    --gpu-mem-frac 0.92 [--quantization fp8] [--temperature 0.8 --top-p 0.95] \
    --tag <name>
  # probes are SMOKE TESTS ONLY (first 50 = one hard DB, biased) — never report.

---

## SESSION 4 (2026-06-28) — Arm 3 training+eval underway

### New eval backend: `hf` (4-bit base + optional LoRA adapter, HF-generate)
Added to `eval/predictors.py` (HFPredictor) + `eval/run_eval.py` (`--backend hf
--adapter <dir>`). Loads the cached 4-bit base on GPU and optionally attaches a
PEFT adapter — the exact precision we TRAINED on. No bf16 download, no merged copy,
fits the 14B. This is how the whole 16-cell grid is scored (one consistent precision
= option (c)). Zero-shot = omit --adapter. ~32 min/full-1534 run (slower than vLLM,
that's the tradeoff). Greedy (temp 0) => batch size does NOT change EX, only speed.

### EVAL PROTOCOL (locked): grid scored uniformly at 4-bit+adapter via `hf`.
a/b/c precision study (bf16 / FP8 / 4-bit) run ONCE on the 7B-SFT (merge needed for
a,b). Final: merge→FP8/GGUF only for the grid WINNER's deploy artifact.

### Coder-7B numbers (BIRD dev, full 1534, 4-bit, generic, greedy)
| Run | EX | valid-SQL | notes |
|-----|-----|-----------|-------|
| zero-shot base | **0.512** | 0.900 | anchor. by-diff: simple .585 / mod .413 / chal .368 |
| SFT (outputs/coder7b_sft) | (running) | | adapter eval, batch 8 |

KEY: base instruct = **51.2%**, NOT the 69% literature figure (that's the
POST-TRAINED Coder-7B). Honest headroom: 51.2 base → beat Arctic 62.1 → match 69.
SCOREBOARD: Coder-7B base 51.2 | SLM-SQL 43.3 | Arctic-7B FP8 62.1.

### LESSON: eval OOM with adapter at batch-16 (fix: batch-8)
Base eval ran fine at --batch-size 16, but attaching the LoRA adapter OOM'd 16GB on
a long-schema batch. Two causes: (1) LoRA adds a second compute path per linear
(lora_A→lora_B + bf16 upcast) = extra activation memory vs bare base; (2) BIRD
schemas vary hugely, peak activation scales with the LONGEST example in a batch, so
a batch containing a giant-schema DB spikes the peak. Activation mem ~ linear in
batch size (weights fixed ~6GB), so batch16→8 ~halves it → fits (~9GB). Greedy +
per-example independent decode => batch size does NOT affect EX, only speed. Same
long-schema landmine that bit SFT (batch1) and GRPO (num_gen4) training.

### Coder-7B SFT result: SFT slightly HURT (key finding)
| Run | EX | valid-SQL | by-diff (s/m/c) |
|-----|-----|-----------|-----------------|
| base zero-shot | 0.512 | 0.900 | .585/.413/.368 |
| **SFT** (coder7b_sft) | **0.498** | 0.867 | .575/.397/.326 |
Net -1.4pt EX, -3.3pt valid. NOT a collapse: 161 regressions vs 139 gains (reshuffle),
204 invalid (up from ~154).
ROOT CAUSE = degraded SCHEMA-GROUNDING. SFT outputs are clean SQL (not truncated/
degenerate) but hallucinate column/table names: wrote `FRPMCountK12` instead of the
real `` `FRPM Count (K-12)` ``, `Street` instead of `MailStreet`, even typo `fprm`.
Gold-only SFT @ lr2e-4/1ep made the model pattern-match memorized "normal-looking"
column names instead of reading the actual schema in the prompt = mild overfit to
training surface patterns. token-acc was 0.974 (tight gold fit) — overfit signal.
IMPLICATIONS:
- This is exactly what GRPO's execution reward fixes (punishes columns/tables that
  don't run). Predicts GRPO-only (from strong base) may BEAT SFT->GRPO (from degraded
  SFT) — makes the recipe axis a real experiment, not a formality.
- SFT-recipe fix for later: lower LR (~2e-5), fewer steps, or --sample-rows so model
  sees real column names during training. (Held in reserve.)
DECISION: proceed to both GRPO runs (SFT->GRPO + GRPO-only, --subset 2000); GRPO is
the antidote to the schema hallucination.

### GRPO-only run #1 LOST to OOM at 81% (2026-06-28) — lessons
- coder7b_grpo_only (subset1000, max-prompt-tokens 2048, max-completion 512, num_gen4)
  ran 3h, reached step 772/956 (81%) then OOM in the BACKWARD pass (autograd). It sat
  at ~15.8GB the whole time (2048+512 seq too close to 16GB edge); a heavy backward
  tipped it.
- WORSE: no checkpoint salvageable. GRPOConfig had save_steps=200 but NOT
  save_strategy="steps" → transformers default ignored it → 0 checkpoints. Whole run lost.
- FIXES APPLIED to train/grpo.py (committed, NOT yet rerun):
  (1) save_strategy="steps", save_steps=150, save_total_limit=2  → OOM no longer loses
      everything; can resume via --init-adapter outputs/<dir>/checkpoint-XXX.
  (2) for relaunch use safer mem: --max-prompt-tokens 1536 (keep 94.2%) +
      --max-completion 320 (SQL rarely >250 tok) → seq ~1856 vs 2560, real headroom.
- STATUS: GRPO-only NOT done. PARKED at user request — relaunch later with:
  grpo.py --model unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit \
    --train-root data/train_extracted/train --subset 1000 --max-prompt-tokens 1536 \
    --num-generations 4 --batch-size 4 --max-completion 320 --temperature 1.0 \
    --out outputs/coder7b_grpo_only
  (env: LD_LIBRARY_PATH cu13 + PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True)

### Session 5 (2026-06-29)
- GRPO-only RELAUNCHED with fixed config (checkpointing + tighter mem). Running:
  subset 1000 → prompt-len filter (<=1536 tok) kept 942/1000; LoRA 40.4M params
  (0.53%); GPU ~9.7GB at start (vs 15.8GB edge last time → real headroom now);
  942 steps; checkpoints every 150 (save_total_limit=2) so OOM won't wipe it again.
  out=outputs/coder7b_grpo_only, log=outputs/grpo_only.log.

### TODO: rerun SFT later (planned, NOT done) ⭐
WHY: gold-only SFT @ lr2e-4 HURT the base (-1.4 EX, -3.3 valid) via schema-
hallucination / surface overfit (token-acc 0.974). Want a non-degrading SFT so
the SFT->GRPO recipe arm starts from a healthy adapter, not a damaged one.
PLAN for the rerun:
  - lower LR ~2e-5 (was 2e-4), fewer epochs/steps (was 1ep already — try <1ep / early stop)
  - checkpoint every N steps, EVAL EACH BY EX (not loss), ship best-EX checkpoint
    (do NOT keep last checkpoint = most overfit; do NOT trust load_best_model_at_end,
     it selects by eval loss ≠ best EX)
  - consider --sample-rows so model sees real column names during training
  - re-eval on BIRD dev; target: SFT >= base (51.2) so the warm-start actually helps
STATUS: held in reserve; revisit after GRPO-only + SFT->GRPO results are in.

### GRPO-only COLLAPSE finding (2026-06-29) ⚠️ important
Run #1 (Jun 28) DID save checkpoints (save_steps=200 worked after all). Telemetry
shows reward COLLAPSE, not just the OOM:
  step 200: reward 0.55, reward_std 0.52, entropy 0.090, comp_len 77  <- healthy
  step 400: reward 0.10, reward_std 0.0,  frac_zero_std 1.0, comp_len 53 <- collapsed
  step 600: reward 0.10, reward_std 0.0,  frac_zero_std 1.0, comp_len 22 <- dead
DIAGNOSIS: policy reward-hacked to the 0.1 "runs-but-wrong" floor — emits a short
(~22 tok) always-executing-but-wrong query; all 4 rollouts identical → reward_std=0
→ grad_norm=0 → zero learning for last ~60% of run. entropy crashed 0.09->0.045.
LIKELY CAUSES: (a) 0.1 execution floor rewards trivial always-runs queries;
(b) beta=0 (no KL anchor) lets policy drift to a degenerate mode with no pull back;
(c) lr very low (8e-7) so it can't escape once collapsed.
BEST SURVIVING CKPT: outputs/coder7b_grpo_only/checkpoint-200 (pre-collapse, r=0.55).
Today's relaunch OOM'd at step 119/942 even at 1536/320 — still on the 16GB edge.
NEXT (decide): eval checkpoint-200 on BIRD to see if pre-collapse GRPO > base 51.2;
fix collapse for next run (drop 0.1 floor OR add small KL beta~0.02-0.04 OR raise lr);
fix mem (max-prompt-tokens 1024 + max-completion 256, or move to Coder-3B base).

### EVAL vs TRAIN: long-prompt handling (do NOT confuse the two)
The --max-prompt-tokens filter (drop monster-schema prompts) is TRAINING-ONLY,
for GRPO OOM avoidance (4 rollouts × full-vocab entropy × backward = heavy).
In EVAL we must NEVER drop test examples — dropping changes the denominator and
makes EX non-comparable to official BIRD (you'd be grading an easier subset).
HOW EVAL HANDLES LONG PROMPTS: truncate, not drop —
  tokenizer(prompts, truncation=True, max_length=MAX_SEQ_LEN - MAX_NEW_TOK)
A too-long schema gets cut; if the model then misses a table it just scores wrong
on that example (honest — reflects a real context-window limit), full denominator
intact. Applies to BOTH eval/run_eval.py (HFPredictor) AND bird_dev_eval.ipynb.
WHY EVAL DOESN'T OOM without the filter: inference = forward activations only (no
backward, no rollouts, no full-vocab entropy, no optimizer states) → ckpt-200 eval
runs ~10GB vs GRPO training ~15.8GB on the same 16GB card.
KNOB: bird_dev_eval.ipynb uses MAX_SEQ_LEN=4096; a few BIRD schemas exceed it and
get truncated. Raise MAX_SEQ_LEN (+ lower BATCH_SIZE to 2) to avoid truncation at
a memory cost; 4096/batch-8 is the safe comparable default.

### RESULT: GRPO ckpt-200 on BIRD dev (2026-06-29)
EX 0.5104 (vs base 0.512, SFT 0.498); valid_sql 0.9009; n=1534, gold_runnable=1532.
EX_by_difficulty: simple 0.5827 / moderate 0.4104 / challenging 0.3681.
gen 2868s (~48min HF-generate), exec 39s. results/coder7b_grpo200_bird.jsonl.
READ (per agreed framing): FLAT vs base = INCONCLUSIVE, NOT a condemnation of GRPO.
200 collapse-bound steps simply didn't move EX. Notably GRPO did NOT hurt like SFT
did (SFT dropped EX + valid; GRPO held base EX) → execution reward isn't degrading
schema-grounding, consistent with "GRPO is the right tool, we ran a broken version."
VERDICT STILL OPEN. A real GRPO test needs: (1) fixed reward (drop 0.1 runs-but-
wrong floor and/or add small KL beta~0.02-0.04 to stop collapse) + (2) enough
NON-collapsed steps. Then re-eval. Do NOT judge GRPO from ckpt-200.

### WEB SEARCH: how good researchers design GRPO rewards for NL2SQL (2026)
Our collapse is a KNOWN, documented failure mode — not a one-off mistake.
Key papers (single-turn GRPO, one-shot reward unless noted):
- Reasoning-SQL (arxiv 2503.23157): composite reward = execution + schema-linking
  + n-gram-similarity-to-gold + syntax-check, under GRPO, to fix execution-reward
  SPARSITY. EXPLICITLY reports reward hacking caused by its "Relaxed Exact Match"
  partial-credit term — i.e. exactly our 0.1 floor problem.
- Reward-SQL (2505.04671): stepwise / process-supervised rewards.
- Graph-Reward-SQL (2505.12380): EXECUTION-FREE reward via graph matching (cheap,
  no DB run) — escape hatch if execution cost dominates on 16GB.
- Think2SQL (2504.15077), Reinforcing Code Gen / execution-based (2506.06093),
  Progress-SQL (2606.06825): reasoning + execution-based + progressive rewards.
Cross-paper anti-hacking tools: length penalty, format/tag regularization, KL.

THE CORE TENSION: pure exec-accuracy reward is CORRECT but SPARSE (all-wrong group
=> reward_std=0 => no gradient = our collapse). Partial rewards relieve sparsity
but naive ones are HACKABLE (our 0.1 "executes" floor). Good design = partials that
CORRELATE with correctness and CAN'T be trivially maxed.

HACKABLE (what we had)        vs   SOLID (what researchers adopt)
- +0.1 for "executes"              - partials tied to RESULT-SET OVERLAP (Jaccard of
  (trivial always-running            returned rows): trivial/empty query scores ~0
  query farms it) -> COLLAPSE      - schema-linking: fraction of GOLD tables/cols used
- beta=0 (no KL leash) ->            (can't max without ~writing the right query)
  free drift to degenerate mode    - n-gram/component similarity to gold SQL
- no length control ->             - syntax/parse valid = TINY only
  22-token degenerate collapse     - length penalty (kills degenerate-short + verbose)
                                   - partials CAPPED << 1.0 so CORRECT (1.0) dominates
                                   - small KL (beta~0.02-0.04) = leash vs collapse
                                   - temp>=1.0, num_gen>=4, drop zero-std groups

### IMPLEMENTED composite reward + KL anchor (2026-06-29)
train/sql_reward.py: added composite_reward (default) — exact-match 1.0 dominates;
partials CAPPED at 0.30 = W_OVERLAP*row_jaccard(0.20) + W_SCHEMA*gold_id_coverage(0.10)
+ W_SYNTAX(0.05 if executes) - LEN_PENALTY(0.10 if <5 tokens). execution_reward kept
as legacy ref (DO NOT USE). train/grpo.py: --reward {composite,execution} default
composite; --beta default 0.03 (KL anchor; with LoRA, TRL toggles adapter off for ref
= ~no extra VRAM). Helpers unit-checked (schema_link 1.0/0.0, jaccard 1.0/0.33/0.0).
LAUNCH (proper GRPO v2, run #2):
  grpo.py --model unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit \
    --train-root data/train_extracted/train --subset 1000 --max-prompt-tokens 1024 \
    --num-generations 4 --batch-size 4 --max-completion 256 --temperature 1.0 \
    --beta 0.03 --reward composite --out outputs/coder7b_grpo_v2
  (env: LD_LIBRARY_PATH cu13 + PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True)

### GRPO v2 (composite reward) — early read + relaunch as v3 (2026-06-29)
v2 (temp 1.0) ran to step ~114/887 then STOPPED for relaunch. KEY OBSERVATIONS:
- COLLAPSE FIXED: composite reward + beta=0.03 KL → NO degenerate absorbing state
  (run #1's permanent r=0.1/len=22 trap is gone). reward bounces 1.0(solved)<->0.1
  (failed), entropy not crashing to 0, kl tiny (~5e-5), no OOM. Reward fix WORKS.
- NEW ISSUE (milder): SATURATED GROUPS / low rollout diversity. Most steps
  reward_std=0, frac_reward_zero_std=1, all 4 completions identical length → zero
  gradient. Only ~30% of steps (std 0.42-0.48) teach. Cause: low entropy (0.02-0.2),
  confident instruct model → temp=1.0 samples barely differ. Also ≤1024-tok filter
  keeps EASY short-schema prompts the base already aces (the r=1.0 zero-std groups).
- GPU 14.4GB (KL ref forward via adapter-toggle pushed it up from 10.4) — watch edge.
FIX → v3: temperature 1.0 -> 1.3 (more rollout diversity, no extra VRAM). Everything
else identical. out=outputs/coder7b_grpo_v3.
FUTURE LEVERS if v3 still saturates: num_generations up (costs VRAM), allow longer/
harder prompts (OOM risk), or higher temp still.

## RESUME HERE — 2026-06-29 (pre-compact state)
RUNNING NOW: GRPO v3 (background nohup, NOT harness-tracked). Relaunch/inspect cmd:
  CU13=$(.venv/bin/python -c "import os,nvidia;print(os.path.dirname(nvidia.__file__))")/cu13/lib
  export LD_LIBRARY_PATH="$CU13:$LD_LIBRARY_PATH" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
  .venv/bin/python train/grpo.py --model unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit \
    --train-root data/train_extracted/train --subset 1000 --max-prompt-tokens 1024 \
    --num-generations 4 --batch-size 4 --max-completion 256 --temperature 1.3 \
    --beta 0.03 --reward composite --out outputs/coder7b_grpo_v3
  log: outputs/grpo_v3.log  | 887 steps | ckpts every 150.
  CHECK: does temp 1.3 reduce saturated groups? want frac_reward_zero_std <1 more
  often (more std>0 groups). grep trajectory:
    grep -aoE "\{'loss':[^}]*\}" outputs/grpo_v3.log | tail -25
  THEN: eval the final/best ckpt on BIRD vs base 51.2:
    .venv/bin/python eval/run_eval.py --dataset bird --root data/dev_20240627 \
      --backend hf --model unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit \
      --adapter outputs/coder7b_grpo_v3 --batch-size 8 --tag coder7b_grpo_v3 \
      --out results/coder7b_grpo_v3_bird.jsonl   (HF-gen ~48min over 1534)

KAGGLE (ready, for parallelizing grid 3B/4B cells): kaggle CLI 2.2.3 in .venv;
auth = KGAT token in ~/.kaggle/api_token.env (mode 600, env var KAGGLE_API_TOKEN,
NOT old kaggle.json key field). `source ~/.kaggle/api_token.env` before kaggle calls.
Account = zineelabidine52. ⚠️ ROTATE this token (shared in chat). Plan: package
~/nl2sql + BIRD-train as a Kaggle dataset, `kaggle kernels push` a 3B GRPO/eval
kernel (T4 16GB, same mem ceiling), pull via `kaggle kernels output`. NOT started.

DELIVERABLE for intern: bird_dev_eval.ipynb — real BIRD-dev EX eval (real populated
DBs, real schema from sqlite_master, official set() EX, truncate-not-drop long
prompts). He sets ADAPTER_PATH to his Llama-3.1-8B LoRA. Sent feedback = 9 points,
serious ones are eval problems (empty tables, non-standard test, exact-dedup leakage).
Told him to use Spider (Kaggle-friendly, real populated DBs) if BIRD too heavy.

SCOREBOARD (BIRD dev EX): SLM-SQL-1.5B 43.3 | Arctic-7B(FP8) 62.1 | Coder-7B base
51.2 | Coder-7B SFT 49.8(hurt) | GRPO ckpt-200(broken reward) 51.04 | GRPO v3 PENDING.

### KAGGLE OFFLOAD live (2026-06-29)
ACCELERATORS confirmed: Kaggle = T4 x2 (two 16GB, 32GB AGGREGATE not one card -
need device_map=auto/FSDP to use both) or P100 16GB; ~30h/wk. T4/P100 = NO bf16
(Turing/Pascal) -> grpo.py now AUTO-detects (torch.cuda.is_bf16_supported) and
falls back fp16. Edit is a no-op on local Blackwell (still bf16).
PUBLIC BIRD datasets (no upload needed): jenishk/bird-dev (dev_20240627/ tree,
matches our layout) + himanshusinghal19/bird-dataset (train/train.json + nested
train_databases/train_databases/<db>/<db>.sqlite; our recursive-glob loader handles it).
OUR CODE dataset: zineelabidine52/nl2sql-code (train.zip+eval.zip; kernel unzips).
KERNEL (smoke, RUNNING): zineelabidine52/nl2sql-3b-smoke — script kernel, GPU+net on,
mounts the 3 datasets, unzips code, pip installs trl/peft/bitsandbytes/accelerate/
datasets/sqlparse, runs Coder-3B GRPO (subset 40, composite, temp1.3, fp16) then
bird-dev eval (limit 100). Validates the whole pipeline cheaply (~25min) before any
long run. CHECK/PULL:
  source ~/.kaggle/api_token.env
  .venv/bin/kaggle kernels status zineelabidine52/nl2sql-3b-smoke
  .venv/bin/kaggle kernels output zineelabidine52/nl2sql-3b-smoke -p <dir>
COMPAT RISK to watch: Kaggle's trl version vs our GRPOConfig API (use_vllm/beta/
max_completion_length). If smoke errors on GRPOConfig kwargs, pin trl in the kernel.
NEXT after green smoke: full Coder-3B GRPO (subset 1000) + Coder-4B, in parallel
with local 7B; later 14B QLoRA via device_map=auto across 2xT4 (the thing local
16GB CAN'T do).

### KAGGLE smoke v1 ERROR + fix (2026-06-29)
v1 errored immediately: FileNotFoundError /kaggle/input/nl2sql-code/train.zip.
CAUSE: Kaggle AUTO-EXTRACTS dataset zips on mount -> code lands at
/kaggle/input/nl2sql-code/{train,eval}/ (raw dirs), NOT train.zip.
FIX (v2 pushed): kernel now shutil.copytree('/kaggle/input/nl2sql-code',
'/kaggle/working/nl2sql', dirs_exist_ok=True) instead of unzip. trl-API compat
still UNVERIFIED (v1 died before reaching grpo.py). Re-check status/output:
  source ~/.kaggle/api_token.env
  .venv/bin/kaggle kernels status zineelabidine52/nl2sql-3b-smoke
  .venv/bin/kaggle kernels output zineelabidine52/nl2sql-3b-smoke -p <dir>

### Kaggle smoke kernel v3 (2026-06-29) — mount fix
v2 errored at 0.77s: `FileNotFoundError: /kaggle/input/nl2sql-code` — the dataset
mounted under a different/absent path even though it exists & is `ready` (verified
via `kaggle datasets files zineelabidine52/nl2sql-code`). v3 fix: stop hardcoding
the mount path; glob `/kaggle/input/*/train/grpo.py` to locate the code dataset,
print `/kaggle/input/*` for diagnostics, raise a clear error if not mounted.
Pushed as kernel v3. trl-API compat on T4 STILL UNVERIFIED (every version so far
died before reaching grpo.py). Re-check: `kaggle kernels status zineelabidine52/nl2sql-3b-smoke`.

### Kaggle smoke v4/v5 (2026-06-29) — both blockers solved, NO upload
v4 finally reached grpo.py (recursive glob fixed the mount path: Kaggle nests at
/kaggle/input/datasets/<owner>/<name>/...). Two real blockers found there:
1. GPU LOTTERY: API push gave a P100 (sm_60); Kaggle's torch supports sm_70+ only
   -> torch unusable. FIX: `kaggle kernels push --accelerator NvidiaTeslaT4`
   (CLI 2.2.3 supports --accelerator; valid IDs incl NvidiaTeslaT4/Highmem, P100,
   A100, L4, H100). kernel-metadata.json canNOT set GPU type; must use the flag.
2. NO TRAIN DATA UPLOAD NEEDED: our nl2sql-code dataset is code-only (14 files).
   BIRD train already exists PUBLIC in himanshusinghal19/bird-dataset (mounted):
   train/train.json + train/train_databases/train_databases/<db>/<db>.sqlite
   (double-nested). data.py _db_dir globs recursively + validates */*.sqlite, so it
   lands on the inner dir correctly — no code change. Fix was only the kernel:
   resolve TRAIN_ROOT/DEV_ROOT by their data file (train.json/dev.json) not dirname.
v5 pushed with T4. First run that should actually train -> real trl-on-T4 compat test.

### Kaggle smoke v6/v7 (2026-06-29) — bitsandbytes pin
v5 (latest bnb) crashed step 0: illegal memory access ops.cu:81 / "CUDA driver
error: unknown error" -> SIGABRT, on the T4 4-bit kernel. v6 tried DROPPING bnb
(thought Kaggle shipped one) -> WRONG: Kaggle ships NO bitsandbytes
(ModuleNotFoundError; transformers needs >=0.46.1). v7 fix: pin
`bitsandbytes==0.46.1` (transformers floor, stable sm_75 kernels) + keep trl/peft/
sqlparse + version-probe print. NOTE: model load + LoRA attach + trainer init all
already proven to work on T4 in v5 — only the bnb 4-bit compute kernel was the blocker.

### Kaggle smoke v8 (2026-06-29) — DROP 4-bit on T4 (root fix)
v7 (bnb pinned 0.46.1) STILL crashed: cudaErrorIllegalAddress during generation
(ran 3min, step 0). Confirmed stack: torch 2.10.0+cu128, bnb 0.46.1, T4 sm_75 (7,5).
=> bnb's 4-bit kernel is unstable on this T4+torch regardless of version. ROOT FIX:
drop 4-bit entirely — a 3B in fp16 is ~6GB, fits 16GB T4 with LoRA easily (4-bit was
only needed for the local 7B on 16GB Blackwell). Added BACKWARD-COMPAT `--no-4bit`
flag to train/grpo.py (fp16/bf16 load + gradient_checkpointing_enable +
enable_input_require_grads) AND eval/run_eval.py+predictors.py (fp16 load, bf16 if
supported else fp16). 4-bit stays DEFAULT so local pipeline unchanged. Kernel now uses
NON-quantized base `Qwen/Qwen2.5-Coder-3B-Instruct` + --no-4bit for both train & eval.
Re-uploaded nl2sql-code dataset (v with no-4bit, verified live) + pushed kernel v8.

### Kaggle smoke v9 (2026-06-29) — single-T4 (DataParallel OOM)
v8 (fp16 --no-4bit) KILLED the bnb crash (real progress) but OOM'd at step 0 in
torch/nn/parallel/comm.py reduce_add = HF Trainer auto-wrapped nn.DataParallel
across BOTH T4s -> model replicated, single 16GB card OOM (14.18GB alloc). v9 fix
(kernel-only, no dataset re-upload): CUDA_VISIBLE_DEVICES=0 (one T4) +
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True for both train & eval subprocesses.
3B fp16 + LoRA fits one T4. (Proper T4x2 later needs accelerate/FSDP, not naive DP.)

### TODO (perf, user-flagged 2026-06-29): make vLLM eval work — time is valuable
Keep HF generate for now (robust, zero-setup, works with live LoRA adapter, fine for
smoke). But the REAL BIRD-dev eval (1534 ex) on HF generate is ~1h on a single T4;
vLLM = ~5-20x faster. ACTION when we do real runs:
- Eval: switch run_eval to `--backend vllm` (already implemented in predictors.py).
  Needs a merged fp16/bf16 model (merge LoRA->16bit first) + reserved VRAM (gpu-mem-frac).
  On T4 (sm_75) vLLM should be fine; the Blackwell vLLM bugs (BLACKWELL_BUGS #5/#6,
  FlashInfer sampler) are LOCAL-only — re-verify env exports there.
- GRPO rollouts: TRL use_vllm=True is the fast path but BROKEN on local Blackwell
  (.ref.weight crash). UNTESTED on T4 — worth trying for real Kaggle GRPO runs.
- Alternatives to vLLM if it fights us: SGLang (good w/ shared schema prefixes), TGI.

### Websearch (2026-06-29): Unsloth/vLLM GRPO bugs — root pattern + fast-path plan
Searched bugs #5(.ref.weight)/#6(Half vs Float). Our exact error strings aren't
documented (stack-specific), BUT the broader pattern IS well-known:
- unslothai/unsloth #1930: QLoRA-4bit GRPO + fast_inference=True -> '...down_proj.
  weight.absmax' crash (UNRESOLVED); vLLM loads 16bit even w/ load_in_4bit=True.
- #1694/#2009: Unsloth LoRA adapters silently ignored/mis-loaded in vLLM.
- vLLM LoRA only adapts the 7 projection layers; Unsloth defaults sometimes add
  embed_tokens/lm_head -> rejected.
=> Our two bugs = same systemic issue: Unsloth's live-LoRA->vLLM bridge is fragile,
   4-bit makes it worse. Confirms dropping Unsloth was correct (not bad luck).
CONSISTENT WORKAROUND across threads (aligns w/ where we are):
  1. GRPO+vLLM in 16-bit, NOT 4-bit (4-bit weight-load = crash epicenter). Kaggle
     already fp16 via --no-4bit (bnb reason) = coincidentally most vLLM-friendly.
  2. Feed vLLM a MERGED 16bit model, not a live adapter.
  3. Restrict target_modules to the 7 proj layers (our grpo.py already does this).
FAST-PATH PLAN (the "time is valuable" win):
  - KAGGLE T4 (no Unsloth): try TRL-native vLLM rollouts use_vllm=True in GRPOConfig
    -> none of the Unsloth bugs apply there. Cleanest shot. TEST after smoke green.
  - LOCAL Blackwell: stay plain TRL+PEFT; upgrade = 16bit GRPO + TRL vLLM rollouts
    (triton>=3.3.1, sm_120 vLLM caveats vllm#37242). NOT Unsloth QLoRA+vLLM.

### SESSION CLOSE (2026-06-29)
TWO things finished green:
1. KAGGLE SMOKE v9 = COMPLETE / end-to-end GREEN. Pipeline validated on T4:
   mount data -> load Qwen2.5-Coder-3B fp16 -> LoRA -> GRPO 40-ex (composite reward)
   -> eval 100 BIRD-dev. Summary: EX 0.12, valid_sql 0.52, gen 1099s (~18min, HF
   generate = slow, confirms vLLM TODO). NUMBERS ARE THROWAWAY (40-ex GRPO degrades
   the model; smoke = plumbing proof only). Final kernel config that works on Kaggle:
   --accelerator NvidiaTeslaT4, fp16 NON-quant base + --no-4bit (bnb 4-bit kernel
   crashes T4), CUDA_VISIBLE_DEVICES=0 (avoid DataParallel OOM across 2xT4),
   TRAIN_ROOT=himanshusinghal19/bird-dataset (BIRD train already public, no upload).
2. LOCAL 7B v3 GRPO = DONE, 887/887, adapter saved -> outputs/coder7b_grpo_v3
   (composite reward + beta 0.03 + temp 1.3; collapse fixed, no OOM).

>>> RESUME HERE NEXT SESSION (the real results, not done yet) <<<
A. Eval local v3 on full BIRD dev vs base 51.2 / SFT 49.8 / ckpt200 51.04:
   cd ~/nl2sql && <env exports from BLACKWELL_BUGS S0> && .venv/bin/python eval/run_eval.py \
     --dataset bird --root data/dev_extracted/dev_20240627 --backend hf \
     --model unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit --adapter outputs/coder7b_grpo_v3 \
     --tag coder7b_grpo_v3 --out outputs/coder7b_grpo_v3_bird.jsonl
B. Kaggle real run: flip kernel to --subset 1000 (real 3B GRPO ~956 steps) now smoke green.
C. vLLM fast-path (user priority "time valuable"): TRL use_vllm=True on Kaggle T4 (no
   Unsloth bugs there). See websearch entry above.
D. Code edits this session (backward-compat, 4-bit still default): train/grpo.py +
   eval/run_eval.py + eval/predictors.py gained --no-4bit. nl2sql-code dataset re-uploaded.

### Base 3B baseline eval launched (2026-06-29)
GAP found: Coder-3B base was NEVER evaluated (only the throwaway 40-ex smoke adapter
@ EX 0.12). Launched zero-shot base eval on Kaggle to anchor Arm-3 3B (like base-51.2
anchors 7B). Kernel: zineelabidine52/nl2sql-base3b-eval (separate from smoke). Config:
Qwen/Qwen2.5-Coder-3B-Instruct, fp16 --no-4bit, single T4, FULL BIRD dev (no --limit),
NO adapter. Out -> coder3b_base_bird.jsonl. ETA ~30-45min.

================================================================================
## RESULTS + DIAGNOSIS: why our finetuning didn't improve EX (2026-06-29)
================================================================================

### EVAL RESULTS (full BIRD dev, 1534 ex, HF generate, official set() EX)
| model                          | EX     | valid_sql | notes                         |
|--------------------------------|--------|-----------|-------------------------------|
| base Qwen2.5-Coder-7B (4bit)   | 0.512  | -         | reference                     |
| SFT (gold-only, lr2e-4)        | 0.498  | -3.3pt    | HURT (overfit/halluc)         |
| GRPO ckpt-200                  | 0.5104 | -         | flat                          |
| GRPO v3 FULL (887 steps)       | 0.5091 | 0.899     | FLAT (simple .583/mod .410/chal .354) |
=> SFT hurt; GRPO neither helped nor hurt. Collapse was FIXED (valid 90%, no 0.1-floor)
   but produced NO EX gain.

### ROOT CAUSE: we starved the trainer (not a bug, a recipe gap)
1. SATURATED GROUPS / zero gradient: --max-prompt-tokens 1024 dropped the HARD
   long-schema prompts (the only room to improve) -> trained on EASY prompts the base
   already aces -> all rollouts correct -> reward_std=0 -> ~no gradient. lr1e-6 x
   near-zero grad x ~950 steps = near no-op. 50.91≈51.2 = barely updated, not "GRPO bad".
2. NO VALUES in prompt (sample_rows=0): schema DDL only, model never sees column values;
   huge share of BIRD errors are value/literal linking.
3. max_completion 256 CAPS correct answers: gold SQL >256 tok can never be rewarded.
4. SFT hurt: gold-only CE @ lr2e-4 overfit surface form -> schema hallucination.

### LITERATURE VALIDATION (websearch 2026-06-29)
- ExCoT (2503.19988): gold SFT barely moves a strong base (32B 58.93->59.65 +0.7).
  ALL gains from CONTRASTIVE exec-feedback DPO: SFT->offpolicy DPO 66.23->onpolicy 68.25.
  Sample up to 32 cands/query, execute, label by EX, pair correct-vs-incorrect (offpolicy
  =max edit dist, onpolicy=min edit dist near-miss). "Incorrect solutions are essential."
- Reasoning-SQL (2503.23157): 7B GRPO +6.77% vs SFT +4.11%, BUT 8026 filtered ex, 3
  epochs, 6 rollouts, SCHEMA LINKING (Gemini filters tables), reward=exec+LLM-judge+
  schema+ngram. LR 1e-6 SAME as ours -> differentiator is DATA(8x)+epochs(3x)+schema-
  link+richer reward, NOT lr. Reward sparsity = the explicit core problem of the field.
- Generic SFT on BIRD overfits, poor out-of-domain (Qwen Coder 43.8/31.4 vs GPT4 59/43).

### IMPLICATIONS / NEXT EXPERIMENTS (priority)
A. SWITCH to ExCoT-style exec-feedback DPO (most reproducible gain on our exact base;
   sidesteps GRPO saturation by construction). Reuse our executor/EX harness to label
   N sampled candidates, build correct/incorrect pairs. <-- TOP PICK
B. SCHEMA LINKING (reduce schema to relevant tables) — non-optional; every system that
   beats base does it. Our full-DDL dump wastes ctx + starves signal.
C. If staying on GRPO: ~8k ex, 3 epochs, num_gen 6, KEEP hard prompts (schema-link to
   fit, don't drop), max_completion >=512, add sample_rows (values).
D. Inference-time: self-consistency / candidate selection (sample N, pick by exec
   agreement) — reliable +EX, how SLM-SQL/DeepEye get headline numbers. We only ever
   measured 1 greedy pass.
DEEPER LIT SEARCH: in progress (see next entry).

### DEEPER LIT SEARCH (2026-06-29) — the decisive findings
THE KEY FINDING — Arctic-Text2SQL-R1-7B (2505.20315) = OUR EXACT base family + GRPO +
a SIMPLER reward than ours (pure exec 1.0/0.1/0.0 — the very reward we called "hackable")
=> 68.9 BIRD dev. So reward shape was NOT our core problem. The differences:
  1. STRONG INIT: they start from OmniSQL-7B (pretrained on 2.5M SQL), NOT vanilla
     Qwen2.5-Coder-7B-Instruct. Paper: "strong initialization prevents the sparse-reward
     problem that defeats naive GRPO" = EXACTLY our saturated/zero-grad failure.
  2. CURATED DATA ~28k: 8k BIRD + 8k Spider + 12k synth, MODEL-FILTERED (keep only
     queries where >=1 of 10 gens correct -> removes impossible, keeps learnable), drop
     empty-gold + >5s queries. We had ~942 uncurated easy-filtered.
  3. 16 rollouts/group (we: 4), UNCONSTRAINED completion (we capped 256), beta 0.001
     (we: 0.03), temp 0.8.
  => Our composite-reward pivot treated a SYMPTOM. Disease = weak init + tiny data +
     few rollouts + completion cap. Arctic avoids collapse via init+16roll+tiny-KL, and
     argues SIMPLE reward is BETTER (partial rewards invite hacking). Reconsider our
     composite reward vs reverting to simple exec.

DATA SCALE = dominant lever: OmniSQL (2503.02240) + SynSQL-2.5M (2.5M synth, open HF
seeklhy/SynSQL-2.5M) -> +9.0 EX BIRD from data alone. OmniSQL-7B/14B/32B open weights.

TEST-TIME SCALING = free +4-8 EX, orthogonal to training (WE ONLY DID 1 GREEDY PASS):
  - Self-consistency: sample N @ temp0.8, execute all, cluster by result, pick largest.
  - CSC-SQL (2505.13271): + correction step when top-2 clusters disagree -> 68.70@n=64
    vs 64.45 single-pass (+4.25 from correction); 7B -> 69.19 BIRD. GRPO-trained gen+rev.
  - GradeSQL (2509.01308): outcome reward model > majority vote by +4.33, keeps scaling.
SCHEMA LINKING standard at scale: reduce schema to relevant tables/cols + retrieve
  VALUES (LSH/semantic) pre-gen (2510.14296 Rethinking SL, 2503.18596 LinkAlign,
  Amazon RASL). Our full-DDL dump + sample_rows=0 is the opposite.

OPEN STRONG 7Bs ALREADY AT TARGET (= our Arm 2 adopt): Arctic-Text2SQL-R1-7B 68.9,
  CSC-SQL-7B 69.19, OmniSQL-7B. (XiYanSQL-QwenCoder-7B also strong.)

THREE PATHS (recommended order):
  1. ADOPT a strong open 7B (low effort, 68-69) — Arm 2 already.
  2. TEST-TIME self-consistency on current models (low effort, +4-8, stacks w/ anything).
  3. GRPO DONE RIGHT only if bespoke model wanted: start from OmniSQL-7B, curate ~10-30k
     model-filtered, 16 rollouts, no completion cap, beta 0.001, SIMPLE exec reward.
BLUNT TAKEAWAY: training from vanilla instruct on ~1k ex could never beat a 51% base;
  field reaches 68-69 via 2.5M-ex init + curated RL data + 16 rollouts + test-time sampling.
Sources: arxiv 2505.20315, 2503.02240, 2505.13271, 2509.01308, 2510.14296, 2503.18596,
  2503.19988 (ExCoT), 2503.23157 (Reasoning-SQL).

================================================================================
SESSION 6 — BROADER LITERATURE SWEEP (general fine-tuning / RL, not NL2SQL-only)
  (user: "broader search, more papers, wider scope, general finetuning and RL")
  Date 2026-06-29. Goal: explain the FLAT EX from first principles, beyond SQL.
================================================================================

THE 4 GENERAL FINDINGS THAT EXPLAIN OUR FLAT EX
-----------------------------------------------
1) RLVR SHARPENS, IT DOESN'T CREATE CAPABILITY (the big one).
   "Limit of RLVR" (limit-of-rlvr.github.io) + multiple 2026 follow-ups:
   RL-trained models beat the base at pass@1 but the BASE BEATS THEM at pass@k
   for k in the dozens-hundreds. Every correct RL solution already existed in
   the base's sampling distribution. GRPO is a CONSERVATIVE REWEIGHTING bounded
   by base support; it cannot invent reasoning the base can't already sample.
   => Implication for us: GRPO on Qwen2.5-Coder-7B (base EX 51) can at best
      concentrate mass on SQL the base already gets sometimes. If the base
      *never* samples a correct query for the hard BIRD items, no amount of our
      GRPO surfaces it. The ceiling is the base. This is WHY the strong arms
      swap the base (OmniSQL-7B / 2.5M-SQL pretrain) instead of just doing RL.
   Nuance / counter-camp: "Rewarding the Unlikely: Lifting GRPO Beyond
      Distribution Sharpening" + prolonged-RL-with-KL-control papers argue
      careful RL CAN widen support — but it needs explicit unlikeliness rewards,
      long schedules, tight KL. Not what our vanilla run did.

2) ADVANTAGE / GRPO COLLAPSE IS A NAMED, MEASURABLE FAILURE — and we had it.
   arxiv 2605.21125 "Advantage Collapse in GRPO" (ICML 2026) + AVSPO:
   homogeneous within-group rewards (all-right or all-wrong rollouts) -> ~0
   advantage -> vanishing gradient. They define ACR (Advantage Collapse Rate)
   = fraction of batches with dead gradients; ACR PREDICTS stagnation & final
   score across 0.5B-14B. Fix = inject virtual reward samples into the
   normalization stats (58-63% collapse reduction). Also EDGE-GRPO / EP-GRPO
   (entropy-driven), Transformation-Augmented GRPO (exploration).
   => This is EXACTLY our diagnosed "saturated/zero-gradient groups" from
      filtering hard prompts + only 4 rollouts. We were starving the gradient.
      ACTIONABLE: measure ACR (log reward_std per group); raise num_generations
      (16, not 4); KEEP hard prompts; consider virtual-sample / entropy fix.

3) SFT MEMORIZES, RL GENERALIZES — BUT SFT IS NEEDED FOR FORMAT/STABILITY.
   arxiv 2501.17161 (ICML 2025, the canonical one): outcome-reward RL
   generalizes OOD; SFT memorizes & overfits — BUT "SFT is helpful for
   effective RL training in stabilizing output format." And 2509.12235 "RL
   Fine-Tuning Heals OOD Forgetting in SFT."
   => Our SFT (49.8, BELOW base 51.2) is the textbook SFT-overfit/forgetting
      result. The correct role of SFT here is a SHORT format/cold-start warmup,
      then RL — not a quality lever on its own. Don't chase SFT EX.

4) LoRA IS NOT THE BOTTLENECK FOR RL (kills one of our hypotheses).
   Thinking Machines "LoRA Without Regret" + arxiv 2410.21228:
   - RL needs almost NO capacity: rank-1 LoRA MATCHES full FT on math RL
     (policy gradient absorbs ~1 bit/episode; rank-1 8B already has 3M params >>
     the ~320k bits needed for 10k problems x 32 samples). So our QLoRA rank was
     NEVER the limiter for the GRPO step.
   - BUT two conditions we should verify we meet:
       (a) APPLY LoRA TO ALL LAYERS incl. MLP/MoE — attention-only badly
           underperforms (attn rank256 < mlp rank128). CHECK our target_modules.
       (b) LoRA optimal LR = ~10x full-FT LR (≈15x for <100-step runs). If we
           used a full-FT-scale LR, we under-trained. CHECK our grpo.py LR.
   - SFT side: LoRA == full FT only on small/medium data & if not capacity-
     capped; high rank needed for big SFT corpora. LoRA also tolerates large
     batch sizes worse — keep effective batch modest.

5) DATA: QUALITY >> QUANTITY for RLVR, and curation differs from SFT.
   2026 curation papers: small curated high-quality sets beat large noisy ones;
   task-specific curation = up to +8% vs generic. KEY: "RLVR only needs the
   final answer; SFT needs full traces — SFT data strategies DON'T transfer."
   => For us: don't dump all BIRD train into GRPO. Curate to prompts where the
      base is SOMETIMES right (non-saturated => live gradient, anti-collapse),
      drop trivially-easy and impossible ones. This is the data-side fix for #2.

CROSS-CUTTING SYNTHESIS (the one paragraph)
-------------------------------------------
Our flat EX is overdetermined and consistent with general (non-SQL) theory:
the base is the ceiling (#1), our GRPO gradient was collapsing (#2), our SFT
overfit as theory predicts (#3), LoRA wasn't the limiter but our LR/target-
modules might have been (#4), and we trained on uncurated, partly-saturated
data (#5). NONE of these is SQL-specific. The leverage order is:
  (1) swap to a stronger base/init  >  (2) curate anti-collapse data + raise
  rollouts to 16  >  (3) fix LoRA LR(10x)/all-layers + short SFT cold-start  >
  (4) add test-time sampling (pass@k is where the base is strong — exploit it!).
Note the elegant tie-in: #1 says base wins at pass@k, so SELF-CONSISTENCY /
best-of-N at eval directly cashes in that latent base capacity for cheap — the
single highest ROI move that needs no retraining.

SOURCES (broad sweep)
  Limit of RLVR — limit-of-rlvr.github.io
  Rewarding the Unlikely (GRPO beyond sharpening) — researchgate 397423092
  Advantage Collapse in GRPO / AVSPO — arxiv 2605.21125 (ICML 2026)
  EDGE-GRPO (OpenReview VZermIifAQ); EP-GRPO — arxiv 2605.04960
  Transformation-Augmented GRPO — arxiv 2601.22478
  SFT Memorizes, RL Generalizes — arxiv 2501.17161 (ICML 2025)
  RL Heals OOD Forgetting in SFT — arxiv 2509.12235
  LoRA Without Regret — thinkingmachines.ai/blog/lora
  LoRA vs Full FT: Illusion of Equivalence — arxiv 2410.21228
  Data curation: SUPERNOVA 2604.08477; Front-Loading Reasoning 2510.03264;
    Cross-Domain RL 2506.14965; Klear-Reasoner 2508.07629
  Survey/overview: llm-stats.com post-training-techniques-2026; GRPO++ tricks
    (cameronrwolfe.substack.com/p/grpo-tricks); interconnects base-model-RL

CODE CHECK against grpo.py (Session 6, verified live):
  - target_modules = q,k,v,o,gate,up,down  => ALL layers incl MLP. #4a SATISFIED.
  - --lr default 1e-6  => this is FULL-FT-scale RL LR. Per "LoRA Without Regret"
    LoRA optimum ~10x full-FT (~1e-5; ~15x for <100-step runs). We likely ran
    GRPO at ~1/10 the correct LoRA LR => UNDER-TRAINED. CONCRETE FIX: --lr 1e-5.

================================================================================
SESSION 7 — TEST-TIME SELF-CONSISTENCY (built + first result) 2026-06-30
================================================================================
BUILT (eval harness): execution-based self-consistency / majority vote.
  - eval/vote.py: select_by_execution(db, candidates) — run all N, cluster by
    result-set (frozenset of row tuples, official-EX semantics), pick largest
    cluster's query; valid-only voting; fallback to cand[0] if none run. Unit-
    tested on real BIRD db (3-vote cluster wins over diff-but-valid + invalid).
  - eval/predictors.py: VLLMPredictor n-sampling via SamplingParams(n=N) +
    generate_candidates() (prefix-caching shares the prompt KV across the N
    samples → N-sampling on long schemas is cheap, no n× prompt-KV blowup).
  - eval/run_eval.py: --n-samples flag; voting stage (parallel across examples);
    summary gains n_samples/mean_agreement/mean_cluster_size; per-ex "vote" meta.
    Guards: n>1 requires vllm backend + temperature>0.

RESULT — Arctic-Text2SQL-R1-7B + SC n=8 (fp8, temp 0.8, top_p 0.95, generic=arctic
  template, max-model-len 8192, max-num-seqs 16, full BIRD dev 1534):
  | model                    | EX     | valid | simple/mod/chall   |
  | Arctic single-pass (Arm2)| 0.6208 | 0.994 | .679/.540/.507     |
  | Arctic + SC n=8          | 0.6377 | 1.000 | .697/.553/.528     |  Δ +1.69
  mean_agreement 0.843, mean_cluster_size 6.75/8. gen 9122s (~2.5h), vote 108s.
  results/arctic_sc8_full.jsonl.

READ: +1.7 only (lit says +4-8) BECAUSE Arctic is ALREADY RL-sharpened to pass@1:
  agreement 0.84 = the 8 samples almost always AGREE → little diversity for voting
  to exploit. This CONFIRMS the Session-6 "RLVR collapses to single mode" finding
  empirically on our own setup. Prediction: SC pays MORE on a NON-RL base (real
  sampling diversity). NEXT: SC n=8 on base Coder-7B (51.2 single-pass) to measure
  the free pass@k headroom on an un-sharpened model. Memory ceiling stable: vLLM
  pre-allocs KV pool (gpu-mem-frac 0.88), enforce_eager (no graph burst), prefix-
  cache shares prompt KV — NO runtime spike observed across the 2.5h run.

GPU-MEM NOTE: gpu-mem-frac 0.92 failed at init (desktop holds ~1.2GB → only 14.17
  free); 0.88 works. Use 0.88 for vLLM on this box when desktop is up.

--------------------------------------------------------------------------------
SESSION 7 RESULT 2 — base Coder-7B + SC n=8 (the diversity test) 2026-06-30
--------------------------------------------------------------------------------
Qwen/Qwen2.5-Coder-7B-Instruct, vllm fp8, generic template, temp 0.8 top_p 0.95,
n=8, max-model-len 8192, max-num-seqs 16, full BIRD dev. results/coder7b_base_sc8.jsonl

| model            | single-pass | +SC n=8 | Δ    | agreement | gen    |
| base Coder-7B    | 0.512*      | 0.5666  | +5.5 | 0.751     | 1281s  |
| Arctic-7B (RLVR) | 0.6208      | 0.6377  | +1.7 | 0.843     | 9122s  |
base SC by-diff: simple .646 / mod .458 / chal .403. valid 0.979, cluster 6.01/8.

THE FINDING (clean confirmation of Session-6 pass@k thesis):
  Self-consistency helps the UN-sharpened base ~3x more than the RL-sharpened model.
  Base has more sampling diversity (agreement 0.75 vs 0.84) → more for voting to fix.
  Arctic's GRPO already collapsed that diversity into pass@1 → little SC headroom.
  ALSO: base SC was 7x FASTER (base emits short direct SQL; Arctic emits long CoT)
  → on the base, SC is cheap + high-ROI. BUT base-SC 56.7 STILL < Arctic single 62.1
  → adopting a strong RL model still beats voting on a weak base.

⚠ OPEN CAVEAT (the ONLY loose end): base single-pass anchor 0.512* was measured at
  4-bit GREEDY (hf backend); this SC run is fp8 temp-0.8. So part of +5.5 may be
  precision/decoding, not pure voting. TO CLOSE: run base fp8 GREEDY single-pass
  (~3min, short outputs) for an apples-to-apples delta. NOT yet done.

>>> RESUME HERE TOMORROW (2026-07-01) <<<
1. RUN the clean baseline (closes the caveat above):
   cd ~/nl2sql && <env exports BLACKWELL_BUGS §0> PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   .venv/bin/python eval/run_eval.py --dataset bird --root data/dev_20240627 \
     --backend vllm --model Qwen/Qwen2.5-Coder-7B-Instruct --template generic \
     --sample-rows 3 --max-model-len 8192 --max-tokens 2048 --max-num-seqs 16 \
     --gpu-mem-frac 0.88 --quantization fp8 --tag coder7b_base_fp8_greedy \
     --out results/coder7b_base_fp8_greedy.jsonl
   (single-pass = no --n-samples, temp default 0). Then real Δ = 0.5666 − this.
2. DECISION GATE is essentially reachable now. Current full-dev EX scoreboard:
   SLM-SQL-1.5B 43.3 | Coder-3B base 39.3 | Coder-7B base 51.2(4bit)/~?(fp8) |
   Coder-7B base +SC8 56.7 | our 7B SFT 49.8 | our 7B GRPO-v3 50.9 |
   Arctic-7B FP8 62.1 | Arctic-7B +SC8 63.8.
   Likely ship answer shaping up: ADOPT Arctic (+ optional SC for +1.7). Our-own
   training (Arm 3) can't beat that without OmniSQL-init + curated data + 16 roll.
3. OPTIONAL stacking experiments if pursuing further:
   - Arctic +SC with HIGHER temp (1.0-1.2) to force diversity → may lift the +1.7.
   - SC on OmniSQL-7B / CSC-SQL-7B (other strong open 7Bs from Session-5 sweep).
   - CSC-SQL-style correction step when top-2 clusters disagree (+4.25 in lit).
4. Arm 4 (DeepEye agentic) still unstarted — eval-only reference for the gap.

HARNESS STATE: self-consistency fully wired (eval/vote.py + predictors n-sampling +
run_eval --n-samples). vLLM mem on this box: gpu-mem-frac 0.88 (0.92 fails when
desktop up), enforce_eager, prefix-cache → no spike. GPU now FREE, no bg runs.

--------------------------------------------------------------------------------
SESSION 8 (2026-07-01) — CLEAN baseline closes the SC caveat
--------------------------------------------------------------------------------
Ran base Coder-7B fp8 GREEDY single-pass (was the missing apples-to-apples anchor).
  base 4-bit greedy (old) : 0.512
  base fp8  greedy (clean): 0.5379   <- +2.6 from PRECISION alone (4bit->fp8)
  base fp8  + SC n=8       : 0.5666
=> yesterday's apparent +5.5 = +2.6 precision + 2.9 TRUE voting. results/coder7b_base_fp8_greedy.jsonl
CORRECTED (both fp8, true SC voting Δ):
  | model         | greedy | +SC8  | Δ_voting | agreement |
  | base Coder-7B | 0.5379 | 0.5666| +2.9     | 0.75      |
  | Arctic-7B     | 0.6208 | 0.6377| +1.7     | 0.84      |
THESIS HOLDS but ~1.7x not 3x: SC helps un-sharpened base more than RL-sharpened
Arctic (+2.9 vs +1.7); precision confound had inflated the gap. Honest number now.
GPU note: desktop using ~2GB today → gpu-mem-frac 0.88 ALSO failed init; 0.85 works.
  Rule: vLLM on this box = 0.85 to be safe when desktop is up.

SCOREBOARD (full BIRD dev EX): SLM-SQL-1.5B 43.3 | Coder-3B base 39.3 |
  Coder-7B base 51.2(4bit)/53.8(fp8) | +SC8 56.7 | our 7B SFT 49.8 | our 7B GRPO 50.9 |
  Arctic-7B 62.1 | Arctic-7B +SC8 63.8.  => ship leader: ADOPT Arctic (+SC optional).

>>> RESUME TOMORROW (2026-07-02) — ARM 4: DeepEye-SQL (agentic, eval-only) <<<
Goal: run the agentic pipeline through the SAME EX harness → measure single-model-
vs-agentic gap (lit ~75 on BIRD). It's eval-only (no training). From the plan:
3B-MoE generator + schema-linking + candidate gen + self-correction.
FIRST STEPS tomorrow:
  1. Recon: find DeepEye-SQL repo/weights on HF/GitHub; confirm license + what the
     pipeline actually needs (generator model id, schema-linker, retriever, judge).
  2. Decide integration: can we wrap our eval/run_eval harness around its generator,
     or do we run their pipeline and score its predict.json with our official scorer?
  3. Watch the 16GB ceiling + the same vLLM mem rule (gpu-mem-frac 0.85 when desktop up).
CONTEXT: bake-off otherwise effectively DONE. Ship leader = ADOPT Arctic-7B 62.1
  (+SC8 63.8). Arm 3 (our own) tops ~51, can't beat Arctic w/o OmniSQL-init+curated
  data+16roll. Arm 4 is the last datapoint before the final decision memo.

---

## SESSION 9 (2026-08-08) — deep reload + study-notes scaffolding
Study session, no runs. Walked the log chronologically (arms 1–2, scorer
validation, Session 4 SFT finding) rebuilding real understanding.
- NEW FILES: FINDINGS.md (portable general lessons — SFT/shortcut-learning set
  written), TECHNIQUES.md (reusable mechanisms — merge-free adapter eval T1,
  official-faithful scorer T2, probe protocol T3). Append as study continues.
- Roadmap (~/Research/Tasks/learning-roadmap.md): added track 2c (Generalization
  & Shortcut Learning) + track 2d (RL math REINFORCE→PPO→GRPO). 2d is the
  PREREQUISITE to finish this log's GRPO sections — stopped the walkthrough at
  the GRPO run #1 collapse story pending that math study.
- Repo git-initialized (.gitignore excludes data/outputs/.venv/caches; results/
  committed). README still the pre-bake-off draft — rewrite pending.
NEXT: (1) study GRPO math (track 2d), (2) finish log walkthrough (collapse →
composite reward → v2/v3 flat → Arctic counter-lesson → Session 6 lit → SC),
(3) README draft from FINDINGS/TECHNIQUES + scoreboard, (4) publish checklist
(secrets audit done for code files 2026-08-08 — clean; recheck notebooks/results
before going public).

### SESSION 10 (2026-08-11) — GRPO math study material built
No runs. Recapped the run-#1 collapse story, then built the track-2d study
reference: ~/Research/Study/grpo-math.html — REINFORCE→baseline→PPO→GRPO with
symbol table, piece-by-piece breakdowns, the step-200/collapsed/saturated groups
computed by hand, telemetry decoder, knob→math map. Milestone questions at the
bottom = the gate.
NEXT: study that page (track 2d), then resume the log walkthrough at the
composite-reward fix → v2 saturation → v3 flat → Arctic counter-lesson →
Session 6 lit → self-consistency; append RL lessons to FINDINGS.md as we go.

### SESSION 11 (2026-09-17) — walkthrough resumed: arms 1-3 recapped
No runs. Re-read arms 1-2 in depth + arm 3 overview.
- FINDINGS.md: added "Decoding & benchmarking lessons (from the SLM-SQL arm)" —
  8 portable items (protocol behind headline numbers, diagnose-before-fix,
  measure-the-ceiling-of-a-fix, greedy=absorbing state, rep-penalty wrong tool for
  structured output, sampling as voting substrate, never report a probe).
- TWO LOOSE ENDS confirmed undone (checked results/): (1) SLM-SQL self-consistency
  n=8 to reproduce their 67.3% — machinery now exists (eval/vote.py), never run on
  SLM-SQL; (2) Arctic follow-ups (a) bf16 and (b) --template omnisql to close the
  ~7pt gap to 68.9 — neither run. The omnisql rerun is ~35min and likely the bigger
  half of that gap. Both are cheap README "next steps" or quick pre-publish wins.
- Also unexplained in-log: SLM-SQL generic+greedy 37.1% (pass 1) vs 40.6% (pass 3),
  same nominal config; likely --sample-rows 3 + official-faithful scorer, NOT logged.
  Verify or phrase carefully in the README.
NEXT: chapter 3 onward — composite reward -> v2 saturated groups -> v3 flat (50.9)
-> "starved the trainer" diagnosis -> Arctic counter-lesson -> Session 6 RL theory
-> self-consistency. GRPO math page (~/Research/Study/grpo-math.html) still the
recommended prerequisite; user has the intuition (sample N, score, push toward
better-THAN-GROUP-AVERAGE), deep math deferred.

### SESSION 12 (2026-09-19) — REINFORCE understood; study pages extended
No runs. Study session on track 2d.
- NEW: ~/Research/Study/rl-primer.html — RL vocabulary translated for LLMs
  (mapping table, 8 core concepts, "what you can safely skip", family tree,
  RLHF vs RLVR). Read-once prerequisite; cross-linked both ways with grpo-math.
- grpo-math.html §2 rebuilt around FAILURE-FIRST teaching (what actually made it
  click): Monte Carlo rule + its 2 requirements, then Attempt A (f=r(y) -> scalar,
  wrong object) and Attempt B (f=grad r(y) -> structurally zero, SQLite has no
  theta), then the operator swap grad-E vs E-grad. Symbol table now states grad is
  a VECTOR (one entry per parameter).
- CORRECTION made: primer had called PPO/GRPO "just off-policy" — WRONG. Both are
  ON-POLICY (same family as REINFORCE); the importance ratio exists to PRESERVE the
  on-policy approximation across multiple epochs per batch, not to escape it. With
  TRL num_iterations=1 (our default) ratio == 1, strictly on-policy. User caught it.
STATUS: REINFORCE = understood (log-derivative trick, why sampling failed naively).
NEXT: §3 baseline (short: A = r - b, unbiased for any b, proof hinges on
probabilities summing to 1) -> §4 GRPO (group mean AS the baseline) -> §5 the three
groups by hand. THEN resume PROGRESS.md walkthrough at line ~447 (composite reward
-> v2 saturation -> v3 flat -> Arctic counter-lesson -> Session 6 theory -> SC).
