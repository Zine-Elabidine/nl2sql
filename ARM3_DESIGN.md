# Arm 3 — Train Our Own (locked design, 2026-06-27)

Part of the EX-first 4-arm NL→SQL bake-off (see PROGRESS.md). Arms 1 (SLM-SQL)
and 2 (Arctic-7B) are done; this is Arm 3. Arm 4 (DeepEye-SQL agentic, eval-only)
is still pending after this.

## Study design — 2 axes: BASE × RECIPE

Graded by Execution Accuracy (EX), our validated official-BIRD-faithful scorer.
Every cell evaluated on **BIRD dev (primary) + Spider dev (generalization)**.

### Recipe axis (the scientific question)
Does the SFT warm-start earn its keep, or can GRPO go straight from instruct?
- **zero-shot** — instruct base, no training (free baseline, eval only)
- **SFT-only** — after SFT warm-start (eval the SFT checkpoint; ~free byproduct)
- **SFT→GRPO** — warm-start then RL  (standard recipe)
- **GRPO-only** — RL straight from the instruct model (R1-Zero style)

### Base axis (4 bases, sequenced roomy → tight; order otherwise free)
| # | Base | Size / arch | GPU fit (16GB QLoRA) | Notes |
|---|------|-------------|----------------------|-------|
| 1 | Qwen2.5-Coder-7B-Instruct | 7B dense, code | 🟢 comfortable (~5GB wts) | **anchor; proven 69% EX on BIRD** |
| 2 | Qwen2.5-Coder-3B-Instruct | 3B dense, code | 🟢 very comfortable | size-down, code-specialist |
| 3 | Qwen3.5-4B (fallback Qwen3-4B) | 4B DeltaNet+MoE+multimodal, thinking | 🟡 smoke-test arch first | general; if Unsloth chokes → dense Qwen3-4B |
| 4 | Qwen2.5-Coder-14B-Instruct | 14B dense, code | 🔴 tight — test LAST | SFT peak 15.3GB; GRPO probe peaks ~12.4GB; conservative GRPO settings (num_gen=4, short completion, bsz1); long BIRD schemas may OOM |

**Off the table (verified):** Qwen3.5-Coder does NOT exist (blog fiction);
Qwen3-Coder only ships 30B-A3B / 480B MoE (too big for 16GB).

### Full grid
4 bases × {zero-shot, SFT-only, SFT→GRPO, GRPO-only}
= 4 zero-shot evals + 4 SFT runs + **8 GRPO runs**, each scored on 2 benchmarks.

## Fixed parameters
- **GRPO dataset:** BIRD train (9.4k, 69 DBs) — in-distribution w/ BIRD dev eval
  (disjoint DBs, not leakage); same data Arctic used; "hardcore" objection moot
  since compute is free. Reward = execution-accuracy (reuse eval/db.py+compare.py).
- **SFT data:** sql-create-context + synthetic_text_to_sql (ShareGPT), chat
  template matched per base.
- **Quant:** 4-bit QLoRA (NF4) both stages, both 7B/14B — only thing that fits
  GRPO+rollouts on 16GB; QLoRA quality ≈ 16-bit LoRA. Serving precision decoupled
  (merge LoRA→16bit then FP8/GGUF at export, like Arctic FP8). LoRA-16bit SFT held
  in reserve as a quality-delta check if a 4-bit-trained model underperforms.
- **Target:** beat Qwen2.5-Coder-7B's proven 69% EX; min match Arctic FP8 62.1%.

## Gates / open work before GRPO runs
1. **GRPO path — RESOLVED.** Unsloth GRPO is doubly broken on our stack:
   (a) vLLM fast-rollout rejects `.ref.weight` on vLLM 0.23; (b) Unsloth `fast_lora`
   kernel `addmm_` casts LoRA weight to fp32 while activations are fp16 → "Half vs
   Float" (not fixed by beta=0 or explicit dtype=bf16). **Fix = use plain TRL+PEFT
   for GRPO** (NO Unsloth): bnb 4-bit + PEFT LoRA + TRL GRPOTrainer, `use_vllm=False`
   (HF-generate rollouts), `beta=0.0` (Dr.GRPO/DAPO, no ref model). Validated:
   `~/nl2sql/smoke_grpo_plain.py` trained 5 steps, full GRPO telemetry, 1.5B peak
   3.55GB. NOTE config: generation_batch (= per_device_bs × grad_accum) must be
   divisible by num_generations.
2. **Qwen3.5-4B arch — RESOLVED/CONFIRMED.** Loads + trains on plain TRL+PEFT GRPO
   (AutoModelForCausalLM, transformers 5.12), peak 8.01GB. Slot 3 = Qwen3.5-4B locked
   (fallback Qwen3-4B no longer needed). CAVEAT: thinking-by-default → never emits EOS
   at max_completion_length=128 (all completions clipped). Real Qwen3.5-4B runs need a
   LARGE max_completion_length (or disable thinking).
3. Env reminder every run: `export LD_LIBRARY_PATH=.../nvidia/cu13/lib:...`
   (bitsandbytes libnvJitLink.so.13) + the Blackwell VLLM_* exports.

## EVAL PROTOCOL DECISION (2026-06-28) — locked
- **Whole 16-cell grid scored at ONE precision = (c) 4-bit base + adapter via
  HF-generate.** Free (no merge/no disk), training-faithful (we trained on 4-bit),
  every cell comparable. Needs a new `hf` backend in run_eval.py (loads 4-bit base
  + optional `--adapter` with PEFT; zero-shot = no adapter). Quantization cost is
  ~size-dependent not recipe-dependent, so we do NOT eval every cell at 3 precisions.
- **a/b/c precision study run ONCE** on the Coder-7B SFT checkpoint:
  (a) merge→bf16, (b) merge→FP8, (c) 4-bit+adapter — gives the bf16/FP8/4-bit EX gap.
- **Final step:** take the grid WINNER and export deploy artifact (merge→FP8/GGUF);
  re-run a/b/c on it if useful. That's the only other place merge is needed.

## RESUME 2026-06-28 — Coder-7B SFT DONE, eval is the next action
**State at shutdown:** GPU free, no bg jobs. Disk TIGHT = **16GB free (94% used)**.
- ✅ **Coder-7B SFT complete** → `outputs/coder7b_sft` (adapter_model.safetensors 80MB
  + checkpoint-590). Full epoch (590 steps, ~3.95h @ batch1/accum16/max-seq2048).
  Final: train_loss 0.12, mean_token_accuracy **0.974**. Clean.
- ⬜ **NOT YET DONE: any Coder-7B EX number.** Need zero-shot BASE eval (the anchor)
  + SFT-adapter eval. THIS IS THE NEXT ACTION.

**EVAL PATH DECISION (disk-driven) — resolve first thing:**
`eval/run_eval.py` backends = gold|echo|**vllm** only; flags: --model --template
--quantization fp8 --max-model-len --max-tokens --temperature etc. **It has NO
LoRA/adapter loading** — it serves a single model dir via vLLM. So:
  - **Zero-shot base eval:** easy — point vllm at the base. Options: (a) eval the
    cached 4-bit `unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit` directly if vLLM
    accepts bnb; or (b) download official `Qwen/Qwen2.5-Coder-7B-Instruct` bf16
    (~15GB) and eval with `--quantization fp8` (like Arctic). (b) is the
    apples-to-apples match to how we'll deploy, but **15GB won't fit in 16GB free**
    alongside anything else.
  - **SFT-adapter eval:** needs `merge.py` → merged model dir. merge needs the bf16
    base (~15GB) AND writes a merged copy (~15GB) → **won't fit on 16GB free**.
  - **=> DISK PLAN NEEDED before eval.** Cheapest: free space first (HF cache has
    the 4-bit bases; check `du -sh ~/.cache/huggingface/*`). Then either: download
    bf16 base once, eval zero-shot, merge SFT (delete bf16 base after merge to make
    room), eval merged, delete merged. OR add a tiny HF-generate+PEFT eval backend
    to run_eval.py that loads 4-bit base + adapter (no download, no merge, fits
    disk; slower). The HF-generate backend is the disk-safe, reusable choice and
    also solves the 14B-merge problem the design already flagged.

**After Coder-7B eval cell:** GRPO runs (SFT->GRPO + GRPO-only) on `--subset 2000`,
then repeat whole cell for Coder-3B / Qwen3.5-4B / Coder-14B. Commands below.

## SMOKE PASSED (2026-06-28) — pipeline validated, launching grid
Real-pipeline smoke done on Coder-7B: SFT (loss 1.28, acc 0.75) + GRPO both run
end-to-end. Execution reward is REAL (not the dummy constant) and shows within-group
variance (reward_std 0.45–0.52) = genuine GRPO gradient. Fixed an OOM (bug #8):
GRPO needs `--num-generations 4` + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
(8 gens × long BIRD schema OOM'd 16GB). Default temp bumped 0.9→1.0. Coder-3B now
cached (all 4 bases ready). GRPO data = stratified `--subset 2000` (all 69 DBs).

**CORRECTED launch commands (copy-paste; note real train-root path):**
```
export LD_LIBRARY_PATH=$(.venv/bin/python -c "import os,nvidia;print(os.path.dirname(nvidia.__file__))")/cu13/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TR=data/train_extracted/train
# 1. SFT (full data, ~minutes):
.venv/bin/python train/sft.py  --model unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit --train-root $TR --out outputs/coder7b_sft
# 2. SFT->GRPO (subset):
.venv/bin/python train/grpo.py --model unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit --train-root $TR --init-adapter outputs/coder7b_sft --subset 2000 --num-generations 4 --batch-size 4 --out outputs/coder7b_sft_grpo
# 3. GRPO-only (subset, no --init-adapter):
.venv/bin/python train/grpo.py --model unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit --train-root $TR --subset 2000 --num-generations 4 --batch-size 4 --out outputs/coder7b_grpo_only
```

## RESUME HERE (2026-06-27, end of session 3)
Everything is staged; nothing trained yet. Next action = **real-pipeline smoke**
(`--max-examples 50`) then launch the grid. WAIT for user's go-ahead.

Ready:
- **Data:** BIRD train complete & validated — `data/train_extracted/train/` (train.json
  9428 ex, 69/69 DBs ok). BIRD dev at `data/dev_20240627`. (Spider dev = floor, check.)
- **Models cached (4-bit):** Coder-7B (`unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit`),
  Coder-14B (`unsloth/Qwen2.5-Coder-14B-Instruct-bnb-4bit`), Qwen3.5-4B (`Qwen/Qwen3.5-4B`),
  SLM-SQL-1.5B (smoke proxy). NOT cached: Coder-3B (download `unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit`).
- **Scripts (import-clean):** train/data.py, train/sql_reward.py, train/sft.py,
  train/grpo.py, train/merge.py. See run examples in each file's docstring.

Run order (per base): zero-shot eval (run_eval.py) → sft.py → eval → grpo.py
(--init-adapter for SFT→GRPO; omit for GRPO-only) → merge.py → eval on bird+spider.

Immediate next steps:
1. Real-pipeline smoke: `train/sft.py --max-examples 50` + `train/grpo.py
   --max-examples 50` on Coder-7B → confirm BIRD data + exec reward give reward VARIANCE
   (smoke used dummy reward = constant; real reward must show spread).
2. Decide SFT data: currently sft.py uses BIRD-train gold (aligned, no extra dl). Plan
   originally said sql-create-context + synthetic_text_to_sql — revisit if desired.
3. Eval/merge needs full-precision base (not -bnb-4bit) for merge.py → bf16 base
   download per family at eval time (disk: 18GB free, manage; 14B bf16 ~28GB won't fit
   → for 14B eval use 4-bit+adapter HF-generate instead of merge).
4. Qwen3.5-4B runs need LARGE max_completion_length (thinking mode).

ENV every run: `export LD_LIBRARY_PATH=$(.venv/bin/python -c "import os,nvidia;
print(os.path.dirname(nvidia.__file__))")/cu13/lib:$LD_LIBRARY_PATH` + VLLM_* (see
BLACKWELL_BUGS.md §0).

## Pipeline status (from smoke tests, 2026-06-27)
- Unsloth 2026.6.9 on Blackwell sm_120: import/4bit-load/LoRA/in-process-vLLM ✅
- **SFT** ✅ — Unsloth (1.5B peak 2.78GB; 14B peak 15.3GB) OR plain TRL+PEFT.
- **GRPO** ✅ — plain TRL+PEFT only (Unsloth GRPO broken). 1.5B peak 3.55GB.
- Decision: can use plain TRL+PEFT for BOTH stages (one consistent LoRA code path,
  robust, compute is free) — Unsloth SFT is the ~2x-faster option if wanted.
  Smoke scripts: smoke_sft.py (Unsloth SFT), smoke_grpo_plain.py (TRL+PEFT GRPO),
  smoke_14b_mem.py (14B mem probe).

## Training continuation — how to train a model FURTHER (discussed 2026-06-28)
**Mechanism:** `train/grpo.py --init-adapter <saved_adapter_dir>` loads a saved
adapter as the TRAINABLE start (same hook SFT->GRPO uses). Learned weights carry
over; optimizer state does NOT (fresh optimizer) — still a genuine continuation.
Example: continue GRPO-only further →
  grpo.py --model <4bit> --init-adapter outputs/coder7b_grpo_only \
          --subset ... --out outputs/coder7b_grpo_only_v2

**Three ways to "train more", best first:**
1. MORE DISTINCT DATA (raise subset / use full 9428) — GRPO benefits from coverage,
   not repetition. We used 1000 of 9428, so ~8400 unused → natural next data.
2. More epochs on the SAME subset — okay, diminishing returns (see risk below).
3. Same subset, fresh-from-base rerun — only for seed/variance checks.

**Is re-training on the same subset bad?**
- SFT: yes-ish — repeats of fixed gold → memorization/overfit (this is what made our
  SFT lose schema-grounding: token-acc 0.974, EX 49.8 < base 51.2).
- GRPO: less dangerous — target is "SQL that executes correctly", not a fixed string,
  so re-seeing a prompt = another shot at reward. BUT once the model reliably solves a
  prompt, all rollouts get equal reward → `reward_std=0` → `frac_reward_zero_std`→1 →
  ZERO gradient for that prompt = wasted compute. Over-training same set also collapses
  output diversity (entropy ↓, brittle). So watch `frac_reward_zero_std` in logs; if it
  climbs toward 1.0 the subset is saturated — switch to new prompts.

**Choosing a DIFFERENT subset (current limitation + planned fix):**
- `stratified_subset(exs, n, seed=3407)` is DETERMINISTIC: `subset 1000` is always the
  same 1000, and `subset 2000` = those same 1000 + 1000 more (round-robin continues).
  So `subset 1000 ⊂ subset 2000`. Good for fair recipe comparison (GRPO-only and
  SFT->GRPO train on the IDENTICAL 1000), bad for "give me a fresh disjoint 1000".
- PLANNED (not yet implemented): add `--subset-offset` (skip first N in round-robin,
  take next M → disjoint continuation) and/or `--subset-seed`. Then continuation =
  `--init-adapter <prev> --subset 1000 --subset-offset 1000` = genuine new data, no
  wasted re-training on mastered prompts. ~5-line change to data.py round-robin.
