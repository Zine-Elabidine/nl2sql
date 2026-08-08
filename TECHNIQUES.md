# Techniques — reusable mechanisms built or adopted in this project

Study notes (2026-08-08). Each entry: the problem, the mechanism, the cost/tradeoff.

## T1. Merge-free adapter evaluation (`--backend hf`): score QLoRA checkpoints without materializing merged models
- **Problem:** Arm 3 trains LoRA adapters on a 4-bit base (QLoRA — only way a 7B/14B
  trains on 16GB). The vLLM eval backend wants a full merged 16-bit model on disk. With
  a 16-cell grid × checkpoints, that's a bf16 merge + multi-GB disk copy *per scored
  checkpoint* — on a machine that was at 94% disk.
- **Mechanism:** `HFPredictor` in `eval/predictors.py` (+ `run_eval.py --backend hf
  --adapter <dir>`): load the already-cached 4-bit base with bitsandbytes, attach the
  LoRA adapter live with PEFT (no merge — the adapter's lora_A/lora_B run as a second
  path alongside each frozen linear), generate with plain HF `generate()`. Zero-shot =
  same command minus `--adapter`.
- **Why it's also *fairer*, not just cheaper:** it scores at the exact precision we
  trained at. Locked protocol: all 16 grid cells scored uniformly at 4-bit+adapter, so
  cell-vs-cell deltas are purely recipe effects. The bf16/FP8/4-bit precision gap is
  measured ONCE (a/b/c study on one checkpoint), not confounded into every comparison;
  only the grid winner gets the expensive merge→FP8/GGUF deploy export.
- **Cost:** HF generate is slow (~32 min/full-1534 vs vLLM's minutes). Acceptable for
  grid scoring; vLLM stays the backend for adopt-model runs and self-consistency.
- **Gotcha (OOM rule):** unmerged LoRA adds activation memory (second compute path +
  bf16 upcast per linear), and a batch's activation peak is set by its LONGEST example
  — BIRD schemas vary hugely. Weights ~6GB fixed; activations ~linear in batch size.
  Base eval fit at batch 16; adapter eval needs **batch 8** (~9GB). Greedy decoding ⇒
  batch size changes speed only, never EX (per-example decode is deterministic and
  independent), so shrinking the batch is a free fix.
  Same landmine bit SFT training (batch 1) and GRPO (num_generations 4).
  *[Mechanical deep-dive deferred — flagged for later study: what lives in 4-bit
  (frozen base) vs bf16 (LoRA A/B, activations, optimizer states) and why activations,
  not weights, are what OOM.]*

## T2. Official-faithful EX scoring embedded in our own harness
- **Problem:** official BIRD `evaluation_ex.py` is a ~5-line batch scorer over a
  pre-made predictions file — no generation, prompting, schema rendering, extraction,
  or diagnostics. Using it directly means building 90% of a harness anyway.
- **Mechanism:** mirror its exact comparison semantics inside `eval/compare.py`:
  `set(tuple(row))` over native Python types, no stringify, no cell-sort, ORDER BY
  ignored (kept behind `enforce_order=True` as a stricter debug path). Validated two
  ways: gold self-test = EX 1.000, and re-scoring saved predictions with the official
  logic verbatim → agreement within 1 example (see FINDINGS F3).
- **Bonus:** thread-safe timeouts via `threading.Timer` + `connection.interrupt()`
  instead of fork/subprocess (fork-from-threads deadlocked; fix made eval ~20× faster).

## T3. Probes as smoke tests, full-dev as results (protocol rule)
- **Problem:** `--limit 50` probes sample only california_schools (one hard DB) —
  probe *rankings* flipped on full dev (FINDINGS F2).
- **Rule:** probes validate plumbing and support diagnosis; every reported number is
  full 1534, official-faithful scorer, stated precision + prompt + decoding.

---
*Pending — to append as the walkthrough continues:*
- Composite reward design (capped partials: row-Jaccard / schema-linking / syntax −
  length penalty) + KL anchor for GRPO
- Execution-based self-consistency (eval/vote.py: result-set clustering, prefix-cached
  n-sampling)
- Kaggle T4 offload recipe (fp16 --no-4bit, single-GPU pin, dataset mounting)
- Blackwell/vLLM survival kit (FlashInfer off, eager mode, gpu-mem-frac 0.85 rule)
