# Blackwell (RTX 5060Ti, sm_120) + Unsloth/TRL — bugs & fixes

Hard-won fixes for the NL2SQL Arm-3 training pipeline on a bleeding-edge stack:
**torch 2.11.0+cu130, vLLM 0.23.0, transformers 5.12.1, trl 1.7.0, peft 0.19.1,
bitsandbytes 0.49.2, unsloth 2026.6.9**, Python 3.12, venv `~/nl2sql/.venv`.
All smoke scripts in `~/nl2sql/` (smoke_sft.py, smoke_grpo_plain.py,
smoke_unsloth.py, smoke_14b_mem.py). Verified 2026-06-27.

---

## 0. Required env exports (EVERY run)
```bash
CU13=$(.venv/bin/python -c "import os,nvidia;print(os.path.dirname(nvidia.__file__))")/cu13/lib
export LD_LIBRARY_PATH="$CU13:$LD_LIBRARY_PATH"           # bitsandbytes (bug #1)
export VLLM_USE_FLASHINFER_SAMPLER=0                       # Blackwell vLLM
export VLLM_ATTENTION_BACKEND=FLASH_ATTN                   # FlashInfer can't read sm_120
export VLLM_CACHE_ROOT=~/nl2sql/.vllm_cache               # ~/.cache/vllm is root-owned
export TOKENIZERS_PARALLELISM=false
```

---

## 1. bitsandbytes: `libnvJitLink.so.13: cannot open shared object file`
**Symptom:** `import bitsandbytes` → cextension load error; 4-bit/QLoRA unusable.
**Cause:** bnb 0.49.2 needs CUDA-13 `libnvJitLink.so.13`; torch's cu130 wheels ship
it under `site-packages/nvidia/cu13/lib/` but it's not on the loader path.
**Fix:** add that dir to `LD_LIBRARY_PATH` (see §0). Verified: `cextension.lib` loads.

## 2. Unsloth install must not clobber the Blackwell stack
**Symptom (risk):** `pip install unsloth` drags in pinned torch/vLLM/transformers,
breaking the hand-built Blackwell stack.
**Fix:** `uv pip install --python ~/nl2sql/.venv/bin/python --no-deps unsloth unsloth_zoo`.
Verified torch/vllm/transformers/trl/peft versions unchanged afterward.

## 3. trl 1.7 GRPOConfig dropped `max_prompt_length`
**Symptom:** `TypeError: GRPOConfig.__init__() got an unexpected keyword 'max_prompt_length'`.
**Fix:** remove it; keep `max_completion_length`. (Field gone in trl 1.7.)

## 4. trl 1.7 SFTConfig: `padding_free=True` default rejects `max_length`
**Symptom:** `ValueError: When padding_free=True without packing, max_length is not
enforced...`. Note Unsloth re-injects `max_length`, so setting it `None` doesn't stick.
**Fix:** pre-render the chat template into a `text` column, then set
`packing=False, padding_free=False, dataset_text_field="text"`. (Or `packing=True`
+ a `formatting_func`.) Verified SFT trains: loss 2.31→1.27, peak 2.78GB (1.5B).

## 5. Unsloth GRPO + vLLM fast-rollout: `unsupported LoRA weight ...lora_A.ref.weight`
**Symptom:** with `fast_inference=True`/`use_vllm=True`, GRPO crashes in vLLM's LoRA
loader on `...lora_A.ref.weight`.
**Cause:** Unsloth syncs LoRA *including reference-model copies* to vLLM; vLLM 0.23's
LoRA loader rejects the `.ref.weight` keys.
**Fix:** don't use the Unsloth+vLLM fast-rollout path → see bug #6 resolution.

## 6. Unsloth GRPO HF-generate: `self and mat2 must have same dtype (Half vs Float)`
**Symptom:** `use_vllm=False` GRPO crashes at
`unsloth fast_lora ... out.addmm_(XA, B.to(dtype), alpha=s)`. `out` is fp16 (Half),
LoRA weight cast to fp32 (Float) during the inference/ref forward.
**Tried & FAILED:** `beta=0.0` (moves crash to fast_lora kernel),
`dtype=torch.bfloat16` at load. The bug is inside Unsloth's custom `fast_lora`
kernel on this torch/transformers combo.
**FIX (resolved): use plain TRL+PEFT for GRPO — NO Unsloth.** Standard
`BitsAndBytesConfig` 4-bit (nf4, compute=bf16, double-quant) +
`prepare_model_for_kbit_training` + PEFT `LoraConfig`/`get_peft_model` + TRL
`GRPOTrainer`, `use_vllm=False` (HF-generate rollouts), `beta=0.0` (Dr.GRPO/DAPO —
no reference model, also dodges bug-prone ref-logprob path). This avoids Unsloth's
custom kernels entirely. Verified `smoke_grpo_plain.py`: 5 steps, full GRPO
telemetry, 1.5B peak 3.55GB.

## 7. TRL GRPO: `generation_batch_size must be divisible by num_generations`
**Symptom:** `ValueError: generation_batch_size (N) must be divisible by
num_generations (G)`. generation_batch = per_device_train_batch_size × grad_accum × world.
**Fix:** make `per_device_train_batch_size × gradient_accumulation_steps` a multiple
of `num_generations` (e.g. bs=4, accum=1, num_generations=4).

---

## 8. TRL GRPO OOM in `entropy_from_logits` on long-schema prompts (7B, 16GB)
**Symptom:** GRPO runs fine for several steps then `torch.OutOfMemoryError` in
`trl/.../grpo_trainer.py _get_per_token_logps_and_entropies → entropy_from_logits
→ logits.reshape(-1, num_classes)`. The full logits tensor is
`[num_generations × seqlen × vocab(151936)]`; short prompts fit, but a long BIRD
schema prompt × 8 generations blows past 16GB. Confirmed: 7B, num_gen=8 OOM'd at a
long-schema step (~14.7GB used, +1.26GB needed); ~1.9GB was reserved-unallocated
(fragmentation).
**Fix:** `--num-generations 4` (halves the logits batch; 4 is a valid GRPO group)
+ `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (reclaims fragmented
reserve). Re-ran full 24-step smoke clean, peak survived the long-schema prompt,
reward_std nonzero (0.45–0.52) = real learning signal. Also bumped default
temperature 0.9→1.0 for more rollout diversity (fewer zero-std groups).
NOTE for 14B: keep num_gen=4, short completion, bs=1 (already planned).

---

## Pipeline verdict
- **SFT:** Unsloth (works, ~2x faster) OR plain TRL+PEFT.
- **GRPO:** plain TRL+PEFT ONLY (Unsloth GRPO broken on this stack — bugs #5, #6).
- Recommended: plain TRL+PEFT for BOTH stages = one consistent LoRA code path,
  robust; compute is free so the Unsloth SFT speedup isn't critical.
- **14B QLoRA fits 16GB but tight:** SFT peak 15.3GB; GRPO mem probe peaks ~12.4GB
  (test it last, conservative settings: num_generations=4, short completions, bs=1).

---

## 9. GRPO OOM on long-schema prompts (subset training, 16GB) — prompt-len filter
**Symptom:** GRPO-only on `--subset 1000` ran fine ~18 steps then `OutOfMemoryError`
in the attention forward (`sdpa_attention_forward`) at step 19 (peak 14.98GB). The
num_gen=4 smoke survived ONLY because it used `--max-examples 24` = first-24 = short
schemas. Real subset spans all 69 DBs incl. giant schemas.
**Cause:** BIRD prompt token lengths are long-tailed: median 458, p90 1133, p95 1927,
**p99 6610, max 6710**. A 6k-token prompt × 4 rollouts × full-vocab (152k) entropy/
logits + attention OOMs 16GB. Not fragmentation (expandable_segments already on).
**Fix:** added `--max-prompt-tokens` to train/grpo.py (default 2048) — tokenizes each
prompt and drops those over budget BEFORE training. Budget 2048 keeps 956/1000
(95.6%), dropping only the ~44 monster-schema prompts. (1536→94.2%, 1024→88.7% if
more headroom needed.) Trains on the kept set; ~956 steps for subset 1000.

## 10. GOTCHA: `apply_chat_template(tokenize=True)` returns BatchEncoding, not a list
`len(tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True))` == **2**
(it returns a BatchEncoding dict with keys input_ids/attention_mask, so len = #keys),
NOT the token count. My first prompt-len filter used this and was a silent no-op
(2 <= 2048 always true → kept everything → still OOM'd). FIX: read input_ids:
`ids = enc["input_ids"] if hasattr(enc,"keys") else enc; n_tok = len(ids)`.
Lesson: when measuring prompt length, always verify the tokenizer return type.
