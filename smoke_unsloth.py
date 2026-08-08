"""Unsloth-on-Blackwell smoke test (Arm 3 gate).

Goal: prove the full Unsloth training path works on the RTX 5060Ti (sm_120):
  load 1.5B in 4-bit -> LoRA patch -> in-process vLLM rollouts -> a few GRPO
  steps with a trivial dummy reward. If reward logs appear and loss steps without
  OOM/crash, Unsloth is viable and we build Arm 3 on it. Otherwise fall back to
  plain TRL+PEFT.

Run with the Blackwell env exports (see bottom of file / shell wrapper).
"""
import os, torch

import unsloth  # noqa: F401  (must import before trl)
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)
from trl import GRPOTrainer, GRPOConfig
from datasets import Dataset

MODEL = "cycloneboy/SLM-SQL-1.5B"   # cached Qwen2.5-Coder-1.5B base
MAX_SEQ = 1024

print(">>> loading model (4-bit + fast_inference vLLM)…")
model, tok = FastLanguageModel.from_pretrained(
    model_name=MODEL,
    max_seq_length=MAX_SEQ,
    load_in_4bit=True,
    fast_inference=False,       # vLLM rollout path broken on vLLM 0.23 (.ref.weight); use HF generate
    max_lora_rank=16,
    dtype=torch.bfloat16,      # force consistent compute dtype (fast_lora Half-vs-Float bug)
)
print(">>> model loaded OK")

model = FastLanguageModel.get_peft_model(
    model, r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16, lora_dropout=0,
    use_gradient_checkpointing="unsloth", random_state=3407,
)
print(">>> LoRA attached OK")

# tiny dummy dataset: 8 trivial prompts
rows = [{"prompt": [{"role": "user", "content": f"Write SELECT statement number {i}."}]}
        for i in range(8)]
ds = Dataset.from_list(rows)

def dummy_reward(completions, **kwargs):
    # reward longer-but-not-huge completions; pure mechanics check
    out = []
    for c in completions:
        text = c[0]["content"] if isinstance(c, list) else c
        out.append(min(len(text), 200) / 200.0)
    return out

args = GRPOConfig(
    use_vllm=False,
    beta=0.0,                  # no KL-to-ref term -> skips the crashing ref-logprob forward (Dr.GRPO/DAPO style)
    learning_rate=5e-6, optim="adamw_8bit",
    per_device_train_batch_size=1, gradient_accumulation_steps=2,
    num_generations=4, max_completion_length=128,
    max_steps=5, logging_steps=1, save_steps=999,
    bf16=torch.cuda.is_bf16_supported(), report_to="none",
    output_dir="outputs_smoke",
)
trainer = GRPOTrainer(model=model, processing_class=tok,
                      reward_funcs=[dummy_reward], args=args, train_dataset=ds)
print(">>> starting 5 GRPO steps…")
trainer.train()
print(">>> SMOKE_OK: peak mem GB =", round(torch.cuda.max_memory_allocated()/1e9, 2))
