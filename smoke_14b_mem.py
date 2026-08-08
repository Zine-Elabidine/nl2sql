"""14B GRPO memory probe on 16GB Blackwell.

Doesn't run the (currently dtype-buggy) GRPO trainer; instead reproduces the two
memory peaks GRPO actually hits, so we get a definitive fits/doesn't-fit answer:
  PEAK A: rollout generation — num_generations completions for a batch at once.
  PEAK B: policy training step — forward + backward on those long sequences.
If both stay under ~15.5GB with headroom, 14B QLoRA GRPO is viable here.
"""
import os, torch
import unsloth  # noqa
from unsloth import FastLanguageModel

MODEL = os.environ.get("SMOKE_MODEL", "unsloth/Qwen2.5-Coder-14B-Instruct-bnb-4bit")
MAX_SEQ = 2048
NUM_GEN = 4          # GRPO num_generations
PROMPT_LEN = 512     # realistic BIRD prompt (schema is big)
GEN_TOKENS = 256

print(f">>> probing {MODEL}")
model, tok = FastLanguageModel.from_pretrained(
    model_name=MODEL, max_seq_length=MAX_SEQ, load_in_4bit=True)
model = FastLanguageModel.get_peft_model(
    model, r=16,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_alpha=16, lora_dropout=0, use_gradient_checkpointing="unsloth", random_state=3407)
print(">>> loaded 14B 4-bit + LoRA. weights mem GB =", round(torch.cuda.memory_allocated()/1e9,2))

# ---- PEAK A: generation (rollout) ----
FastLanguageModel.for_inference(model)
prompt = "SELECT " * (PROMPT_LEN // 2)
ids = tok([prompt]*NUM_GEN, return_tensors="pt", truncation=True, max_length=PROMPT_LEN).to("cuda")
torch.cuda.reset_peak_memory_stats()
with torch.no_grad():
    model.generate(**ids, max_new_tokens=GEN_TOKENS, do_sample=True, temperature=0.8)
print(">>> PEAK A (rollout gen) GB =", round(torch.cuda.max_memory_allocated()/1e9,2))

# ---- PEAK B: training step (forward+backward) ----
FastLanguageModel.for_training(model)
seq = PROMPT_LEN + GEN_TOKENS
batch = torch.randint(0, 1000, (NUM_GEN, seq), device="cuda")
torch.cuda.reset_peak_memory_stats()
out = model(input_ids=batch, labels=batch)
out.loss.backward()
print(">>> PEAK B (train step) GB =", round(torch.cuda.max_memory_allocated()/1e9,2))
print(">>> VERDICT: total GPU =", round(torch.cuda.get_device_properties(0).total_memory/1e9,1),
      "GB; if peaks < ~15 with headroom, 14B GRPO fits.")
