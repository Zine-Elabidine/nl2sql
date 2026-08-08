"""Plain TRL+PEFT GRPO smoke (NO Unsloth) — robust fallback for the GRPO stage.

Unsloth's GRPO path is doubly broken on our stack (vLLM .ref.weight + fast_lora
Half/Float). This uses standard HF + bitsandbytes 4-bit + PEFT LoRA + TRL
GRPOTrainer (HF-generate rollouts), which avoids Unsloth's custom kernels.
If this trains, the GRPO stage is unblocked.
"""
import os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import GRPOTrainer, GRPOConfig
from datasets import Dataset

MODEL = os.environ.get("SMOKE_MODEL", "cycloneboy/SLM-SQL-1.5B")
print(f">>> plain TRL+PEFT GRPO on {MODEL}")

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16,
                         bnb_4bit_use_double_quant=True)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, quantization_config=bnb, dtype=torch.bfloat16, device_map={"": 0})
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
lora = LoraConfig(r=16, lora_alpha=16, lora_dropout=0.0, bias="none",
                  task_type="CAUSAL_LM",
                  target_modules=["q_proj","k_proj","v_proj","o_proj",
                                  "gate_proj","up_proj","down_proj"])
model = get_peft_model(model, lora)
model.print_trainable_parameters()
print(">>> loaded base 4-bit + PEFT LoRA OK")

rows = [{"prompt": [{"role": "user", "content": f"Write SELECT statement number {i}."}]}
        for i in range(8)]
ds = Dataset.from_list(rows)

def dummy_reward(completions, **kwargs):
    out = []
    for c in completions:
        text = c[0]["content"] if isinstance(c, list) else c
        out.append(min(len(text), 200) / 200.0)
    return out

args = GRPOConfig(
    use_vllm=False, beta=0.0,
    learning_rate=5e-6, optim="adamw_8bit",
    per_device_train_batch_size=4, gradient_accumulation_steps=1,
    num_generations=4, max_completion_length=128,
    max_steps=5, logging_steps=1, save_steps=999,
    bf16=True, report_to="none", output_dir="outputs_smoke_grpo_plain",
    gradient_checkpointing=True,
)
trainer = GRPOTrainer(model=model, processing_class=tok,
                      reward_funcs=[dummy_reward], args=args, train_dataset=ds)
print(">>> starting 5 GRPO steps (plain)…")
trainer.train()
print(">>> PLAIN_GRPO_OK: peak mem GB =", round(torch.cuda.max_memory_allocated()/1e9, 2))
