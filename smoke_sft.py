"""Unsloth SFT smoke test on Blackwell — stage 1 of our recipe.
Load 1.5B 4-bit -> LoRA -> a few SFT steps on a tiny dummy chat dataset.
"""
import os, torch
import unsloth  # noqa
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

MODEL = os.environ.get("SMOKE_MODEL", "cycloneboy/SLM-SQL-1.5B")
MAX_SEQ = 1024
print(f">>> SMOKE_MODEL = {MODEL}")

model, tok = FastLanguageModel.from_pretrained(
    model_name=MODEL, max_seq_length=MAX_SEQ, load_in_4bit=True,
)
print(">>> model loaded OK")
model = FastLanguageModel.get_peft_model(
    model, r=16,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_alpha=16, lora_dropout=0, use_gradient_checkpointing="unsloth", random_state=3407,
)
print(">>> LoRA attached OK")

rows = []
for i in range(16):
    msgs = [{"role": "user", "content": f"Translate question {i} to SQL."},
            {"role": "assistant", "content": f"SELECT col{i} FROM t WHERE id = {i};"}]
    rows.append({"text": tok.apply_chat_template(msgs, tokenize=False)})
ds = Dataset.from_list(rows)

trainer = SFTTrainer(
    model=model, processing_class=tok, train_dataset=ds,
    args=SFTConfig(
        per_device_train_batch_size=2, gradient_accumulation_steps=2,
        warmup_steps=1, max_steps=5, learning_rate=2e-4, logging_steps=1,
        optim="adamw_8bit", max_length=MAX_SEQ, packing=False, padding_free=False,
        dataset_text_field="text", output_dir="outputs_smoke_sft",
        bf16=torch.cuda.is_bf16_supported(), report_to="none", seed=3407,
    ),
)
print(">>> starting 5 SFT steps…")
trainer.train()
print(">>> SFT_SMOKE_OK: peak mem GB =", round(torch.cuda.max_memory_allocated()/1e9, 2))
