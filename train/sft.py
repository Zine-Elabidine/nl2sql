"""Arm-3 SFT warm-start (plain TRL+PEFT, 4-bit QLoRA).

Trains a LoRA adapter to emit gold SQL for BIRD-train (question+schema -> ```sql```).
Saves the adapter to --out for later GRPO and/or eval (via merge.py + run_eval.py).

Example:
  python train/sft.py --model unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit \
      --train-root data/train --template generic --epochs 1 --out outputs/coder7b_sft
"""
from __future__ import annotations
import argparse, os, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train-root", default="data/train")
    ap.add_argument("--template", default="generic")
    ap.add_argument("--sample-rows", type=int, default=0)
    ap.add_argument("--max-examples", type=int, default=0, help="0=all")
    ap.add_argument("--max-seq", type=int, default=4096)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    exs = D.load_bird_train(args.train_root)
    if args.max_examples:
        exs = exs[: args.max_examples]
    print(f"[sft] {len(exs)} BIRD-train examples")
    rows = [D.sft_row(e, tok, args.template, args.sample_rows) for e in exs]
    ds = Dataset.from_list(rows)

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb, dtype=torch.bfloat16, device_map={"": 0})
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_r, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]))
    model.print_trainable_parameters()

    trainer = SFTTrainer(
        model=model, processing_class=tok, train_dataset=ds,
        args=SFTConfig(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            warmup_ratio=0.03, num_train_epochs=args.epochs, learning_rate=args.lr,
            logging_steps=10, optim="adamw_8bit", max_length=args.max_seq,
            packing=False, padding_free=False, dataset_text_field="text",
            bf16=True, gradient_checkpointing=True, report_to="none",
            output_dir=args.out, save_strategy="epoch", seed=3407,
        ),
    )
    trainer.train()
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"[sft] saved adapter -> {args.out}")


if __name__ == "__main__":
    main()
