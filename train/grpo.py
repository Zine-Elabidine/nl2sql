"""Arm-3 GRPO with execution-accuracy reward (plain TRL+PEFT, 4-bit QLoRA).

Two recipes via --init-adapter:
  GRPO-only : omit --init-adapter -> fresh LoRA on the instruct base
  SFT->GRPO : --init-adapter outputs/<base>_sft -> continue-train the SFT adapter

Unsloth GRPO is broken on this stack (see BLACKWELL_BUGS.md); this uses standard
HF+PEFT+TRL with HF-generate rollouts and beta=0 (Dr.GRPO, no reference model).

Example:
  python train/grpo.py --model unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit \
      --init-adapter outputs/coder7b_sft --train-root data/train \
      --num-generations 8 --max-completion 512 --out outputs/coder7b_sft_grpo
"""
from __future__ import annotations
import argparse, os, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import GRPOTrainer, GRPOConfig
from datasets import Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D
from sql_reward import composite_reward, execution_reward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="base model (4-bit)")
    ap.add_argument("--init-adapter", default=None, help="SFT adapter to continue (SFT->GRPO); omit for GRPO-only")
    ap.add_argument("--train-root", default="data/train")
    ap.add_argument("--template", default="generic")
    ap.add_argument("--sample-rows", type=int, default=0)
    ap.add_argument("--max-examples", type=int, default=0)
    ap.add_argument("--subset", type=int, default=0, help="stratified subset across db_id (0=all)")
    ap.add_argument("--max-seq", type=int, default=4096)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--max-completion", type=int, default=512)
    ap.add_argument("--max-prompt-tokens", type=int, default=2048,
                    help="drop prompts longer than this (long BIRD schemas OOM 16GB); 0=off")
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--beta", type=float, default=0.03,
                    help="KL anchor to ref model; >0 prevents collapse to a "
                         "degenerate policy (0 = Dr.GRPO no-ref, collapse-prone)")
    ap.add_argument("--reward", choices=["composite", "execution"],
                    default="composite",
                    help="composite=anti-hacking (default); execution=legacy 0.1-floor")
    ap.add_argument("--batch-size", type=int, default=8, help="must make bs*accum a multiple of num_generations")
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=1.0, help="higher => more rollout diversity => fewer zero-std groups")
    ap.add_argument("--no-4bit", action="store_true",
                    help="load in fp16/bf16 instead of bnb 4-bit (fits small models "
                         "on 16GB; avoids bnb's unstable 4-bit kernel on T4/sm_75). "
                         "Use the NON-quantized base model with this flag.")
    ap.add_argument("--exec-workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.init_adapter or args.model)
    exs = D.load_bird_train(args.train_root)
    if args.subset:
        exs = D.stratified_subset(exs, args.subset)
    if args.max_examples:
        exs = exs[: args.max_examples]
    print(f"[grpo] {len(exs)} BIRD-train prompts; recipe={'SFT->GRPO' if args.init_adapter else 'GRPO-only'}")
    rows = [D.grpo_row(e, args.template, args.sample_rows) for e in exs]
    # Drop pathologically long-schema prompts: with num_generations rollouts +
    # full-vocab entropy/logits, a giant BIRD schema OOMs 16GB (see BLACKWELL_BUGS).
    if args.max_prompt_tokens:
        kept = []
        for r in rows:
            enc = tok.apply_chat_template(
                r["prompt"], tokenize=True, add_generation_prompt=True)
            ids = enc["input_ids"] if hasattr(enc, "keys") else enc
            n_tok = len(ids)
            if n_tok <= args.max_prompt_tokens:
                kept.append(r)
        print(f"[grpo] prompt-len filter (<= {args.max_prompt_tokens} tok): "
              f"kept {len(kept)}/{len(rows)}")
        rows = kept
    ds = Dataset.from_list(rows)

    # Blackwell supports bf16; Kaggle T4/P100 (Turing/Pascal) do NOT -> fall back fp16
    use_bf16 = torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f"[grpo] compute dtype = {'bf16' if use_bf16 else 'fp16'}")
    if args.no_4bit:
        print("[grpo] loading in", "bf16" if use_bf16 else "fp16", "(no 4-bit)")
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=compute_dtype, device_map={"": 0})
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    else:
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=compute_dtype,
                                 bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=bnb, dtype=compute_dtype, device_map={"": 0})
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    if args.init_adapter:
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_r, lora_dropout=0.0, bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]))
    model.print_trainable_parameters()

    reward_fn = composite_reward if args.reward == "composite" else execution_reward
    print(f"[grpo] reward={args.reward}  beta(KL)={args.beta}")
    trainer = GRPOTrainer(
        model=model, processing_class=tok,
        reward_funcs=[reward_fn],
        train_dataset=ds,
        args=GRPOConfig(
            use_vllm=False, beta=args.beta,
            learning_rate=args.lr, optim="adamw_8bit",
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_generations=args.num_generations,
            max_completion_length=args.max_completion,
            temperature=args.temperature,
            num_train_epochs=args.epochs, warmup_ratio=0.03,
            logging_steps=1, save_strategy="steps", save_steps=150,
            save_total_limit=2, max_grad_norm=0.1,
            bf16=use_bf16, fp16=not use_bf16,
            gradient_checkpointing=True, report_to="none",
            output_dir=args.out, seed=3407,
        ),
    )
    trainer.train()
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"[grpo] saved adapter -> {args.out}")


if __name__ == "__main__":
    main()
