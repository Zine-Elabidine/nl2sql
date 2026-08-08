"""Merge a trained LoRA adapter into its base -> 16-bit dir for eval/serving.

Loads the base in bf16 on CPU (avoids VRAM contention), merges the adapter, saves
a standalone model dir. Then evaluate with the existing harness:
  python eval/run_eval.py --dataset bird --root data/dev --backend vllm \
      --model <out_dir> --template generic --quantization fp8

Example:
  python train/merge.py --base unsloth/Qwen2.5-Coder-7B-Instruct \
      --adapter outputs/coder7b_sft_grpo --out outputs/coder7b_sft_grpo_merged
"""
from __future__ import annotations
import argparse, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="full-precision base (NOT the -bnb-4bit repo)")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"[merge] loading base {args.base} (bf16, CPU)…")
    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16, device_map="cpu")
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()
    model.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.adapter).save_pretrained(args.out)
    print(f"[merge] saved merged model -> {args.out}")


if __name__ == "__main__":
    main()
