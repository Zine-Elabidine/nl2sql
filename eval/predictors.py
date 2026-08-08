"""Pluggable model backends that turn chat messages into SQL text.

Heavy deps (torch / vllm / transformers) are imported lazily inside each
predictor so the harness (data loading, execution, scoring) imports and runs on
a CPU-only box with no ML stack installed. Build/test the harness first; attach a
GPU backend when the GPU is free.
"""
from __future__ import annotations

from typing import Protocol


class Predictor(Protocol):
    def generate(self, batch: list[list[dict]]) -> list[str]:
        """Map a batch of chat-message lists to a batch of raw output strings."""
        ...


class EchoPredictor:
    """No-model stub for wiring/smoke tests: returns a trivial constant query."""

    def __init__(self, sql: str = "SELECT 1"):
        self.sql = sql

    def generate(self, batch):
        return [f"```sql\n{self.sql}\n```" for _ in batch]


class GoldPredictor:
    """Returns the gold SQL — used to self-test the harness (must score EX=100%)."""

    def __init__(self, gold_by_index: list[str]):
        self._gold = gold_by_index
        self._i = 0

    def generate(self, batch):
        out = [f"```sql\n{self._gold[self._i + k]}\n```" for k in range(len(batch))]
        self._i += len(batch)
        return out


class VLLMPredictor:
    """vLLM offline inference. Lazy import; constructs the engine on first use."""

    def __init__(self, model: str, max_model_len: int = 4096,
                 temperature: float = 0.0, max_tokens: int = 1024,
                 dtype: str = "bfloat16", gpu_mem_frac: float = 0.85,
                 tokenizer_chat: bool = True,
                 enforce_eager: bool = True, max_num_seqs: int = 16,
                 top_p: float = 1.0, repetition_penalty: float = 1.0,
                 quantization: str | None = None, n: int = 1):
        # Blackwell (sm_120) + CUDA-13 vLLM quirks, set before importing vllm:
        # - FlashInfer's bundled build can't read sm_120 ("requires sm75+") → off
        # - force FlashAttention for attention (it detects sm_120 fine)
        # - redirect cache off the root-owned ~/.cache/vllm
        import os
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
        os.environ.setdefault(
            "VLLM_CACHE_ROOT", os.path.expanduser("~/nl2sql/.vllm_cache"))
        from vllm import LLM, SamplingParams
        # enforce_eager + capped max_num_seqs avoid the CUDA-graph capture
        # memory burst that can freeze a 16GB Blackwell box (see vllm-local NOTES).
        # quantization="fp8" halves a 7B's weights (~14.5GB→~7.5GB) so it fits a
        # 16GB Blackwell card with room for KV cache; native FP8 on sm_120.
        self.llm = LLM(model=model, max_model_len=max_model_len, dtype=dtype,
                       gpu_memory_utilization=gpu_mem_frac,
                       enforce_eager=enforce_eager, max_num_seqs=max_num_seqs,
                       quantization=quantization)
        # repetition_penalty>1 breaks the degenerate loops small reasoning models
        # fall into under greedy decoding (e.g. endless </div>); top_p<1 enables
        # nucleus sampling when temperature>0 (SLM-SQL's intended decoding).
        # n>1 = draw n independent samples per prompt in ONE batched call; vLLM
        # shares the prompt KV-cache across the n samples (PagedAttention), so
        # self-consistency costs far less than n separate generations.
        self.n = n
        self.sp = SamplingParams(temperature=temperature, max_tokens=max_tokens,
                                 top_p=top_p, repetition_penalty=repetition_penalty,
                                 n=n)
        self.tok = self.llm.get_tokenizer()
        self.tokenizer_chat = tokenizer_chat

    def _render(self, messages):
        if self.tokenizer_chat and self.tok.chat_template:
            return self.tok.apply_chat_template(messages, tokenize=False,
                                                add_generation_prompt=True)
        return "\n\n".join(m["content"] for m in messages)

    def generate(self, batch):
        prompts = [self._render(m) for m in batch]
        outs = self.llm.generate(prompts, self.sp)
        return [o.outputs[0].text for o in outs]

    def generate_candidates(self, batch):
        """Return all n samples per prompt: list (per prompt) of list[str]."""
        prompts = [self._render(m) for m in batch]
        outs = self.llm.generate(prompts, self.sp)
        return [[c.text for c in o.outputs] for o in outs]


class HFPredictor:
    """Plain HuggingFace `generate` on a 4-bit base + optional LoRA adapter.

    Disk/RAM-free alternative to merge+vLLM: loads the cached 4-bit base on the
    GPU and (optionally) attaches a PEFT adapter — the exact precision we trained
    on. Slower than vLLM but needs no bf16 download and no merged copy, and is the
    only path that fits the 14B. Zero-shot = omit `adapter`.
    """

    def __init__(self, model: str, adapter: str | None = None,
                 max_model_len: int = 4096, max_tokens: int = 1024,
                 temperature: float = 0.0, top_p: float = 1.0,
                 repetition_penalty: float = 1.0, no_4bit: bool = False,
                 **_ignored):
        import os
        os.environ.setdefault(
            "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        import torch
        from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                  BitsAndBytesConfig)
        self.torch = torch
        self.max_model_len = max_model_len
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        tok_src = adapter or model
        self.tok = AutoTokenizer.from_pretrained(tok_src)
        # left-pad for correct batched decoder generation
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        dt = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if no_4bit:
            self.model = AutoModelForCausalLM.from_pretrained(
                model, dtype=dt, device_map={"": 0})
        else:
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_compute_dtype=dt,
                                     bnb_4bit_use_double_quant=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model, quantization_config=bnb, dtype=dt,
                device_map={"": 0})
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()

    def _render(self, messages):
        return self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)

    def generate(self, batch):
        prompts = [self._render(m) for m in batch]
        enc = self.tok(prompts, return_tensors="pt", padding=True,
                       truncation=True,
                       max_length=self.max_model_len - self.max_tokens).to(
            self.model.device)
        do_sample = self.temperature > 0
        gen_kw = dict(max_new_tokens=self.max_tokens, do_sample=do_sample,
                      repetition_penalty=self.repetition_penalty,
                      pad_token_id=self.tok.pad_token_id)
        if do_sample:
            gen_kw.update(temperature=self.temperature, top_p=self.top_p)
        with self.torch.no_grad():
            out = self.model.generate(**enc, **gen_kw)
        # strip the prompt tokens; decode only the continuation
        gen = out[:, enc["input_ids"].shape[1]:]
        return self.tok.batch_decode(gen, skip_special_tokens=True)


def build(backend: str, **kw) -> Predictor:
    if backend == "echo":
        return EchoPredictor(**kw)
    if backend == "vllm":
        return VLLMPredictor(**kw)
    if backend == "hf":
        return HFPredictor(**kw)
    raise ValueError(f"unknown backend {backend!r}")
