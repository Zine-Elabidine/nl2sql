"""Execution-Accuracy eval harness — entry point.

Usage examples:
  # self-test the harness (no model, must report EX≈100%):
  python run_eval.py --dataset bird --root ../data/dev --backend gold --limit 50

  # wiring smoke test (constant wrong query, EX≈0, valid-SQL≈100%):
  python run_eval.py --dataset bird --root ../data/dev --backend echo --limit 50

  # real model via vLLM:
  python run_eval.py --dataset bird --root ../data/dev --backend vllm \
      --model Snowflake/Arctic-Text2SQL-R1-7B --template arctic
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import datasets as ds
import predictors as pr
import prompts as pt
from compare import score_one
from schema import schema_ddl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=ds.LOADERS, required=True)
    ap.add_argument("--root", required=True, help="extracted dataset root dir")
    ap.add_argument("--backend", default="gold", help="gold|echo|vllm|hf")
    ap.add_argument("--model", default=None)
    ap.add_argument("--adapter", default=None,
                    help="LoRA adapter dir for backend=hf (omit for zero-shot base)")
    ap.add_argument("--template", default="generic", choices=pt.TEMPLATES)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--sample-rows", type=int, default=0)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--n-samples", type=int, default=1,
                    help="self-consistency: sample N per question, vote by "
                         "result set, score the winner (vllm backend; use temp>0)")
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    ap.add_argument("--quantization", default=None, help="e.g. fp8 (fits 7B on 16GB)")
    ap.add_argument("--no-4bit", action="store_true",
                    help="hf backend: load fp16/bf16 instead of bnb 4-bit "
                         "(small models fit 16GB; avoids bnb's unstable T4 kernel)")
    ap.add_argument("--max-num-seqs", type=int, default=16)
    ap.add_argument("--gpu-mem-frac", type=float, default=0.85)
    ap.add_argument("--cuda-graphs", action="store_true",
                    help="enable CUDA graphs (default off — safer on 16GB Blackwell)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--exec-timeout", type=float, default=30.0)
    ap.add_argument("--exec-workers", type=int, default=8)
    ap.add_argument("--out", default=None, help="output JSONL path")
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()

    examples = ds.LOADERS[args.dataset](args.root)
    if args.limit:
        examples = examples[: args.limit]
    print(f"[{args.tag}] {len(examples)} examples from {args.dataset}")

    # missing-db guard (surfaces extraction problems early)
    missing = {e.db_path for e in examples if not os.path.exists(e.db_path)}
    if missing:
        print(f"WARNING: {len(missing)} db files missing, e.g. {next(iter(missing))}")

    self_consistency = args.n_samples > 1
    if self_consistency:
        if args.backend != "vllm":
            raise SystemExit("--n-samples > 1 requires --backend vllm "
                             "(needs n-sampling; merge LoRA→fp16 to vote on our own model)")
        if args.temperature <= 0:
            raise SystemExit("--n-samples > 1 needs --temperature > 0 "
                             "(identical greedy samples can't vote); use e.g. 0.8")

    tmpl = pt.TEMPLATES[args.template]

    # ---- build predictor ----
    if args.backend == "gold":
        predictor = pr.GoldPredictor([e.gold_sql for e in examples])
    elif args.backend == "echo":
        predictor = pr.EchoPredictor()
    elif args.backend == "vllm":
        predictor = pr.build("vllm", model=args.model,
                             max_model_len=args.max_model_len,
                             max_tokens=args.max_tokens,
                             max_num_seqs=args.max_num_seqs,
                             gpu_mem_frac=args.gpu_mem_frac,
                             enforce_eager=not args.cuda_graphs,
                             temperature=args.temperature, top_p=args.top_p,
                             repetition_penalty=args.repetition_penalty,
                             quantization=args.quantization, n=args.n_samples)
    elif args.backend == "hf":
        predictor = pr.build("hf", model=args.model, adapter=args.adapter,
                             max_model_len=args.max_model_len,
                             max_tokens=args.max_tokens,
                             temperature=args.temperature, top_p=args.top_p,
                             repetition_penalty=args.repetition_penalty,
                             no_4bit=args.no_4bit)
    else:
        raise SystemExit(f"unknown backend {args.backend}")

    # ---- generate ----
    t0 = time.time()
    preds: list[str] = []
    cand_sql: list[list[str]] = []  # self-consistency: extracted candidates per example
    for i in range(0, len(examples), args.batch_size):
        chunk = examples[i:i + args.batch_size]
        msgs = [tmpl(e, schema_ddl(e.db_path, args.sample_rows)) for e in chunk]
        if self_consistency:
            cands = predictor.generate_candidates(msgs)
            cand_sql.extend([pt.extract(c) for c in lst] for lst in cands)
        else:
            raw = predictor.generate(msgs)
            preds.extend(pt.extract(r) for r in raw)
        print(f"  generated {min(i+args.batch_size,len(examples))}/{len(examples)}", end="\r")
    gen_s = time.time() - t0
    print(f"\n[{args.tag}] generation done in {gen_s:.1f}s")

    # ---- self-consistency: vote by result set, pick winner per example ----
    votes_meta: list[dict] = [None] * len(examples)
    if self_consistency:
        from vote import select_by_execution
        preds = [None] * len(examples)
        t_v = time.time()
        def _vote(k):
            sql, meta = select_by_execution(
                examples[k].db_path, cand_sql[k], args.exec_timeout)
            return k, sql, meta
        with ThreadPoolExecutor(max_workers=args.exec_workers) as ex:
            for done, (k, sql, meta) in enumerate(
                    ex.map(_vote, range(len(examples)))):
                preds[k] = sql
                votes_meta[k] = meta
                print(f"  voted {done+1}/{len(examples)}", end="\r")
        print(f"\n[{args.tag}] voting ({args.n_samples}-sample) done in "
              f"{time.time()-t_v:.1f}s")

    # ---- execute + score (parallel across examples) ----
    def _score(k):
        e = examples[k]
        correct, pred_ok, gold_ok, detail = score_one(
            e.db_path, preds[k], e.gold_sql, args.exec_timeout)
        return k, correct, pred_ok, gold_ok, detail

    results = [None] * len(examples)
    t1 = time.time()
    with ThreadPoolExecutor(max_workers=args.exec_workers) as ex:
        for n, (k, correct, pred_ok, gold_ok, detail) in enumerate(
                ex.map(_score, range(len(examples)))):
            results[k] = (correct, pred_ok, gold_ok, detail)
            print(f"  scored {n+1}/{len(examples)}", end="\r")
    exec_s = time.time() - t1
    print(f"\n[{args.tag}] execution done in {exec_s:.1f}s")

    # ---- aggregate ----
    n = len(examples)
    n_gold_ok = sum(1 for r in results if r[2])
    n_correct = sum(1 for r in results if r[0])
    n_valid = sum(1 for r in results if r[1])
    # EX over examples whose gold actually runs (standard)
    ex_denom = n_gold_ok or 1
    summary = {
        "tag": args.tag, "dataset": args.dataset, "backend": args.backend,
        "model": args.model, "template": args.template, "n": n,
        "gold_runnable": n_gold_ok,
        "EX": round(n_correct / ex_denom, 4),
        "EX_over_all": round(n_correct / n, 4),
        "valid_sql_rate": round(n_valid / n, 4),
        "gen_seconds": round(gen_s, 1), "exec_seconds": round(exec_s, 1),
    }
    if self_consistency:
        summary["n_samples"] = args.n_samples
        summary["mean_agreement"] = round(
            sum(m["agreement"] for m in votes_meta) / n, 3)
        summary["mean_cluster_size"] = round(
            sum(m["cluster_size"] for m in votes_meta) / n, 2)

    # per-difficulty (BIRD)
    diffs = {}
    for k, e in enumerate(examples):
        if not e.difficulty:
            continue
        d = diffs.setdefault(e.difficulty, [0, 0])
        if results[k][2]:
            d[1] += 1
            if results[k][0]:
                d[0] += 1
    if diffs:
        summary["EX_by_difficulty"] = {
            d: round(c / (t or 1), 4) for d, (c, t) in diffs.items()}

    print(json.dumps(summary, indent=2))

    # ---- write per-example JSONL ----
    out = args.out or f"../results/{args.tag}_{args.dataset}.jsonl"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write(json.dumps({"__summary__": summary}) + "\n")
        for k, e in enumerate(examples):
            correct, pred_ok, gold_ok, detail = results[k]
            f.write(json.dumps({
                "id": e.id, "db_id": e.db_id, "difficulty": e.difficulty,
                "question": e.question, "pred_sql": preds[k], "gold_sql": e.gold_sql,
                "correct": correct, "pred_ok": pred_ok, "gold_ok": gold_ok,
                **detail,
                **({"vote": votes_meta[k]} if self_consistency else {}),
            }) + "\n")
    print(f"[{args.tag}] wrote {out}")


if __name__ == "__main__":
    main()
