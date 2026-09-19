#!/usr/bin/env python3
"""Methodology 2 (Diverse Ensemble Reasoning) — Qwen2.5-7B, SVAMP N=300.

Standalone: embeds the whole ensemble pipeline (3 strategy-diverse solvers +
majority vote), identical to the repo's `diverse ensemble classifier/`. Runs in
fp16 on CUDA (one H100 MIG slice fits a 7B). Resumable; writes predictions.json
and summary.json (with per-strategy accuracy + diversity stats).

Run inside a Slurm allocation:
    python3 ensemble_qwen_sweep.py                 # uses the allocated GPU
    python3 ensemble_qwen_sweep.py --list-gpus     # list visible devices
    python3 ensemble_qwen_sweep.py --gpu 0 --n 300
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
SLUG = "qwen2.5_7b"
STRATEGIES = ["chain_of_thought", "formula_first", "backward"]


# ----------------------------- data (SVAMP) ---------------------------------
def load_svamp(n: int, seed: int = 42):
    from datasets import load_dataset
    ds = load_dataset("ChilleD/SVAMP", split="train").shuffle(seed=seed)
    rows = []
    for item in ds.select(range(min(n, len(ds)))):
        body = str(item.get("Body", "")).strip()
        q = str(item.get("Question", "")).strip()
        rows.append({"question": f"{body} {q}".strip(), "answer": str(item["Answer"]).strip()})
    return rows


# ----------------------------- prompts --------------------------------------
_TAIL = ("You MUST end your response with exactly this format: FINAL_ANSWER: [your numerical answer]\n"
         "Do not write placeholders. Write the actual number or expression as your answer.\n"
         "Do not ask follow-up questions. Do not invite further input. Produce a complete solution now.\n\n")

def prompt_cot(q):
    return ("Solve the following problem by thinking step by step.\n"
            "Show each reasoning step clearly and in order.\n" + _TAIL +
            f"Problem: {q}\n\nStep-by-step solution:")

def prompt_formula(q):
    return ("Solve the following problem using a formula-first approach.\n"
            "First, write out every equation or formula you will need.\n"
            "Then substitute values and compute the result.\n" + _TAIL +
            f"Problem: {q}\n\nEquations and solution:")

def prompt_backward(q):
    return ("Solve the following problem using backward reasoning.\n"
            "Start by clearly stating what quantity the question asks for.\n"
            "Work backwards: what do you need to compute that? What do you need before that?\n" + _TAIL +
            f"Problem: {q}\n\nBackward reasoning:")

PROMPTS = {"chain_of_thought": prompt_cot, "formula_first": prompt_formula, "backward": prompt_backward}


# ----------------------------- extraction / scoring -------------------------
_FA = re.compile(r"FINAL_ANSWER\s*:\s*(.+)", re.IGNORECASE)
_PLACEHOLDER = re.compile(r"^(<[^>]*>|\[[^\]]*\])$")
_NUM = re.compile(r"[-+]?\d[\d,.]*")
_NUMF = re.compile(r"[-+]?\d+(?:\.\d+)?")
_FRAC = re.compile(r"\\frac\{([^{}]+)\}\{([^{}]+)\}")

def extract_final_answer(text):
    m = _FA.search(text)
    if m:
        cand = m.group(1).strip()
        if not _PLACEHOLDER.match(cand):
            return cand
    nums = _NUM.findall(text)
    if nums:
        return nums[-1].replace(",", "")
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return lines[-1] if lines else text.strip()

def _norm(t):
    t = t.strip()
    b = re.search(r"\\boxed\{(.+)\}", t, re.DOTALL)
    if b: t = b.group(1)
    t = t.strip().lower()
    t = re.sub(r"[$,%]", "", t); t = re.sub(r"\s+", " ", t); t = t.rstrip(".")
    return t.strip()

def _f(t):
    try: return float(t)
    except ValueError: return None

def _frac(t):
    m = _FRAC.match(t.strip())
    if m:
        try: return float(m.group(1)) / float(m.group(2))
        except (ValueError, ZeroDivisionError): return None
    return None

def is_correct(pred, gold):
    p, g = _norm(pred), _norm(gold)
    if p == g: return True
    pf, gf = _f(p), _f(g)
    if pf is not None and gf is not None: return abs(pf - gf) < 1e-6
    prf, grf = _frac(p), _frac(g)
    if prf is not None and grf is not None: return abs(prf - grf) < 1e-6
    if prf is not None and gf is not None: return abs(prf - gf) < 1e-6
    if grf is not None and pf is not None: return abs(pf - grf) < 1e-6
    nums = _NUMF.findall(p)
    if nums:
        lf = _f(nums[-1])
        if lf is not None and gf is not None: return abs(lf - gf) < 1e-6
    return False


def summarize(rows):
    total = len(rows); correct = sum(r["correct"] for r in rows)
    tok = sum(int(r["token_cost"]) for r in rows)
    avg = tok / total if total else 0.0; acc = correct / total if total else 0.0
    s = {"num_examples": total, "num_correct": correct, "accuracy": round(acc, 4),
         "total_token_cost": tok, "avg_token_cost": round(avg, 2),
         "nate": round((acc / avg) * 1000, 6) if avg else 0.0}
    if rows and rows[0].get("strategy_predictions"):
        strategies = list(rows[0]["strategy_predictions"].keys())
        for st in strategies:
            sc = sum(is_correct(r["strategy_predictions"].get(st, ""), r.get("gold_answer", "")) for r in rows)
            s[f"accuracy_{st}"] = round(sc / total, 4)
        if len(strategies) > 1:
            s["disagreement_rate"] = round(sum(1 for r in rows if len(set(r["strategy_predictions"].values())) > 1) / total, 4)
            s["unique_correct_rate"] = round(sum(1 for r in rows if sum(is_correct(r["strategy_predictions"].get(st, ""), r.get("gold_answer", "")) for st in strategies) == 1) / total, 4)
    return s


# ----------------------------- model ----------------------------------------
class Generator:
    def __init__(self, model_id, max_new_tokens=512):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch; self.max_new_tokens = max_new_tokens
        self.tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        if self.tok.pad_token is None: self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, trust_remote_code=True)
        self.model.to("cuda")

    def generate(self, prompt):
        torch = self.torch
        maxlen = getattr(self.model.config, "max_position_embeddings", 2048)
        enc = self.tok(prompt, return_tensors="pt", truncation=True, max_length=maxlen)
        enc = {k: v.to("cuda") for k, v in enc.items()}
        with torch.inference_mode():
            out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens,
                                      do_sample=False, pad_token_id=self.tok.pad_token_id,
                                      eos_token_id=self.tok.eos_token_id)
        ilen = enc["input_ids"].shape[1]; comp = out[0][ilen:]
        return {"text": self.tok.decode(comp, skip_special_tokens=True).strip(),
                "prompt_tokens": int(ilen), "completion_tokens": int(comp.shape[0])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out-root", default="outputs_ensemble_n300")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--gpu", default=None, help="CUDA device index or MIG UUID to pin (optional)")
    ap.add_argument("--list-gpus", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import torch
    if args.list_gpus:
        print("CUDA available:", torch.cuda.is_available(), "| count:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f"  [{i}] {torch.cuda.get_device_name(i)}")
        return
    assert torch.cuda.is_available(), "CUDA not available — run inside a Slurm GPU allocation"
    print("GPU:", torch.cuda.get_device_name(0))

    out_dir = os.path.join(args.out_root, SLUG)
    os.makedirs(out_dir, exist_ok=True)
    pred_path, summ_path = f"{out_dir}/predictions.json", f"{out_dir}/summary.json"

    rows, done = [], set()
    if os.path.exists(pred_path):
        try:
            rows = json.load(open(pred_path)); done = {r["index"] for r in rows}
            print("resume:", len(done), "done")
        except Exception:
            rows, done = [], set()

    data = load_svamp(args.n)
    print(f"Loaded {len(data)} SVAMP examples")
    print("Loading", args.model, "(fp16)...")
    gen = Generator(args.model, max_new_tokens=args.max_new_tokens)

    t0 = time.time()
    for idx, ex in enumerate(data):
        if idx in done:
            continue
        texts, preds, toks = {}, {}, {}
        try:
            for st in STRATEGIES:
                o = gen.generate(PROMPTS[st](ex["question"]))
                texts[st] = o["text"]; preds[st] = extract_final_answer(o["text"])
                toks[st] = o["prompt_tokens"] + o["completion_tokens"]
            final = Counter(preds.values()).most_common(1)[0][0]  # tie -> first inserted (CoT)
            result = {"strategy_texts": texts, "strategy_predictions": preds,
                      "prediction": final, "correct": is_correct(final, ex["answer"]),
                      "token_cost": sum(toks.values()), "per_strategy_tokens": toks,
                      "aggregation": "majority_vote"}
        except Exception as e:
            print(f"[{idx+1}] ERROR {e}")
            result = {"strategy_texts": {}, "strategy_predictions": {}, "prediction": "",
                      "correct": False, "token_cost": 0, "per_strategy_tokens": {},
                      "aggregation": "error", "error": str(e)}
        rows.append({"index": idx, "question": ex["question"], "gold_answer": ex["answer"], **result})
        if (idx + 1) % 10 == 0 or (idx + 1) == len(data):
            rows.sort(key=lambda r: r["index"])
            json.dump(rows, open(pred_path, "w"), indent=2, ensure_ascii=False)
            acc = sum(r["correct"] for r in rows) / len(rows)
            print(f"[{SLUG}] {idx+1}/{len(data)} acc={acc:.3f} "
                  f"{(idx+1-len(done))/max(time.time()-t0,1e-6):.3f}it/s")

    rows.sort(key=lambda r: r["index"])
    json.dump(rows, open(pred_path, "w"), indent=2, ensure_ascii=False)
    s = summarize(rows); json.dump(s, open(summ_path, "w"), indent=2, ensure_ascii=False)
    print("DONE:", s)


if __name__ == "__main__":
    main()
