#!/usr/bin/env python3
"""Methodology 2 (Diverse Ensemble Reasoning) — LOCAL small models, SVAMP N=300.

Runs the 3-strategy ensemble (CoT + Formula-First + Backward -> majority vote) on
the three small models on Apple Silicon (MPS, fp16), one after another. Standalone,
resumable. Writes outputs_ensemble_n300/<slug>/{predictions,summary}.json.

Overnight launch (detached + caffeinated):
    cd <this folder>
    nohup caffeinate -i python3 -u ensemble_local_sweep.py > ensemble_local.log 2>&1 &
    disown
    tail -n 30 ensemble_local.log
"""
from __future__ import annotations

import argparse, json, os, re, time
from collections import Counter

MODELS = {  # model_id -> output slug (matches Methodology-1 outputs_n300 slugs)
    "meta-llama/Llama-3.2-3B-Instruct": "llama3.2",
    "google/gemma-2-2b-it": "gemma2-2b",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": "Deepseek-r1-1.5b",
}
STRATEGIES = ["chain_of_thought", "formula_first", "backward"]

# ----- data (SVAMP, seed-42 shuffle: first 100 of 300 == published subset) -----
def load_svamp(n, seed=42):
    from datasets import load_dataset
    ds = load_dataset("ChilleD/SVAMP", split="train").shuffle(seed=seed)
    rows = []
    for it in ds.select(range(min(n, len(ds)))):
        body = str(it.get("Body", "")).strip(); q = str(it.get("Question", "")).strip()
        rows.append({"question": f"{body} {q}".strip(), "answer": str(it["Answer"]).strip()})
    return rows

# ----- prompts (verbatim from the ensemble repo) -----
_TAIL = ("You MUST end your response with exactly this format: FINAL_ANSWER: [your numerical answer]\n"
         "Do not write placeholders. Write the actual number or expression as your answer.\n"
         "Do not ask follow-up questions. Do not invite further input. Produce a complete solution now.\n\n")
PROMPTS = {
 "chain_of_thought": lambda q: ("Solve the following problem by thinking step by step.\n"
    "Show each reasoning step clearly and in order.\n"+_TAIL+f"Problem: {q}\n\nStep-by-step solution:"),
 "formula_first": lambda q: ("Solve the following problem using a formula-first approach.\n"
    "First, write out every equation or formula you will need.\n"
    "Then substitute values and compute the result.\n"+_TAIL+f"Problem: {q}\n\nEquations and solution:"),
 "backward": lambda q: ("Solve the following problem using backward reasoning.\n"
    "Start by clearly stating what quantity the question asks for.\n"
    "Work backwards: what do you need to compute that? What do you need before that?\n"+_TAIL+f"Problem: {q}\n\nBackward reasoning:"),
}

# ----- extraction / scoring (verbatim from the ensemble repo) -----
_FA=re.compile(r"FINAL_ANSWER\s*:\s*(.+)",re.I); _PH=re.compile(r"^(<[^>]*>|\[[^\]]*\])$")
_NUM=re.compile(r"[-+]?\d[\d,.]*"); _NUMF=re.compile(r"[-+]?\d+(?:\.\d+)?"); _FRAC=re.compile(r"\\frac\{([^{}]+)\}\{([^{}]+)\}")
def extract_final_answer(text):
    m=_FA.search(text)
    if m:
        c=m.group(1).strip()
        if not _PH.match(c): return c
    nums=_NUM.findall(text)
    if nums: return nums[-1].replace(",","")
    lines=[l.strip() for l in text.splitlines() if l.strip()]
    return lines[-1] if lines else text.strip()
def _norm(t):
    t=t.strip(); b=re.search(r"\\boxed\{(.+)\}",t,re.DOTALL)
    if b: t=b.group(1)
    t=t.strip().lower(); t=re.sub(r"[$,%]","",t); t=re.sub(r"\s+"," ",t); return t.rstrip(".").strip()
def _f(t):
    try: return float(t)
    except ValueError: return None
def _fr(t):
    m=_FRAC.match(t.strip())
    if m:
        try: return float(m.group(1))/float(m.group(2))
        except (ValueError,ZeroDivisionError): return None
    return None
def is_correct(p,g):
    a,b=_norm(p),_norm(g)
    if a==b: return True
    af,bf=_f(a),_f(b)
    if af is not None and bf is not None: return abs(af-bf)<1e-6
    ar,br=_fr(a),_fr(b)
    if ar is not None and br is not None: return abs(ar-br)<1e-6
    if ar is not None and bf is not None: return abs(ar-bf)<1e-6
    if br is not None and af is not None: return abs(af-br)<1e-6
    nums=_NUMF.findall(a)
    if nums:
        lf=_f(nums[-1])
        if lf is not None and bf is not None: return abs(lf-bf)<1e-6
    return False
def summarize(rows):
    total=len(rows); correct=sum(r["correct"] for r in rows)
    tok=sum(int(r["token_cost"]) for r in rows); avg=tok/total if total else 0.0; acc=correct/total if total else 0.0
    s={"num_examples":total,"num_correct":correct,"accuracy":round(acc,4),"total_token_cost":tok,
       "avg_token_cost":round(avg,2),"nate":round((acc/avg)*1000,6) if avg else 0.0}
    if rows and rows[0].get("strategy_predictions"):
        strat=list(rows[0]["strategy_predictions"].keys())
        for st in strat:
            sc=sum(is_correct(r["strategy_predictions"].get(st,""),r.get("gold_answer","")) for r in rows)
            s[f"accuracy_{st}"]=round(sc/total,4)
        if len(strat)>1:
            s["disagreement_rate"]=round(sum(1 for r in rows if len(set(r["strategy_predictions"].values()))>1)/total,4)
            s["unique_correct_rate"]=round(sum(1 for r in rows if sum(is_correct(r["strategy_predictions"].get(st,""),r.get("gold_answer","")) for st in strat)==1)/total,4)
    return s

# ----- model -----
class Generator:
    def __init__(self, model_id, device, max_new_tokens=512):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch=torch; self.device=device; self.max_new_tokens=max_new_tokens
        self.tok=AutoTokenizer.from_pretrained(model_id, use_fast=True)
        if self.tok.pad_token is None: self.tok.pad_token=self.tok.eos_token
        self.model=AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, trust_remote_code=True)
        self.model.to(device)
    def generate(self, prompt):
        torch=self.torch
        maxlen=getattr(self.model.config,"max_position_embeddings",2048)
        enc=self.tok(prompt,return_tensors="pt",truncation=True,max_length=maxlen)
        enc={k:v.to(self.device) for k,v in enc.items()}
        with torch.inference_mode():
            out=self.model.generate(**enc,max_new_tokens=self.max_new_tokens,do_sample=False,
                                    pad_token_id=self.tok.pad_token_id,eos_token_id=self.tok.eos_token_id)
        ilen=enc["input_ids"].shape[1]; comp=out[0][ilen:]
        return {"text":self.tok.decode(comp,skip_special_tokens=True).strip(),
                "prompt_tokens":int(ilen),"completion_tokens":int(comp.shape[0])}

def run_model(model_id, slug, data, out_root, device, mnt):
    out_dir=os.path.join(out_root, slug); os.makedirs(out_dir, exist_ok=True)
    pred_path,summ_path=f"{out_dir}/predictions.json",f"{out_dir}/summary.json"
    rows,done=[],set()
    if os.path.exists(pred_path):
        try: rows=json.load(open(pred_path)); done={r["index"] for r in rows}; print(f"[{slug}] resume {len(done)}")
        except Exception: rows,done=[],set()
    print(f"===== LOADING {model_id} ({device}, fp16) =====")
    gen=Generator(model_id, device, mnt); t0=time.time()
    for idx,ex in enumerate(data):
        if idx in done: continue
        texts,preds,toks={}, {}, {}
        try:
            for st in STRATEGIES:
                o=gen.generate(PROMPTS[st](ex["question"]))
                texts[st]=o["text"]; preds[st]=extract_final_answer(o["text"]); toks[st]=o["prompt_tokens"]+o["completion_tokens"]
            final=Counter(preds.values()).most_common(1)[0][0]
            result={"strategy_texts":texts,"strategy_predictions":preds,"prediction":final,
                    "correct":is_correct(final,ex["answer"]),"token_cost":sum(toks.values()),
                    "per_strategy_tokens":toks,"aggregation":"majority_vote"}
        except Exception as e:
            print(f"[{slug}] {idx+1} ERROR {e}")
            result={"strategy_texts":{},"strategy_predictions":{},"prediction":"","correct":False,
                    "token_cost":0,"per_strategy_tokens":{},"aggregation":"error","error":str(e)}
        rows.append({"index":idx,"question":ex["question"],"gold_answer":ex["answer"],**result})
        if (idx+1)%5==0 or (idx+1)==len(data):
            rows.sort(key=lambda r:r["index"]); json.dump(rows,open(pred_path,"w"),indent=2,ensure_ascii=False)
            acc=sum(r["correct"] for r in rows)/len(rows)
            print(f"[{slug}] {idx+1}/{len(data)} acc={acc:.3f} {(idx+1-len(done))/max(time.time()-t0,1e-6):.3f}it/s")
    rows.sort(key=lambda r:r["index"]); json.dump(rows,open(pred_path,"w"),indent=2,ensure_ascii=False)
    json.dump(summarize(rows),open(summ_path,"w"),indent=2,ensure_ascii=False)
    print(f"[{slug}] DONE")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--n",type=int,default=300)
    ap.add_argument("--out-root",default="outputs_ensemble_n300")
    ap.add_argument("--device",default="mps")
    ap.add_argument("--max-new-tokens",type=int,default=512)
    a=ap.parse_args()
    data=load_svamp(a.n); print(f"Loaded {len(data)} SVAMP examples")
    for mid,slug in MODELS.items():
        run_model(mid,slug,data,a.out_root,a.device,a.max_new_tokens)
        try:
            import torch,gc; gc.collect()
            if torch.backends.mps.is_available(): torch.mps.empty_cache()
        except Exception: pass
    print("ALL LOCAL ENSEMBLE MODELS DONE")

if __name__=="__main__":
    main()
