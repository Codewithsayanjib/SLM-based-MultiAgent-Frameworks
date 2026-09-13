"""A1 sweep driver: run the planner-solver-verifier pipeline across multiple
models and all three communication modes at a larger sample count (N=500).

Design goals:
- Load each model ONCE and reuse it across the three communication modes
  (natural / constrained / structured) to avoid repeated model loading.
- Deterministic: uses the same seed-42 shuffle as the published 100-sample runs,
  so the first 100 samples of a 500-sample run reproduce the paper's Table II.
- Resumable: if a run is interrupted, re-running skips already-completed indices
  by reading the existing predictions.json.
- Non-destructive: writes to a separate output root (default: outputs_500/) so the
  published 100-sample outputs/ are never overwritten.

Communication budget follows the committed config.yaml (max_tokens=120, num_steps=5),
which is what produced the published results.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from slm_comm.config import (
    AppConfig,
    CommunicationConfig,
    DatasetConfig,
    GenerationConfig,
    ModelConfig,
    PipelineConfig,
)
from slm_comm.data import load_dataset_records
from slm_comm.model import HuggingFaceTextGenerator
from slm_comm.metrics import summarize_results
from slm_comm.pipeline import build_pipeline
from slm_comm.utils import ensure_dir, save_json, set_seed


# model_id -> output-folder slug
DEFAULT_MODELS = {
    "meta-llama/Llama-3.2-3B-Instruct": "llama3.2",
    "google/gemma-2-2b-it": "gemma2-2b",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": "Deepseek-r1-1.5b",
}
LARGE_MODELS = {
    "Qwen/Qwen2.5-7B-Instruct": "qwen2.5_7b",
    "mistralai/Mistral-7B-Instruct-v0.3": "mistral-7b",
}
MODES = ["natural", "constrained", "structured"]


def make_config(model_id: str, mode: str, n: int, out_dir: str, device: str, dtype: str) -> AppConfig:
    return AppConfig(
        seed=42,
        output_dir=out_dir,
        dataset=DatasetConfig(name="svamp", split="train", max_samples=n),
        model=ModelConfig(name=model_id, device=device, dtype=dtype, load_in_4bit=False),
        pipeline=PipelineConfig(type="planner_solver_verifier", communication=mode),
        communication=CommunicationConfig(max_tokens=120, num_steps=5),
        generation=GenerationConfig(max_new_tokens=512, temperature=0.0, top_p=1.0, do_sample=False),
    )


def load_done(pred_path: str) -> tuple[list, set]:
    """Return (existing_rows, set_of_done_indices) if a predictions file exists."""
    if os.path.exists(pred_path):
        try:
            rows = json.load(open(pred_path, encoding="utf-8"))
            done = {r["index"] for r in rows}
            return rows, done
        except Exception:
            return [], set()
    return [], set()


def run_one(model, cfg: AppConfig, dataset, pred_path: str, summ_path: str, tag: str) -> None:
    ensure_dir(os.path.dirname(pred_path))
    rows, done = load_done(pred_path)
    if done:
        print(f"[{tag}] resuming: {len(done)} already done")
    pipeline = build_pipeline(cfg, model)

    t0 = time.time()
    for idx, example in enumerate(dataset):
        if idx in done:
            continue
        try:
            result = pipeline.run(example)
        except Exception as exc:  # keep the sweep alive on a single-example failure
            print(f"[{tag}] [{idx + 1}/{len(dataset)}] ERROR: {exc}")
            result = {
                "planner_text": None, "communication": None, "solver_text": None,
                "verifier_text": None, "prediction": "", "correct": False,
                "token_cost": 0, "planner_tokens": 0, "solver_tokens": 0,
                "verifier_tokens": 0, "error": str(exc),
            }
        rows.append({"index": idx, "question": example["question"],
                     "gold_answer": example["answer"], **result})
        acc = sum(1 for r in rows if r["correct"]) / len(rows)
        rate = (idx + 1 - len(done)) / max(time.time() - t0, 1e-6)
        print(f"[{tag}] [{idx + 1}/{len(dataset)}] correct={result['correct']} "
              f"tok={result['token_cost']} run_acc={acc:.3f} {rate:.2f}it/s")
        if (idx + 1) % 5 == 0 or (idx + 1) == len(dataset):
            rows.sort(key=lambda r: r["index"])
            save_json(pred_path, rows)

    rows.sort(key=lambda r: r["index"])
    save_json(pred_path, rows)
    save_json(summ_path, summarize_results(rows))
    print(f"[{tag}] DONE -> {summ_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out-root", type=str, default="outputs_n300")
    ap.add_argument("--device", type=str, default="mps")
    ap.add_argument("--dtype", type=str, default="float16")
    ap.add_argument("--models", type=str, default="local",
                    help="'local' (3 small), 'large' (2 big), or comma-separated model ids")
    ap.add_argument("--modes", type=str, default=",".join(MODES))
    args = ap.parse_args()

    if args.models == "local":
        models = DEFAULT_MODELS
    elif args.models == "large":
        models = LARGE_MODELS
    else:
        allid = {**DEFAULT_MODELS, **LARGE_MODELS}
        models = {m: allid.get(m, m.split("/")[-1]) for m in args.models.split(",")}
    modes = args.modes.split(",")

    set_seed(42)
    # Dataset is identical across all models/modes (seed-42 shuffle, first N).
    ds_cfg = DatasetConfig(name="svamp", split="train", max_samples=args.n)
    dataset = load_dataset_records(ds_cfg, seed=42)
    print(f"Loaded {len(dataset)} SVAMP examples (N requested={args.n})")

    for model_id, slug in models.items():
        print(f"\n===== LOADING {model_id} =====")
        cfg0 = make_config(model_id, modes[0], args.n, "tmp", args.device, args.dtype)
        model = HuggingFaceTextGenerator(cfg0.model, cfg0.generation)
        for mode in modes:
            out_dir = os.path.join(args.out_root, slug, mode)
            cfg = make_config(model_id, mode, args.n, out_dir, args.device, args.dtype)
            run_one(model, cfg, dataset,
                    os.path.join(out_dir, "predictions.json"),
                    os.path.join(out_dir, "summary.json"),
                    tag=f"{slug}/{mode}")
        del model
        try:
            import torch, gc
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass

    # Sentinel: every requested model/mode has a summary.json -> full sweep done.
    all_done = all(
        os.path.exists(os.path.join(args.out_root, slug, mode, "summary.json"))
        for slug in models.values() for mode in modes
    )
    if all_done:
        with open(os.path.join(args.out_root, ".ALL_DONE"), "w") as f:
            f.write("done\n")
        print("ALL_DONE: every model/mode has a summary.json")


if __name__ == "__main__":
    main()
