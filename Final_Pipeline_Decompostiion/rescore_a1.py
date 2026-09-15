"""Re-score stored A1 predictions with the corrected verifier->solver fallback
extraction (agents.extract_verified_answer). No model re-running: it re-reads the
saved solver_text / verifier_text in each predictions.json, recomputes the
`prediction` and `correct` fields, and rewrites predictions.json + summary.json.

Usage:  python rescore_a1.py [out_root]   (default: outputs_n300)
"""
from __future__ import annotations

import glob
import json
import os
import sys

from slm_comm.agents import extract_verified_answer
from slm_comm.metrics import is_correct, summarize_results


def main() -> None:
    root = sys.argv[1] if len(sys.argv) > 1 else "outputs_n300"
    preds = sorted(glob.glob(os.path.join(root, "*", "*", "predictions.json")))
    print(f"Re-scoring {len(preds)} files under {root}/\n")
    print(f"{'model/mode':32s} {'old':>6} {'new':>6} {'Δ':>6}")
    print("-" * 54)
    for p in preds:
        rows = json.load(open(p, encoding="utf-8"))
        old = sum(1 for r in rows if r["correct"]) / len(rows)
        for r in rows:
            pred = extract_verified_answer(r.get("verifier_text") or "", r.get("solver_text") or "")
            r["prediction"] = pred
            r["correct"] = is_correct(pred, r["gold_answer"])
        with open(p, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
        summ = summarize_results(rows)
        with open(os.path.join(os.path.dirname(p), "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summ, f, indent=2, ensure_ascii=False)
        new = summ["accuracy"]
        tag = os.path.dirname(p).replace(root + "/", "")
        print(f"{tag:32s} {old*100:5.1f}% {new*100:5.1f}% {(new-old)*100:+5.1f}")
    print(f"\nDone. Rewrote predictions.json + summary.json under {root}/")


if __name__ == "__main__":
    main()
