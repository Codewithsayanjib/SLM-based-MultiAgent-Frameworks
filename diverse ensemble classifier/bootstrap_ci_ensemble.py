"""Bootstrap + Wilson 95% CIs for the diverse-ensemble (Methodology 2) results.

For each model it reads outputs_ensemble_n300/<slug>/predictions.json and reports:
  - ensemble accuracy with percentile bootstrap 95% CI (10k resamples, seeded)
    and Wilson score 95% CI (analytic cross-check)
  - avg token cost with bootstrap 95% CI
  - per-strategy accuracy and the diversity stats from summary.json

Writes ci_summary_ensemble.{md,csv} under the out root.
"""
from __future__ import annotations

import csv
import json
import os

import numpy as np

OUT_ROOT = "outputs_ensemble_n300"
B = 10000
Z = 1.959963984540054

ORDER = [("llama3.2", "Llama-3.2-3B"),
         ("gemma2-2b", "Gemma2-2B"),
         ("mistral-7b", "Mistral-7B"),
         ("Deepseek-r1-1.5b", "DeepSeek-R1-1.5B"),
         ("qwen2.5_7b", "Qwen2.5-7B")]


def wilson(k, n, z=Z):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return (c - h, c + h)


def boot(x, rng, B=B):
    n = len(x)
    idx = rng.integers(0, n, size=(B, n))
    m = x[idx].mean(axis=1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    rng = np.random.default_rng(42)
    rows_out = []
    print(f"{'Model':16s} {'Acc%':>6} {'Bootstrap 95% CI':>18} {'±':>5} "
          f"{'Wilson 95% CI':>18} {'T̄':>8} {'CoT':>5} {'Form':>5} {'Back':>5}")
    print("-" * 96)
    for slug, name in ORDER:
        p = f"{OUT_ROOT}/{slug}/predictions.json"
        if not os.path.exists(p):
            continue
        data = json.load(open(p))
        summ = json.load(open(f"{OUT_ROOT}/{slug}/summary.json"))
        correct = np.array([1 if r["correct"] else 0 for r in data], float)
        tokens = np.array([float(r.get("token_cost", 0)) for r in data], float)
        n = len(correct); k = int(correct.sum()); acc = k / n
        blo, bhi = boot(correct, rng); wlo, whi = wilson(k, n)
        tlo, thi = boot(tokens, rng)
        cot = summ.get("accuracy_chain_of_thought", 0) * 100
        form = summ.get("accuracy_formula_first", 0) * 100
        back = summ.get("accuracy_backward", 0) * 100
        print(f"{name:16s} {acc*100:6.1f} [{blo*100:5.1f}, {bhi*100:5.1f}] {(bhi-blo)/2*100:5.1f} "
              f"[{wlo*100:5.1f}, {whi*100:5.1f}] {tokens.mean():8.1f} {cot:5.1f} {form:5.1f} {back:5.1f}")
        rows_out.append({
            "model": name, "n": n, "ensemble_acc": round(acc, 4),
            "boot_lo": round(blo, 4), "boot_hi": round(bhi, 4), "boot_halfwidth": round((bhi-blo)/2, 4),
            "wilson_lo": round(wlo, 4), "wilson_hi": round(whi, 4),
            "avg_tokens": round(float(tokens.mean()), 2),
            "tok_boot_lo": round(tlo, 1), "tok_boot_hi": round(thi, 1),
            "acc_cot": round(cot/100, 4), "acc_formula_first": round(form/100, 4),
            "acc_backward": round(back/100, 4),
            "disagreement_rate": summ.get("disagreement_rate"),
            "unique_correct_rate": summ.get("unique_correct_rate"),
        })

    with open(f"{OUT_ROOT}/ci_summary_ensemble.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys())); w.writeheader(); w.writerows(rows_out)
    with open(f"{OUT_ROOT}/ci_summary_ensemble.md", "w") as f:
        f.write("| Model | Ensemble Acc (%) | Bootstrap 95% CI | Wilson 95% CI | Avg tokens | CoT | Formula | Backward | Disagr | UniqCorrect |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rows_out:
            f.write(f"| {r['model']} | {r['ensemble_acc']*100:.1f} "
                    f"| [{r['boot_lo']*100:.1f}, {r['boot_hi']*100:.1f}] "
                    f"| [{r['wilson_lo']*100:.1f}, {r['wilson_hi']*100:.1f}] "
                    f"| {r['avg_tokens']:.1f} | {r['acc_cot']*100:.1f} | {r['acc_formula_first']*100:.1f} "
                    f"| {r['acc_backward']*100:.1f} | {r['disagreement_rate']*100:.1f}% | {r['unique_correct_rate']*100:.1f}% |\n")
    print(f"\nWrote {OUT_ROOT}/ci_summary_ensemble.{{md,csv}}")


if __name__ == "__main__":
    main()
