"""Bootstrap + Wilson 95% CIs for accuracy on the A1 (N=300) local results.

For each <model>/<mode> it reads predictions.json, treats the per-sample
`correct` flags as the sample, and reports:
  - point accuracy
  - percentile bootstrap 95% CI (10k resamples, seeded)
  - Wilson score 95% CI (analytic cross-check)
  - avg token cost with its bootstrap 95% CI

Outputs a table to stdout and writes ci_summary.{csv,md} under the out root.
"""
from __future__ import annotations

import csv
import glob
import json
import os

import numpy as np

OUT_ROOT = "outputs_n300"
B = 10000
Z = 1.959963984540054  # 95%

ORDER = [("llama3.2", "Llama-3.2-3B"),
         ("gemma2-2b", "Gemma2-2B"),
         ("Deepseek-r1-1.5b", "DeepSeek-R1-1.5B"),
         ("qwen2.5_7b", "Qwen2.5-7B"),
         ("mistral-7b", "Mistral-7B")]
MODES = ["natural", "constrained", "structured"]


def wilson(k: int, n: int, z: float = Z) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (centre - half, centre + half)


def boot_ci(x: np.ndarray, rng: np.random.Generator, B: int = B) -> tuple[float, float]:
    n = len(x)
    idx = rng.integers(0, n, size=(B, n))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> None:
    rng = np.random.default_rng(42)
    rows_out = []
    print(f"{'Model':16s} {'Mode':12s} {'Acc%':>6s} {'Bootstrap 95% CI':>20s} "
          f"{'±':>5s} {'Wilson 95% CI':>18s} {'avgTok':>8s} {'tok 95% CI':>18s}")
    print("-" * 108)
    for slug, name in ORDER:
        for mode in MODES:
            p = f"{OUT_ROOT}/{slug}/{mode}/predictions.json"
            if not os.path.exists(p):
                continue
            data = json.load(open(p))
            correct = np.array([1 if r["correct"] else 0 for r in data], dtype=float)
            tokens = np.array([float(r.get("token_cost", 0)) for r in data], dtype=float)
            n = len(correct)
            k = int(correct.sum())
            acc = k / n
            blo, bhi = boot_ci(correct, rng)
            wlo, whi = wilson(k, n)
            half = (bhi - blo) / 2
            tavg = float(tokens.mean())
            tlo, thi = boot_ci(tokens, rng)
            print(f"{name:16s} {mode:12s} {acc*100:6.1f} "
                  f"[{blo*100:5.1f}, {bhi*100:5.1f}] {half*100:5.1f} "
                  f"[{wlo*100:5.1f}, {whi*100:5.1f}] {tavg:8.1f} [{tlo:6.0f}, {thi:6.0f}]")
            rows_out.append({
                "model": name, "mode": mode, "n": n, "correct": k,
                "acc": round(acc, 4),
                "boot_lo": round(blo, 4), "boot_hi": round(bhi, 4),
                "boot_halfwidth": round(half, 4),
                "wilson_lo": round(wlo, 4), "wilson_hi": round(whi, 4),
                "avg_tokens": round(tavg, 2),
                "tok_boot_lo": round(tlo, 1), "tok_boot_hi": round(thi, 1),
            })

    # CSV
    csv_path = os.path.join(OUT_ROOT, "ci_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    # Markdown (paper-ready)
    md_path = os.path.join(OUT_ROOT, "ci_summary.md")
    with open(md_path, "w") as f:
        f.write("| Model | Mode | Acc (%) | Bootstrap 95% CI | Wilson 95% CI | Avg tokens |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in rows_out:
            f.write(f"| {r['model']} | {r['mode']} | {r['acc']*100:.1f} "
                    f"| [{r['boot_lo']*100:.1f}, {r['boot_hi']*100:.1f}] "
                    f"| [{r['wilson_lo']*100:.1f}, {r['wilson_hi']*100:.1f}] "
                    f"| {r['avg_tokens']:.1f} |\n")
    print(f"\nWrote {csv_path} and {md_path}")


if __name__ == "__main__":
    main()
