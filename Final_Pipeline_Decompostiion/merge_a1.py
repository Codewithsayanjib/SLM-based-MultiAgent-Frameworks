"""Merge A1 result folders (from the 5 Colab/Drive accounts) into one outputs_n300/.

Each Colab run writes  <root>/<slug>/<mode>/predictions.json  (+ summary.json).
This script scans one or more source roots (extracted zips or synced Drive folders),
unions their predictions by `index` for each (slug, mode), recomputes summary.json,
and writes the merged tree into the target (default: ./outputs_n300).

Usage:
    python merge_a1.py ~/Downloads/qwen2.5_7b_natural_n300 ~/Downloads/mistral-7b_constrained_n300 ...
    # each source may itself be an outputs_n300 root, a single <slug>/<mode> tree,
    # or a parent folder — the script finds every predictions.json beneath it.

Disjoint indices (e.g. sharded 0-149 / 150-299) merge cleanly; overlapping indices
keep the last one seen and a warning is printed.
"""
from __future__ import annotations

import glob
import json
import os
import sys

from slm_comm.metrics import summarize_results

TARGET_DEFAULT = "outputs_n300"


def find_prediction_files(root: str) -> list[str]:
    return sorted(glob.glob(os.path.join(root, "**", "predictions.json"), recursive=True))


def slug_mode_from_path(pred_path: str) -> tuple[str, str] | None:
    """.../<slug>/<mode>/predictions.json -> (slug, mode)."""
    parts = os.path.normpath(pred_path).split(os.sep)
    if len(parts) >= 3 and parts[-1] == "predictions.json":
        return parts[-3], parts[-2]
    return None


def main() -> None:
    sources = sys.argv[1:]
    if not sources:
        print(__doc__)
        sys.exit(1)

    target = os.environ.get("A1_TARGET", TARGET_DEFAULT)
    # (slug, mode) -> {index -> row}
    merged: dict[tuple[str, str], dict[int, dict]] = {}

    for src in sources:
        for pred in find_prediction_files(src):
            sm = slug_mode_from_path(pred)
            if not sm:
                print(f"  skip (unrecognised path): {pred}")
                continue
            try:
                rows = json.load(open(pred, encoding="utf-8"))
            except Exception as exc:
                print(f"  skip (unreadable {exc}): {pred}")
                continue
            bucket = merged.setdefault(sm, {})
            added = 0
            for r in rows:
                idx = r["index"]
                if idx in bucket and bucket[idx].get("correct") != r.get("correct"):
                    print(f"  ! overlap {sm} idx={idx}: differing rows, keeping newest source")
                bucket[idx] = r
                added += 1
            print(f"  {sm[0]}/{sm[1]}: +{added} rows from {pred}")

    if not merged:
        print("No predictions.json found under given sources.")
        sys.exit(1)

    print("\n=== merged summaries ===")
    for (slug, mode), bucket in sorted(merged.items()):
        rows = [bucket[i] for i in sorted(bucket)]
        out_dir = os.path.join(target, slug, mode)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "predictions.json"), "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
        summ = summarize_results(rows)
        with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summ, f, indent=2, ensure_ascii=False)
        print(f"{slug:16s} {mode:12s} n={summ['num_examples']:4d} "
              f"acc={summ['accuracy']*100:5.1f}%  avg_tok={summ['avg_token_cost']:.1f}")
    print(f"\nWrote merged tree to {target}/")


if __name__ == "__main__":
    main()
