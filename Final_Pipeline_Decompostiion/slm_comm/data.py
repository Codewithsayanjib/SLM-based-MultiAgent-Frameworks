from __future__ import annotations

import re
from typing import Any

from datasets import load_dataset

from slm_comm.config import DatasetConfig
from slm_comm.utils import extract_gsm8k_answer


def _extract_boxed_answer(solution: str) -> str:
    """Extract content of the last \\boxed{} in a MATH solution string."""
    idx = solution.rfind(r"\boxed{")
    if idx == -1:
        return solution.strip()
    start = idx + len(r"\boxed{")
    depth = 1
    pos = start
    while pos < len(solution) and depth > 0:
        if solution[pos] == "{":
            depth += 1
        elif solution[pos] == "}":
            depth -= 1
        pos += 1
    return solution[start : pos - 1].strip()


def load_dataset_records(cfg: DatasetConfig, seed: int = 42) -> list[dict[str, str]]:
    if cfg.name == "gsm8k":
        ds = load_dataset("gsm8k", "main", split=cfg.split)
        ds = ds.shuffle(seed=seed)
        rows: list[dict[str, str]] = []
        for item in ds.select(range(min(cfg.max_samples, len(ds)))):
            rows.append(
                {
                    "question": str(item["question"]),
                    "answer": extract_gsm8k_answer(str(item["answer"])),
                }
            )
        return rows

    if cfg.name == "math":
        ds = load_dataset("nlile/hendrycks-MATH-benchmark", split=cfg.split)
        subject_lower = cfg.subject.lower()

        def _keep(ex: dict) -> bool:
            if ex.get("subject", "").lower() != subject_lower:
                return False
            level = ex.get("level")
            if level is None:
                return False
            return cfg.level_min <= int(level) <= cfg.level_max

        ds = ds.filter(_keep)
        ds = ds.shuffle(seed=seed)
        rows = []
        for item in ds.select(range(min(cfg.max_samples, len(ds)))):
            raw_answer = item.get("answer") or ""
            answer = raw_answer.strip() if raw_answer.strip() else _extract_boxed_answer(str(item["solution"]))
            rows.append(
                {
                    "question": str(item["problem"]),
                    "answer": answer,
                }
            )
        return rows

    if cfg.name == "svamp":
        # SVAMP has no official train/test split — the full 1000-sample set is
        # treated as a benchmark. ChilleD/SVAMP exposes it under split="train".
        ds = load_dataset("ChilleD/SVAMP", split="train")
        ds = ds.shuffle(seed=seed)
        rows = []
        for item in ds.select(range(min(cfg.max_samples, len(ds)))):
            # Body contains the story context (which may include distractor info).
            # Question is the actual question sentence.
            # Concatenating both gives the full noisy problem statement the
            # planner must filter before passing to the solver.
            body = str(item.get("Body", "")).strip()
            question = str(item.get("Question", "")).strip()
            full_question = f"{body} {question}".strip()

            # Answer is always a plain number (int or float).
            # Normalise to string; trailing .0 stripped in metrics.py.
            answer = str(item["Answer"]).strip()

            rows.append({"question": full_question, "answer": answer})
        return rows

    raise ValueError(f"Unsupported dataset: {cfg.name!r}. Choose from: gsm8k, math, svamp")