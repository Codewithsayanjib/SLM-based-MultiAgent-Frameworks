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
        ds = load_dataset("nlile/hendrycks-MATH-benchmark", "default", split=cfg.split)
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
            answer = (
                raw_answer.strip()
                if raw_answer.strip()
                else _extract_boxed_answer(str(item["solution"]))
            )
            rows.append(
                {
                    "question": str(item["problem"]),
                    "answer": answer,
                }
            )
        return rows

    if cfg.name == "svamp":
        # SVAMP: Simple Variations on Arithmetic Math word Problems
        # HuggingFace repo: "ChilleD/SVAMP", split is typically "train" (no official test split)
        # Fields: Body (str), Question (str), Answer (int/float), Equation (str), Type (str)
        ds = load_dataset("ChilleD/SVAMP", split=cfg.split)
        ds = ds.shuffle(seed=seed)
        rows = []
        for item in ds.select(range(min(cfg.max_samples, len(ds)))):
            # Combine Body and Question into a single natural-language problem statement
            body = str(item.get("Body", "")).strip()
            question = str(item.get("Question", "")).strip()
            full_question = f"{body} {question}".strip()

            # Answer is a numeric value; normalise to a clean string
            raw_answer = item.get("Answer", "")
            if isinstance(raw_answer, float) and raw_answer.is_integer():
                answer_str = str(int(raw_answer))
            else:
                answer_str = str(raw_answer).strip()

            rows.append(
                {
                    "question": full_question,
                    "answer": answer_str,
                }
            )
        return rows

    raise ValueError(f"Unsupported dataset: {cfg.name}")