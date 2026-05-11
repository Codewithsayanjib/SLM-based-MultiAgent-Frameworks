from __future__ import annotations

import re
from typing import Any


def normalize_answer(text: str) -> str:
    text = text.strip()
    # Unwrap \boxed{...} — model may echo it in FINAL_ANSWER
    boxed = re.search(r"\\boxed\{(.+)\}", text, re.DOTALL)
    if boxed:
        text = boxed.group(1)
    text = text.strip().lower()
    text = re.sub(r"[$,%]", "", text)
    text = re.sub(r"\s+", " ", text)
    text = text.rstrip(".")
    # Strip trailing .0 / .00 etc. so "3.0" == "3", "12.00" == "12"
    text = re.sub(r"\.0+$", "", text)
    # Strip thousands-separator commas in numbers: "1,000" -> "1000"
    text = re.sub(r"(\d),(\d)", r"\1\2", text)
    return text.strip()


_NUMBER_RE_METRIC = re.compile(r"[-+]?\d+(?:\.\d+)?")
_FRAC_RE = re.compile(r"\\frac\{([^{}]+)\}\{([^{}]+)\}")


def _try_frac(text: str) -> float | None:
    """Parse \\frac{a}{b} → float, or None if not matched."""
    m = _FRAC_RE.match(text.strip())
    if m:
        try:
            return float(m.group(1)) / float(m.group(2))
        except (ValueError, ZeroDivisionError):
            return None
    return None


def _try_float(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def is_correct(prediction: str, gold: str) -> bool:
    pred_norm = normalize_answer(prediction)
    gold_norm = normalize_answer(gold)

    if pred_norm == gold_norm:
        return True

    pred_f = _try_float(pred_norm)
    gold_f = _try_float(gold_norm)
    if pred_f is not None and gold_f is not None:
        return abs(pred_f - gold_f) < 1e-6

    # LaTeX fraction comparison (retained for MATH compatibility)
    pred_frac = _try_frac(pred_norm)
    gold_frac = _try_frac(gold_norm)
    if pred_frac is not None and gold_frac is not None:
        return abs(pred_frac - gold_frac) < 1e-6
    if pred_frac is not None and gold_f is not None:
        return abs(pred_frac - gold_f) < 1e-6
    if gold_frac is not None and pred_f is not None:
        return abs(pred_f - gold_frac) < 1e-6

    # Last-number fallback
    numbers = _NUMBER_RE_METRIC.findall(pred_norm)
    if numbers:
        last_f = _try_float(numbers[-1])
        if last_f is not None and gold_f is not None:
            return abs(last_f - gold_f) < 1e-6

    return False


def summarize_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    correct = sum(1 for row in rows if row["correct"])
    token_cost = sum(int(row["token_cost"]) for row in rows)
    avg_tokens = token_cost / total if total else 0.0
    accuracy = correct / total if total else 0.0
    return {
        "num_examples": total,
        "num_correct": correct,
        "accuracy": round(accuracy, 4),
        "total_token_cost": token_cost,
        "avg_token_cost": round(avg_tokens, 2),
        "accuracy_per_1000_tokens": round((accuracy / avg_tokens) * 1000, 6) if avg_tokens else 0.0,
    }