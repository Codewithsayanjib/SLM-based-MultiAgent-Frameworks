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

    # LaTeX fraction comparison
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

    summary: dict[str, Any] = {
        "num_examples": total,
        "num_correct": correct,
        "accuracy": round(accuracy, 4),
        "total_token_cost": token_cost,
        "avg_token_cost": round(avg_tokens, 2),
        "nate": round((accuracy / avg_tokens) * 1000, 6) if avg_tokens else 0.0,
    }

    # Per-strategy accuracy (ensemble runs only)
    if rows and "strategy_predictions" in rows[0] and rows[0]["strategy_predictions"]:
        strategies = list(rows[0]["strategy_predictions"].keys())
        for s in strategies:
            strat_correct = sum(
                1 for row in rows
                if is_correct(
                    row["strategy_predictions"].get(s, ""),
                    row.get("gold_answer", ""),
                )
            )
            summary[f"accuracy_{s}"] = round(strat_correct / total, 4)
            strat_tokens = sum(row["per_strategy_tokens"].get(s, 0) for row in rows)
            avg_s = strat_tokens / total if total else 0.0
            strat_acc = strat_correct / total if total else 0.0
            summary[f"nate_{s}"] = round((strat_acc / avg_s) * 1000, 6) if avg_s else 0.0

        # Disagreement rate: fraction of examples where not all strategies agree
        if len(strategies) > 1:
            disagreements = sum(
                1 for row in rows
                if len(set(row["strategy_predictions"].values())) > 1
            )
            summary["disagreement_rate"] = round(disagreements / total, 4)

            # Unique-correct rate: fraction of examples where only one strategy is correct
            # High value → diversity is genuinely useful (errors are uncorrelated)
            unique_correct = sum(
                1 for row in rows
                if sum(
                    is_correct(row["strategy_predictions"].get(s, ""), row.get("gold_answer", ""))
                    for s in strategies
                ) == 1
            )
            summary["unique_correct_rate"] = round(unique_correct / total, 4)

    return summary
