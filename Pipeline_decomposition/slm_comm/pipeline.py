from __future__ import annotations

from collections import Counter
from typing import Any

from slm_comm.agents import ChainOfThoughtSolver, STRATEGY_REGISTRY, extract_final_answer
from slm_comm.config import AppConfig
from slm_comm.metrics import is_correct
from slm_comm.model import HuggingFaceTextGenerator


class BasePipeline:
    def run(self, example: dict[str, str]) -> dict[str, Any]:
        raise NotImplementedError


class SolverOnlyPipeline(BasePipeline):
    """Single chain-of-thought solver — the baseline."""

    def __init__(self, model: HuggingFaceTextGenerator):
        self.solver = ChainOfThoughtSolver(model)

    def run(self, example: dict[str, str]) -> dict[str, Any]:
        out = self.solver.run(example["question"])
        pred = extract_final_answer(out.text)
        cost = out.prompt_tokens + out.completion_tokens
        return {
            "strategy_texts": {"chain_of_thought": out.text},
            "strategy_predictions": {"chain_of_thought": pred},
            "prediction": pred,
            "correct": is_correct(pred, example["answer"]),
            "token_cost": cost,
            "per_strategy_tokens": {"chain_of_thought": cost},
            "aggregation": "single",
        }


class DiverseEnsemblePipeline(BasePipeline):
    """Run N strategy-diverse solvers, aggregate by majority vote."""

    def __init__(self, cfg: AppConfig, model: HuggingFaceTextGenerator):
        strategies = cfg.ensemble.strategies
        unknown = set(strategies) - set(STRATEGY_REGISTRY)
        if unknown:
            raise ValueError(f"Unknown strategies: {unknown}. Valid: {set(STRATEGY_REGISTRY)}")
        self.solvers = {s: STRATEGY_REGISTRY[s](model) for s in strategies}

    def run(self, example: dict[str, str]) -> dict[str, Any]:
        question = example["question"]
        texts: dict[str, str] = {}
        predictions: dict[str, str] = {}
        tokens: dict[str, int] = {}

        for strategy, solver in self.solvers.items():
            out = solver.run(question)
            pred = extract_final_answer(out.text)
            texts[strategy] = out.text
            predictions[strategy] = pred
            tokens[strategy] = out.prompt_tokens + out.completion_tokens

        # Majority vote: most common prediction wins; tie → first strategy wins
        vote_counts = Counter(predictions.values())
        final_pred, _ = vote_counts.most_common(1)[0]

        return {
            "strategy_texts": texts,
            "strategy_predictions": predictions,
            "prediction": final_pred,
            "correct": is_correct(final_pred, example["answer"]),
            "token_cost": sum(tokens.values()),
            "per_strategy_tokens": tokens,
            "aggregation": "majority_vote",
        }


def build_pipeline(cfg: AppConfig, model: HuggingFaceTextGenerator) -> BasePipeline:
    if cfg.pipeline.type == "solver":
        return SolverOnlyPipeline(model)
    if cfg.pipeline.type == "diverse_ensemble":
        return DiverseEnsemblePipeline(cfg, model)
    raise ValueError(f"Unsupported pipeline type: {cfg.pipeline.type!r}")