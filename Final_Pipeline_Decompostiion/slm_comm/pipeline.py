from __future__ import annotations

from dataclasses import asdict
from typing import Any

from slm_comm.agents import PlannerAgent, SolverAgent, VerifierAgent, extract_final_answer
from slm_comm.communication import build_communication
from slm_comm.config import AppConfig
from slm_comm.metrics import is_correct
from slm_comm.model import HuggingFaceTextGenerator


class BasePipeline:
    def run(self, example: dict[str, str]) -> dict[str, Any]:
        raise NotImplementedError


class SolverOnlyPipeline(BasePipeline):
    def __init__(self, model: HuggingFaceTextGenerator):
        self.solver = SolverAgent(model)

    def run(self, example: dict[str, str]) -> dict[str, Any]:
        solver = self.solver.run(example["question"])
        pred = extract_final_answer(solver.text)
        cost = solver.prompt_tokens + solver.completion_tokens
        return {
            "planner_text": None,
            "communication": None,
            "solver_text": solver.text,
            "verifier_text": None,
            "prediction": pred,
            "correct": is_correct(pred, example["answer"]),
            "token_cost": cost,
            "planner_tokens": 0,
            "solver_tokens": solver.prompt_tokens + solver.completion_tokens,
            "verifier_tokens": 0,
        }


class PlannerSolverPipeline(BasePipeline):
    def __init__(self, cfg: AppConfig, model: HuggingFaceTextGenerator):
        self.planner = PlannerAgent(model)
        self.solver = SolverAgent(model)
        self.communication = build_communication(cfg.pipeline.communication, cfg.communication, model.tokenizer)

    def run(self, example: dict[str, str]) -> dict[str, Any]:
        planner = self.planner.run(example["question"])
        comm = self.communication(planner.text)
        solver = self.solver.run(example["question"], comm)
        pred = extract_final_answer(solver.text)
        cost = sum([
            planner.prompt_tokens,
            planner.completion_tokens,
            solver.prompt_tokens,
            solver.completion_tokens,
        ])
        return {
            "planner_text": planner.text,
            "communication": comm,
            "solver_text": solver.text,
            "verifier_text": None,
            "prediction": pred,
            "correct": is_correct(pred, example["answer"]),
            "token_cost": cost,
            "planner_tokens": planner.prompt_tokens + planner.completion_tokens,
            "solver_tokens": solver.prompt_tokens + solver.completion_tokens,
            "verifier_tokens": 0,
        }


class PlannerSolverVerifierPipeline(BasePipeline):
    def __init__(self, cfg: AppConfig, model: HuggingFaceTextGenerator):
        self.planner = PlannerAgent(model)
        self.solver = SolverAgent(model)
        self.verifier = VerifierAgent(model)
        self.communication = build_communication(cfg.pipeline.communication, cfg.communication, model.tokenizer)

    def run(self, example: dict[str, str]) -> dict[str, Any]:
        planner = self.planner.run(example["question"])
        comm = self.communication(planner.text)
        solver = self.solver.run(example["question"], comm)
        verifier = self.verifier.run(example["question"], solver.text)
        pred = extract_final_answer(verifier.text)
        cost = sum([
            planner.prompt_tokens,
            planner.completion_tokens,
            solver.prompt_tokens,
            solver.completion_tokens,
            verifier.prompt_tokens,
            verifier.completion_tokens,
        ])
        return {
            "planner_text": planner.text,
            "communication": comm,
            "solver_text": solver.text,
            "verifier_text": verifier.text,
            "prediction": pred,
            "correct": is_correct(pred, example["answer"]),
            "token_cost": cost,
            "planner_tokens": planner.prompt_tokens + planner.completion_tokens,
            "solver_tokens": solver.prompt_tokens + solver.completion_tokens,
            "verifier_tokens": verifier.prompt_tokens + verifier.completion_tokens,
        }



def build_pipeline(cfg: AppConfig, model: HuggingFaceTextGenerator) -> BasePipeline:
    if cfg.pipeline.type == "solver":
        return SolverOnlyPipeline(model)
    if cfg.pipeline.type == "planner_solver":
        return PlannerSolverPipeline(cfg, model)
    if cfg.pipeline.type == "planner_solver_verifier":
        return PlannerSolverVerifierPipeline(cfg, model)
    raise ValueError(f"Unsupported pipeline type: {cfg.pipeline.type}")
