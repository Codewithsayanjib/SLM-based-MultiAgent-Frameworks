from __future__ import annotations

import re
from dataclasses import dataclass

from slm_comm.model import HuggingFaceTextGenerator


@dataclass
class AgentOutput:
    text: str
    prompt_tokens: int
    completion_tokens: int


class BaseAgent:
    strategy: str = "base"

    def __init__(self, model: HuggingFaceTextGenerator):
        self.model = model

    def _call(self, prompt: str) -> AgentOutput:
        out = self.model.generate(prompt)
        return AgentOutput(
            text=out["text"],
            prompt_tokens=out["prompt_tokens"],
            completion_tokens=out["completion_tokens"],
        )

    def run(self, question: str) -> AgentOutput:
        raise NotImplementedError


class ChainOfThoughtSolver(BaseAgent):
    """Solve by writing out every reasoning step explicitly."""
    strategy = "chain_of_thought"

    def run(self, question: str) -> AgentOutput:
        prompt = (
            "Solve the following problem by thinking step by step.\n"
            "Show each reasoning step clearly and in order.\n"
            "You MUST end your response with exactly this format: FINAL_ANSWER: [your numerical answer]\n"
            "Do not write placeholders. Write the actual number or expression as your answer.\n"
            "Do not ask follow-up questions. Do not invite further input. Produce a complete solution now.\n\n"
            f"Problem: {question}\n\n"
            "Step-by-step solution:"
        )
        return self._call(prompt)


class FormulaFirstSolver(BaseAgent):
    """Solve by identifying all relevant formulas/equations before computing."""
    strategy = "formula_first"

    def run(self, question: str) -> AgentOutput:
        prompt = (
            "Solve the following problem using a formula-first approach.\n"
            "First, write out every equation or formula you will need.\n"
            "Then substitute values and compute the result.\n"
            "You MUST end your response with exactly this format: FINAL_ANSWER: [your numerical answer]\n"
            "Do not write placeholders. Write the actual number or expression as your answer.\n"
            "Do not ask follow-up questions. Do not invite further input. Produce a complete solution now.\n\n"
            f"Problem: {question}\n\n"
            "Equations and solution:"
        )
        return self._call(prompt)


class BackwardSolver(BaseAgent):
    """Solve by starting from what the answer must look like and working backwards."""
    strategy = "backward"

    def run(self, question: str) -> AgentOutput:
        prompt = (
            "Solve the following problem using backward reasoning.\n"
            "Start by clearly stating what quantity the question asks for.\n"
            "Work backwards: what do you need to compute that? What do you need before that?\n"
            "You MUST end your response with exactly this format: FINAL_ANSWER: [your numerical answer]\n"
            "Do not write placeholders. Write the actual number or expression as your answer.\n"
            "Do not ask follow-up questions. Do not invite further input. Produce a complete solution now.\n\n"
            f"Problem: {question}\n\n"
            "Backward reasoning:"
        )
        return self._call(prompt)


STRATEGY_REGISTRY: dict[str, type[BaseAgent]] = {
    "chain_of_thought": ChainOfThoughtSolver,
    "formula_first": FormulaFirstSolver,
    "backward": BackwardSolver,
}

FINAL_ANSWER_RE = re.compile(r"FINAL_ANSWER\s*:\s*(.+)", re.IGNORECASE)
_PLACEHOLDER_RE = re.compile(r"^(<[^>]*>|\[[^\]]*\])$")
_NUMBER_RE_AGENT = re.compile(r"[-+]?\d[\d,.]*")


def extract_final_answer(text: str) -> str:
    match = FINAL_ANSWER_RE.search(text)
    if match:
        candidate = match.group(1).strip()
        if not _PLACEHOLDER_RE.match(candidate):
            return candidate

    numbers = _NUMBER_RE_AGENT.findall(text)
    if numbers:
        return numbers[-1].replace(",", "")

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else text.strip()