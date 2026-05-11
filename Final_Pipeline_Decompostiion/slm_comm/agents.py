from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from slm_comm.model import HuggingFaceTextGenerator


@dataclass
class AgentOutput:
    text: str
    prompt_tokens: int
    completion_tokens: int


class BaseAgent:
    def __init__(self, model: HuggingFaceTextGenerator):
        self.model = model

    def _call(self, prompt: str) -> AgentOutput:
        out = self.model.generate(prompt)
        return AgentOutput(
            text=out["text"],
            prompt_tokens=out["prompt_tokens"],
            completion_tokens=out["completion_tokens"],
        )


class PlannerAgent(BaseAgent):
    def run(self, question: str) -> AgentOutput:
        prompt = (
    "You are a planning agent. Your ONLY job is to list steps.\n"
    "Rules:\n"
    "- Output ONLY a numbered list. Nothing else.\n"
    "- Maximum 4 steps.\n"
    "- Do NOT compute any numbers.\n"
    "- Do NOT write explanations, examples, or solutions.\n"
    "- Stop immediately after listing the steps.\n"
    "- First step must identify which numbers in the problem are relevant "
    "to the question and which are distractors to ignore.\n\n"
    f"Question: {question}\n\n"
    "Steps:\n"
    "1."
)
        return self._call(prompt)


class SolverAgent(BaseAgent):
    def run(self, question: str, communication: str | None = None) -> AgentOutput:
        plan_section = (
            f"Step-by-step plan:\n{communication}\n\n"
            "Follow the above plan step by step.\n\n"
        ) if communication else ""
        prompt = (
            "You are a solver agent. Solve the problem carefully and give the final answer.\n"
            "End with a line exactly in this format: FINAL_ANSWER: <answer>\n\n"
            f"{plan_section}"
            f"Question: {question}\n\n"
            "Solution:"
        )
        return self._call(prompt)


class VerifierAgent(BaseAgent):
    def run(self, question: str, draft_answer: str) -> AgentOutput:
        prompt = (
            "You are a verification agent. Check the proposed solution carefully.\n"
            "If the reasoning and final answer are correct, output exactly:\n"
            "  VERIFICATION: correct\n"
            "  FINAL_ANSWER: <same answer as proposed>\n"
            "If you find an error, provide the corrected reasoning and output:\n"
            "  VERIFICATION: incorrect\n"
            "  FINAL_ANSWER: <corrected answer>\n"
            "You MUST always end your response with a FINAL_ANSWER: line containing "
            "only the answer value, nothing else.\n\n"
            f"Question: {question}\n\n"
            f"Proposed solution:\n{draft_answer}\n\n"
            "Verification:"
        )
        return self._call(prompt)


FINAL_ANSWER_RE = re.compile(r"FINAL_ANSWER\s*:\s*(.+)", re.IGNORECASE)
_NUMBER_RE_AGENT = re.compile(r"[-+]?\d[\d,.]*")


def extract_final_answer(text: str) -> str:
    match = FINAL_ANSWER_RE.search(text)
    if match:
        return match.group(1).strip()
    # Secondary fallback: last number appearing in the text
    numbers = _NUMBER_RE_AGENT.findall(text)
    if numbers:
        return numbers[-1].replace(",", "")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else text.strip()