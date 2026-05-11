from __future__ import annotations

import re

from slm_comm.config import CommunicationConfig


class CommunicationTransform:
    def __call__(self, planner_text: str) -> str:
        raise NotImplementedError


class NaturalCommunication(CommunicationTransform):
    def __call__(self, planner_text: str) -> str:
        return planner_text.strip()


class ConstrainedCommunication(CommunicationTransform):
    def __init__(self, max_tokens: int, tokenizer=None):
        self.max_tokens = max_tokens
        self.tokenizer = tokenizer

    def __call__(self, planner_text: str) -> str:
        if self.tokenizer is not None:
            ids = self.tokenizer.encode(planner_text.strip(), add_special_tokens=False)
            return self.tokenizer.decode(ids[: self.max_tokens], skip_special_tokens=True).strip()
        words = planner_text.strip().split()
        return " ".join(words[: self.max_tokens])


class StructuredCommunication(CommunicationTransform):
    def __init__(self, max_tokens: int, num_steps: int, tokenizer=None):
        self.max_tokens = max_tokens
        self.num_steps = num_steps
        self.tokenizer = tokenizer

    # Matches: "1.", "1)", "Step 1:", "- text", "• text", "* text"
    _STEP_RE = re.compile(
        r"^(\d+[.):]|step\s*\d+[.:]?|[-•*])\s+\S",
        flags=re.IGNORECASE,
    )
    _STRIP_PREFIX_RE = re.compile(
        r"^(\d+[.):]|step\s*\d+[.:]?|[-•*])\s+",
        flags=re.IGNORECASE,
    )

    def __call__(self, planner_text: str) -> str:
        lines = [line.strip() for line in planner_text.splitlines() if line.strip()]
        candidate_steps = []

        # Check if first line is a step without a number prefix (because prompt pre-filled "1.")
        if lines and not self._STEP_RE.match(lines[0]):
            candidate_steps.append(lines[0])  # Add it manually as step 1

        for line in lines:
            if self._STEP_RE.match(line):
                cleaned = self._STRIP_PREFIX_RE.sub("", line).strip()
                candidate_steps.append(cleaned)

        if not candidate_steps:
            # Fall back: sentence split, keep non-trivial sentences (>15 chars)
            sentences = re.split(r"(?<=[.!?])\s+", planner_text.strip())
            candidate_steps = [s.strip() for s in sentences if len(s.strip()) > 15]

        selected = candidate_steps[: self.num_steps]

        # Use plain "1." numbering — more natural and fewer tokens than [STEP_X]
        structured = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(selected))

        if self.tokenizer is not None:
            ids = self.tokenizer.encode(structured, add_special_tokens=False)
            return self.tokenizer.decode(ids[: self.max_tokens], skip_special_tokens=True).strip()
        words = structured.split()
        return " ".join(words[: self.max_tokens])


def build_communication(mode: str, cfg: CommunicationConfig, tokenizer=None) -> CommunicationTransform:
    mode = mode.lower()
    if mode == "natural":
        return NaturalCommunication()
    if mode == "constrained":
        return ConstrainedCommunication(max_tokens=cfg.max_tokens, tokenizer=tokenizer)
    if mode == "structured":
        return StructuredCommunication(max_tokens=cfg.max_tokens, num_steps=cfg.num_steps, tokenizer=tokenizer)
    raise ValueError(f"Unsupported communication mode: {mode}")