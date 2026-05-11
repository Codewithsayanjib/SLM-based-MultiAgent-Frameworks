from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch



def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def ensure_dir(path: str | os.PathLike[str]) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)



def save_json(path: str | os.PathLike[str], payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


GSM8K_ANSWER_RE = re.compile(r"####\s*([-+]?\d[\d,.]*)")


def extract_gsm8k_answer(answer_text: str) -> str:
    match = GSM8K_ANSWER_RE.search(answer_text)
    if match:
        return match.group(1).replace("$", "").replace(",", "").strip()
    tail = answer_text.split("####")[-1].strip()
    return tail.replace("$", "").replace(",", "").strip()
