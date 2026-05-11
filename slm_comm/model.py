from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from slm_comm.config import GenerationConfig, ModelConfig


class HuggingFaceTextGenerator:
    def __init__(self, model_cfg: ModelConfig, gen_cfg: GenerationConfig):
        self.model_cfg = model_cfg
        self.gen_cfg = gen_cfg

        quant_config = None
        if model_cfg.load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )

        dtype = self._resolve_dtype(model_cfg.dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model_cfg.name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs: dict[str, Any] = {
        "trust_remote_code": False,
        "quantization_config": quant_config,
}
        if quant_config is None:
            model_kwargs["dtype"] = dtype  # was "torch_dtype", now correct key

        self.model = AutoModelForCausalLM.from_pretrained(
            model_cfg.name,
            device_map=None,          # no auto offload — load fully into RAM
            low_cpu_mem_usage=False,  # ensures weights fully materialised
            **model_kwargs,
        )
        device = "mps" if model_cfg.device == "mps" else "cpu"
        self.model.to(device)
        self.model.eval()

    def _resolve_dtype(self, value: str):
        if value == "float16":
            return torch.float16
        if value == "bfloat16":
            return torch.bfloat16
        return torch.float32  # float32 default on CPU — float16 causes NaN on CPU

    @torch.inference_mode()
    def generate(self, prompt: str) -> dict[str, Any]:
        max_length = getattr(self.model.config, "max_position_embeddings", 2048)
        encoded = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
        encoded = {k: v.to(self.model.device) for k, v in encoded.items()}

        gen_kwargs: dict[str, Any] = dict(
            max_new_tokens=self.gen_cfg.max_new_tokens,
            top_p=self.gen_cfg.top_p,
            do_sample=self.gen_cfg.do_sample,
            repetition_penalty=1.2,   # breaks the repetition loops in formula_first
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        # temperature only passed when sampling is on — suppresses the generation flag warning
        if self.gen_cfg.do_sample:
            gen_kwargs["temperature"] = self.gen_cfg.temperature

        output = self.model.generate(**encoded, **gen_kwargs)
        input_len = encoded["input_ids"].shape[1]
        completion_ids = output[0][input_len:]
        text = self.tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
        return {
            "text": text,
            "prompt_tokens": int(input_len),
            "completion_tokens": int(completion_ids.shape[0]),
            "generation_config": asdict(self.gen_cfg),
        }