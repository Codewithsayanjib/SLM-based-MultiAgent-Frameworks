from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DatasetConfig:
    name: str = "gsm8k"
    split: str = "test"
    max_samples: int = 20
    subject: str = "algebra"    # MATH dataset only
    level_min: int = 3           # MATH dataset only
    level_max: int = 4           # MATH dataset only


@dataclass
class ModelConfig:
    name: str
    device: str = "auto"
    dtype: str = "auto"
    load_in_4bit: bool = False


@dataclass
class PipelineConfig:
    type: str = "diverse_ensemble"


@dataclass
class CommunicationConfig:
    max_tokens: int = 80
    num_steps: int = 4


@dataclass
class EnsembleConfig:
    strategies: list[str] = field(
        default_factory=lambda: ["chain_of_thought", "formula_first", "backward"]
    )


@dataclass
class GenerationConfig:
    max_new_tokens: int = 160
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False


@dataclass
class AppConfig:
    seed: int
    output_dir: str
    dataset: DatasetConfig
    model: ModelConfig
    pipeline: PipelineConfig
    ensemble: EnsembleConfig
    generation: GenerationConfig
    communication: CommunicationConfig = field(default_factory=CommunicationConfig)



def _merge_dataclass(cls: type, payload: dict[str, Any] | None):
    import dataclasses
    payload = payload or {}
    known = {f.name for f in dataclasses.fields(cls)}
    filtered = {k: v for k, v in payload.items() if k in known}
    return cls(**filtered)


def _validate_config(cfg: AppConfig) -> None:
    if cfg.dataset.max_samples <= 0:
        raise ValueError("dataset.max_samples must be > 0")
    if cfg.generation.temperature < 0:
        raise ValueError("generation.temperature must be >= 0")
    valid_pipeline_types = {"solver", "diverse_ensemble"}
    if cfg.pipeline.type not in valid_pipeline_types:
        raise ValueError(f"Unknown pipeline.type: {cfg.pipeline.type!r}. Must be one of {valid_pipeline_types}")
    valid_strategies = {"chain_of_thought", "formula_first", "backward"}
    unknown = set(cfg.ensemble.strategies) - valid_strategies
    if unknown:
        raise ValueError(f"Unknown ensemble strategies: {unknown}. Valid: {valid_strategies}")


def load_config(path: str | Path) -> AppConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = AppConfig(
        seed=raw.get("seed", 42),
        output_dir=raw.get("output_dir", "outputs/default"),
        dataset=_merge_dataclass(DatasetConfig, raw.get("dataset")),
        model=_merge_dataclass(ModelConfig, raw.get("model")),
        pipeline=_merge_dataclass(PipelineConfig, raw.get("pipeline")),
        ensemble=_merge_dataclass(EnsembleConfig, raw.get("ensemble")),
        generation=_merge_dataclass(GenerationConfig, raw.get("generation")),
        communication=_merge_dataclass(CommunicationConfig, raw.get("communication")),
    )
    _validate_config(cfg)
    return cfg
