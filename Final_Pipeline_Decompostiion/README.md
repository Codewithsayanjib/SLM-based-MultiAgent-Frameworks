# slm_comm_compact

A compact, configurable codebase for experiments on **multi-agent small language models** with different communication styles.

This repo keeps the entire project in **13 files** and is meant to be easy to run and modify on Kaggle.

## What it does

- Loads a public QA/reasoning dataset (currently **GSM8K** via Hugging Face datasets)
- Runs a configurable multi-agent pipeline:
  - `solver`
  - `planner_solver`
  - `planner_solver_verifier`
- Supports communication modes:
  - `natural`
  - `constrained`
  - `structured`
- Evaluates:
  - exact match
  - token cost
  - accuracy per token

## File layout

```text
slm_comm_compact/
  README.md
  pyproject.toml
  config.yaml
  run.py
  slm_comm/
    __init__.py
    config.py
    model.py
    agents.py
    communication.py
    pipeline.py
    data.py
    metrics.py
    utils.py
```

## Install

```bash
pip install -U pip
pip install -e .
```

## Run

```bash
python run.py --config config.yaml
```

## Kaggle notes

Recommended starter models:
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- `microsoft/phi-2`
- `google/gemma-2-2b-it`

If VRAM is tight, reduce:
- `generation.max_new_tokens`
- `dataset.max_samples`
- `model.load_in_4bit`

## Design decisions

- The code is intentionally **small and opinionated**.
- Prompt templates live inside the agent code to avoid file sprawl.
- Config is plain YAML with dataclass validation.
- Communication is implemented as an actual transform layer between planner and solver.
- The verifier is optional and can revise the solver answer.

## Current scope

Implemented and runnable now:
- natural language communication
- constrained communication (hard truncation)
- structured communication (slot extraction)
- Hugging Face generation backend
- GSM8K data loading and evaluation

Not included yet:
- learned communication bottleneck training
- experiment tracking services
- distributed training

Those can be added later without changing the core structure.
