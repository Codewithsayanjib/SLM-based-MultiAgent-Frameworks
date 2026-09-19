<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=28&pause=1000&color=6C63FF&center=true&vCenter=true&width=1000&lines=SLM-Based+Multi-Agent+Frameworks;Methodology+1+%3A+Pipeline+Decomposition;Methodology+2+%3A+Diverse+Ensemble+Reasoning;SVAMP+Benchmark" alt="Typing SVG" />

<br/>

# 🧠 SLM-Based Multi-Agent Frameworks
## Methodology 1 : Pipeline Decomposition &nbsp;|&nbsp; Methodology 2 : Diverse Ensemble Reasoning

<br/>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![HuggingFace](https://img.shields.io/badge/🤗_HuggingFace-Transformers-FF9D00?style=for-the-badge)](https://huggingface.co)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Dataset](https://img.shields.io/badge/Dataset-SVAMP-4CAF50?style=for-the-badge)](https://huggingface.co/datasets/ChilleD/SVAMP)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
![Status](https://img.shields.io/badge/Status-Submitted_at_ICCUBEA_2026-blueviolet?style=for-the-badge)

<br/>

> **Can structured role decomposition and diverse reasoning strategies push Small Language Models beyond their individual limits?**
> This repository explores two complementary multi-agent methodologies for mathematical reasoning —
> *Pipeline Decomposition* (sequential role-based agents with controlled communication) and
> *Diverse Ensemble Reasoning* (parallel strategy-diverse agents aggregated by majority vote) —
> evaluated across five Small Language Models on the SVAMP benchmark.

<br/>

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Methodology 1 — Pipeline Decomposition](#-methodology-1--pipeline-decomposition)
  - [Architecture](#-architecture-1)
  - [Agent Roles](#agent-roles)
  - [Communication Modes](#communication-modes)
  - [Pipeline Variants](#pipeline-variants)
  - [Results](#-results-methodology-1)
- [Methodology 2 — Diverse Ensemble Reasoning](#-methodology-2--diverse-ensemble-reasoning)
  - [Architecture](#-architecture-2)
  - [Solver Strategies](#solver-strategies)
  - [Aggregation](#aggregation)
  - [Results](#-results-methodology-2)
- [Supported Models](#-supported-models)
- [Supported Datasets](#-supported-datasets)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Configuration Reference](#-configuration-reference)
- [Key Design Decisions](#-key-design-decisions)
- [Citation](#-citation)

---

## 🔭 Overview

This repository contains two methodologies from a broader research program on Small Language Model (SLM) based multi-agent frameworks for mathematical reasoning, evaluated on the **SVAMP benchmark** across **5 SLMs** (1.5B–7B parameters).

| | Methodology 1 | Methodology 2 |
|---|---|---|
| **Name** | Pipeline Decomposition | Diverse Ensemble Reasoning |
| **Agent flow** | Sequential — Planner → Solver → Verifier | Parallel — CoT ∥ Formula-First ∥ Backward |
| **Communication** | Explicit, with 3 controllable modes | None — agents are fully isolated |
| **Aggregation** | Single chain (final agent wins) | Majority vote across strategies |
| **Key variable** | Communication quality and compression | Reasoning strategy diversity |
| **Folder** | `Final_Pipeline_Decompostiion/` | `diverse ensemble classifier/` |

---

## 🧩 Methodology 1 — Pipeline Decomposition

The central premise of Methodology 1 is **role specialization through sequential handoff**. A Planner decomposes the problem, its output is transformed by a controlled communication layer, and passed to a Solver — optionally followed by a Verifier that checks and corrects the answer. The communication channel between agents is a key experimental variable.

### 🏗️ Architecture 1

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INPUT QUESTION                              │
│              (SVAMP / GSM8K / MATH arithmetic problem)              │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       PlannerAgent                                  │
│   Identifies relevant numbers, lists ≤4 steps, no computation      │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   Communication Transform                           │
│                                                                     │
│   ┌──────────────┐   ┌───────────────────┐   ┌──────────────────┐  │
│   │   Natural    │   │   Constrained     │   │   Structured     │  │
│   │  (verbatim)  │   │ (token-truncated) │   │ (step-parsed +   │  │
│   │              │   │                   │   │  re-numbered)    │  │
│   └──────────────┘   └───────────────────┘   └──────────────────┘  │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        SolverAgent                                  │
│   Follows the plan, computes answer, ends with FINAL_ANSWER: ...   │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼ (optional)
┌─────────────────────────────────────────────────────────────────────┐
│                       VerifierAgent                                 │
│   Checks reasoning, confirms or corrects, ends with FINAL_ANSWER:  │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
              ┌──────────────────────────────────┐
              │         FINAL PREDICTION          │
              │  + per-agent token cost           │
              │  + planner / solver / verifier    │
              │    text logged per example        │
              └──────────────────────────────────┘
```

---

### Agent Roles

#### 🗺️ PlannerAgent
> *"You are a planning agent. Your ONLY job is to list steps. Output ONLY a numbered list. Maximum 4 steps. Do NOT compute any numbers. First step must identify which numbers in the problem are relevant to the question and which are distractors to ignore."*

Decomposes the problem into an ordered plan without doing any arithmetic. Crucially, it is required to flag distractor numbers — a key challenge in SVAMP problems.

#### 🔢 SolverAgent
> *"You are a solver agent. Solve the problem carefully and give the final answer. End with a line exactly in this format: FINAL_ANSWER: \<answer\>"*

Receives the question and the (optionally transformed) plan from the Planner. Follows the plan step by step to produce a complete solution.

#### ✅ VerifierAgent
> *"You are a verification agent. Check the proposed solution carefully. If correct, output VERIFICATION: correct and FINAL_ANSWER: \<same answer\>. If incorrect, output VERIFICATION: incorrect and FINAL_ANSWER: \<corrected answer\>."*

Receives the question and the Solver's full output. Either confirms the answer or provides a corrected one. The final prediction always comes from the Verifier in the full pipeline.

---

### Communication Modes

The Planner's output is not passed raw to the Solver. It passes through one of three transforms:

| Mode | Class | Behaviour |
|------|-------|-----------|
| `natural` | `NaturalCommunication` | Pass the planner text verbatim, stripped of whitespace |
| `constrained` | `ConstrainedCommunication` | Hard-truncate to `max_tokens` tokens using the actual tokenizer |
| `structured` | `StructuredCommunication` | Parse numbered/bulleted steps, re-number cleanly as `1. ... 2. ...`, then truncate to `max_tokens`; falls back to sentence splitting if no list structure is found |

Configured via `pipeline.communication` in `config.yaml` and `communication.max_tokens` / `communication.num_steps`.

---

### Pipeline Variants

| `pipeline.type` | Class | Agents active |
|-----------------|-------|---------------|
| `solver` | `SolverOnlyPipeline` | Solver only (baseline — single CoT) |
| `planner_solver` | `PlannerSolverPipeline` | Planner → Communication → Solver |
| `planner_solver_verifier` | `PlannerSolverVerifierPipeline` | Planner → Communication → Solver → Verifier |

---

### 📊 Results — Methodology 1

> Evaluated on **SVAMP** · 300 samples · `planner_solver_verifier` pipeline · greedy decoding (`temperature=0.0`, `max_new_tokens=512`) · communication budget `max_tokens=120`, `num_steps=5`.
> Accuracy is reported with 95% bootstrap confidence intervals (10k resamples); **T̄** is the mean total tokens per problem summed across planner + solver + verifier. Per-mode summaries and CIs are in `outputs_n300/` (`ci_summary.md`).

#### Accuracy (%) by Communication Mode

| Model | Natural | Constrained | Structured | T̄ (tokens) |
|-------|:-------:|:-----------:|:----------:|:----------:|
| DeepSeek-R1-Distill-Qwen-1.5B | 71.3 ±5.2 | 71.7 ±5.0 | 66.3 ±5.3 | ~1630 |
| Gemma2-2B | 66.0 ±5.3 | 66.0 ±5.3 | 70.3 ±5.0 | ~730 |
| LLaMA 3.2-3B | 67.7 ±5.2 | 68.0 ±5.3 | 70.3 ±5.2 | ~1067 |
| Mistral-7B | 63.3 ±5.5 | 62.7 ±5.5 | 62.3 ±5.5 | ~870 |
| Qwen2.5-7B | 55.0 ±5.7 | 52.3 ±5.7 | 54.7 ±5.7 | ~2660 |

> **Answer extraction.** The final answer is read from the Verifier, falling back to the Solver's answer when the Verifier confirms without restating a number (e.g. it outputs only `VERIFICATION: correct`). Reading the Verifier text literally in that case discards the numeric answer and under-counts accuracy — most severely for models with terse verifiers (Mistral, Qwen). See `extract_verified_answer` in `slm_comm/agents.py`.

#### 🔍 Key Design Insights

- 📉 **Token efficiency separates the models more than accuracy does.** Accuracy clusters in the ~52–72% band, while T̄ ranges from ~730 (Gemma2-2B) to ~2660 (Qwen2.5-7B). Gemma2-2B and Mistral-7B give the best accuracy-per-token.
- 🐘 **Qwen2.5-7B is the least efficient** — verbose LaTeX-style generation inflates token cost ~3× with no accuracy benefit; it also runs to the token cap most often.
- ⚖️ **Communication mode effects are small.** Within each model the three modes' confidence intervals overlap; structured is best or tied for LLaMA and Gemma2 but slightly hurts DeepSeek-R1.
- 🚫 **Distractor identification by the Planner** is the critical first step — SVAMP deliberately includes irrelevant numbers, and surfacing them early improves Solver accuracy.
- 🔎 **Answer-extraction robustness is itself a result** — naive verifier-only parsing severely under-counts models whose verifier confirms tersely, so a solver fallback is required for a fair comparison.

---

## 🧩 Methodology 2 — Diverse Ensemble Reasoning

The central premise of Methodology 2 is that **no single reasoning style dominates across all problem types**. By running three cognitively distinct agents in parallel and reconciling their outputs through democratic voting, the ensemble captures a richer solution space than any individual solver.

### 🏗️ Architecture 2

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INPUT QUESTION                              │
│              (SVAMP / GSM8K / MATH arithmetic problem)              │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     DiverseEnsemblePipeline                         │
│                                                                     │
│   ┌─────────────────┐  ┌─────────────────┐  ┌──────────────────┐   │
│   │  🔗 Chain-of-   │  │  📐 Formula-    │  │  🔄 Backward     │   │
│   │   Thought       │  │   First         │  │   Reasoning      │   │
│   │   Solver        │  │   Solver        │  │   Solver         │   │
│   │                 │  │                 │  │                  │   │
│   │  Step-by-step   │  │  Equations →    │  │  Goal → Sub-     │   │
│   │  reasoning      │  │  Substitution   │  │  goals → Answer  │   │
│   └────────┬────────┘  └────────┬────────┘  └────────┬─────────┘   │
│            │                   │                     │             │
│            ▼                   ▼                     ▼             │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │              Answer Extraction  (FINAL_ANSWER: ...)         │  │
│   │         Fallback: last number in output text                │  │
│   └─────────────────────────────────────────────────────────────┘  │
│            │                   │                     │             │
│            └───────────────────┴─────────────────────┘             │
│                                │                                   │
│                                ▼                                   │
│                   ┌────────────────────────┐                       │
│                   │    Majority Vote        │                       │
│                   │  (tie → first strategy) │                       │
│                   └────────────┬───────────┘                       │
└────────────────────────────────┼────────────────────────────────────┘
                                 │
                                 ▼
              ┌──────────────────────────────────┐
              │         FINAL PREDICTION          │
              │  + per-strategy breakdown         │
              │  + token cost per strategy        │
              │  + disagreement analysis          │
              └──────────────────────────────────┘
```

---

### Solver Strategies

Each agent receives the same question but is prompted with a distinct reasoning frame:

#### 🔗 Chain-of-Thought (`chain_of_thought`)
> *"Solve the problem by thinking step by step. Show each reasoning step clearly and in order."*

Encourages sequential, narrative reasoning. Works well for multi-step word problems where tracking intermediate quantities matters.

#### 📐 Formula-First (`formula_first`)
> *"First, write out every equation or formula you will need. Then substitute values and compute the result."*

Forces the model to commit to a mathematical structure before computing. Reduces arithmetic errors by separating planning from execution.

#### 🔄 Backward Reasoning (`backward`)
> *"Start by clearly stating what quantity the question asks for. Work backwards: what do you need to compute that?"*

Grounds the model in the target quantity first, preventing common errors where models lose track of what they are solving for.

All agents are required to terminate their response with:
```
FINAL_ANSWER: [numerical answer]
```
A robust extraction pipeline handles this tag, with fallback to the last number appearing in the output text, and placeholder detection to prevent empty or malformed extractions.

---

### Aggregation

After all three solvers produce predictions, the final answer is determined by **majority vote**:

```python
vote_counts = Counter(predictions.values())
final_pred, _ = vote_counts.most_common(1)[0]
# Tie-breaking: first strategy (chain_of_thought) wins
```

The pipeline also computes:
- **Disagreement Rate** — fraction of examples where not all three strategies agreed
- **Unique Correct Rate** — fraction where exactly one strategy was correct (measures how uncorrelated the errors are across strategies)

---

### 📊 Results — Methodology 2

> Evaluated on **SVAMP** · 300 samples · `diverse_ensemble` pipeline · greedy decoding (`temperature=0.0`, `max_new_tokens=512`).
> Accuracy is reported with 95% bootstrap confidence intervals (10k resamples); **T̄** is the mean total tokens per problem across the three solvers. Full per-model summaries and CIs are in `outputs_ensemble_n300/` (`ci_summary_ensemble.md`).

#### Ensemble Accuracy

| Model | Params | Ensemble Acc | 95% CI | T̄ (tokens) |
|-------|--------|:------------:|:------:|:----------:|
| DeepSeek-R1-Distill-Qwen-1.5B | 1.5B | 52.3% | [46.7, 58.0] | 1518 |
| Gemma2-2B | 2B | 72.0% | [67.0, 77.0] | 814 |
| **LLaMA 3.2-3B** | **3B** | **85.3%** | **[81.3, 89.3]** | **971** |
| Mistral-7B | 7B | 71.0% | [65.7, 76.0] | 952 |
| Qwen2.5-7B | 7B | 19.7% | [15.3, 24.3] | 1929 |

#### Per-Strategy Accuracy Breakdown

| Model | Chain-of-Thought | Formula-First | Backward |
|-------|:----------------:|:-------------:|:--------:|
| DeepSeek-R1-Distill-Qwen-1.5B | 36.7% | 43.3% | **61.0%** |
| Gemma2-2B | **69.0%** | 54.0% | 51.0% |
| LLaMA 3.2-3B | **84.0%** | 70.0% | 46.0% |
| Mistral-7B | **69.7%** | 55.3% | 61.7% |
| Qwen2.5-7B | 18.3% | **64.7%** | 8.3% |

#### Ensemble Diversity Analysis

| Model | Disagreement Rate | Unique Correct Rate |
|-------|:-----------------:|:-------------------:|
| DeepSeek-R1-Distill-Qwen-1.5B | 89% | 34% |
| Gemma2-2B | 74% | 24% |
| LLaMA 3.2-3B | 66% | 20% |
| Mistral-7B | 94% | 19% |
| Qwen2.5-7B | 99% | 54% |

#### 🔍 Key Observations

- 🏆 **LLaMA 3.2-3B achieves the highest ensemble accuracy (85.3%)** — outperforming both 7B models; its CI clears the rest, so the lead is statistically significant.
- 🤝 **Gemma2-2B and Mistral-7B are a statistical tie for second** (72.0% vs 71.0%, overlapping CIs), with Gemma2 the most token-efficient of the group (T̄ 814).
- 🔄 **Backward reasoning is the strongest single strategy for DeepSeek-R1-1.5B (61%)** — its distilled reasoning style aligns with goal-first decomposition.
- 📐 **Formula-First is the only viable strategy for Qwen2.5-7B (64.7%)** — near-random CoT/Backward despite its size, and it is outvoted by them under majority vote.
- 📉 **Majority vote collapses under high disagreement** — when strategies rarely agree and errors are diverse (Qwen2.5-7B, 99% disagreement), voting hurts rather than helps.

---

## 🤖 Supported Models

Any `AutoModelForCausalLM`-compatible HuggingFace model is supported. The following were evaluated in this study:

| Model | Parameters | Device Support |
|-------|-----------|----------------|
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | 1.5B | CPU / MPS |
| `google/gemma-2-2b-it` | 2B | CPU / MPS / CUDA |
| `meta-llama/Llama-3.2-3B` | 3B | CPU / MPS / CUDA |
| `mistralai/Mistral-7B-v0.1` | 7B | CPU / MPS / CUDA |
| `Qwen/Qwen2.5-7B` | 7B | CPU / MPS / CUDA |

> **Apple Silicon users:** Set `device: mps` and `dtype: float32` in config. `float16` produces NaN on CPU/MPS — `float32` is the safe default.

> **GPU users (T4/A100):** Set `device: cuda` and `dtype: float16`. Enable `load_in_4bit: true` for 7B+ models to reduce VRAM.

---

## 📦 Supported Datasets

| Dataset | Config Name | Notes |
|---------|------------|-------|
| [SVAMP](https://huggingface.co/datasets/ChilleD/SVAMP) | `svamp` | Simple arithmetic variations; `train` split |
| [GSM8K](https://huggingface.co/datasets/gsm8k) | `gsm8k` | Grade school math; `test` split |
| [MATH (Hendrycks)](https://huggingface.co/datasets/nlile/hendrycks-MATH-benchmark) | `math` | Filter by `subject` and `level_min`/`level_max` |

---

## 📁 Project Structure

```
SLM-based-MultiAgent-Frameworks/
│
├── Final_Pipeline_Decompostiion/        # Methodology 1
│   ├── slm_comm/
│   │   ├── agents.py                   # PlannerAgent, SolverAgent, VerifierAgent
│   │   ├── pipeline.py                 # SolverOnlyPipeline, PlannerSolverPipeline,
│   │   │                               #   PlannerSolverVerifierPipeline
│   │   ├── communication.py            # Natural, Constrained, Structured transforms
│   │   ├── model.py                    # HuggingFaceTextGenerator (CausalLM wrapper)
│   │   ├── metrics.py                  # is_correct(), normalize_answer(), summarize_results()
│   │   ├── data.py                     # Dataset loaders: SVAMP / GSM8K / MATH
│   │   ├── config.py                   # Dataclass configs + YAML loader + validator
│   │   └── utils.py                    # set_seed(), save_json(), extract_gsm8k_answer()
│   │
│   ├── config.yaml                     # Experiment configuration
│   ├── run.py                          # Entry point
│   └── outputs/                        # Per-run prediction & summary JSONs
│       ├── qwen2.5_7b/
│       │   ├── natural/
│       │   ├── constrained/
│       │   └── structured/
│       ├── mistral-7b/
│       ├── gemma2-2b/
│       ├── Deepseek-r1-1.5b/
│       └── llama3.2/
│
├── diverse ensemble classifier/        # Methodology 2
│   ├── slm_comm/
│   │   ├── agents.py                   # ChainOfThoughtSolver, FormulaFirstSolver,
│   │   │                               #   BackwardSolver + STRATEGY_REGISTRY
│   │   ├── pipeline.py                 # SolverOnlyPipeline, DiverseEnsemblePipeline
│   │   ├── communication.py            # Stub — not active in this methodology
│   │   ├── model.py                    # HuggingFaceTextGenerator (CausalLM wrapper)
│   │   ├── metrics.py                  # is_correct(), summarize_results(),
│   │   │                               #   disagreement_rate, unique_correct_rate, NATE
│   │   ├── data.py                     # Dataset loaders: SVAMP / GSM8K / MATH
│   │   ├── config.py                   # Dataclass configs + YAML loader + validator
│   │   └── utils.py                    # set_seed(), save_json(), extract_gsm8k_answer()
│   │
│   ├── config.yaml                     # Experiment configuration
│   ├── run.py                          # Entry point
│   └── outputs/                        # Per-run prediction & summary JSONs
│       ├── Qwen2.5_7B/
│       ├── gemma2_2b_it/
│       ├── llama3.2b/
│       ├── deepseek_r1_distill_qwen_1.5b/
│       └── mistral_7b/
│
└── README.md
```

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/Codewithsayanjib/SLM-based-MultiAgent-Frameworks.git
cd SLM-based-MultiAgent-Frameworks

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install torch transformers datasets pyyaml numpy

# 4. (Optional) For 4-bit quantization on CUDA
pip install bitsandbytes
```

---

## 🚀 Usage

### Methodology 1 — Pipeline Decomposition

```bash
cd Final_Pipeline_Decompostiion
```

Edit `config.yaml`:

```yaml
seed: 42
output_dir: outputs/gemma2_svamp/constrained

dataset:
  name: svamp
  split: train
  max_samples: 100

model:
  name: google/gemma-2-2b-it
  device: mps          # mps | cpu | cuda
  dtype: float32       # float32 (CPU/MPS) | float16 (CUDA)
  load_in_4bit: false

pipeline:
  type: planner_solver_verifier   # solver | planner_solver | planner_solver_verifier
  communication: constrained      # natural | constrained | structured

communication:
  max_tokens: 120
  num_steps: 5

generation:
  max_new_tokens: 512
  temperature: 0.0
  top_p: 1.0
  do_sample: false
```

```bash
python run.py --config config.yaml
```

**Outputs:**
```
outputs/gemma2_svamp/constrained/
├── predictions.json     # Per-example: question, gold, planner/solver/verifier texts,
│                        #   communication passed, final prediction
└── summary.json         # Accuracy, token cost per agent, accuracy_per_1000_tokens
```

---

### Methodology 2 — Diverse Ensemble Reasoning

```bash
cd "diverse ensemble classifier"
```

Edit `config.yaml`:

```yaml
seed: 42
output_dir: outputs/run_01/llama3.2_3b

dataset:
  name: svamp
  split: train
  max_samples: 100

model:
  name: meta-llama/Llama-3.2-3B
  device: mps          # mps | cpu | cuda
  dtype: float32       # float32 (CPU/MPS) | float16 (CUDA)
  load_in_4bit: false

pipeline:
  type: diverse_ensemble   # diverse_ensemble | solver (baseline)

ensemble:
  strategies:
    - chain_of_thought
    - formula_first
    - backward

generation:
  max_new_tokens: 512
  temperature: 0.0
  top_p: 1.0
  do_sample: false
```

```bash
python run.py --config config.yaml
```

**Outputs:**
```
outputs/run_01/llama3.2_3b/
├── predictions.json     # Per-example: question, gold, all strategy texts,
│                        #   per-strategy predictions, final prediction
└── summary.json         # Accuracy, token cost, per-strategy breakdown,
                         #   disagreement rate, unique correct rate, NATE
```

**`summary.json` example (LLaMA 3.2-3B):**
```json
{
  "num_examples": 100,
  "num_correct": 81,
  "accuracy": 0.81,
  "total_token_cost": 88562,
  "avg_token_cost": 885.62,
  "accuracy_chain_of_thought": 0.77,
  "accuracy_formula_first": 0.60,
  "accuracy_backward": 0.64,
  "disagreement_rate": 0.61,
  "unique_correct_rate": 0.12
}
```

### Run as Baseline (Single Solver — both methodologies)

```yaml
pipeline:
  type: solver    # single chain-of-thought agent only
```

---

## 🛠️ Configuration Reference

### Shared Keys

| Key | Type | Description |
|-----|------|-------------|
| `seed` | `int` | Global random seed for reproducibility |
| `output_dir` | `str` | Directory for `predictions.json` and `summary.json` |
| `dataset.name` | `str` | `gsm8k` / `svamp` / `math` |
| `dataset.split` | `str` | HuggingFace dataset split |
| `dataset.max_samples` | `int` | Number of examples to evaluate |
| `model.name` | `str` | HuggingFace model ID |
| `model.device` | `str` | `mps` / `cpu` / `cuda` |
| `model.dtype` | `str` | `float32` / `float16` / `bfloat16` |
| `model.load_in_4bit` | `bool` | Enable BitsAndBytes 4-bit NF4 quantization |
| `generation.max_new_tokens` | `int` | Max tokens to generate per agent call |
| `generation.temperature` | `float` | Sampling temperature (`0.0` = greedy) |
| `generation.do_sample` | `bool` | Enable stochastic sampling |

### Methodology 1 Only

| Key | Type | Description |
|-----|------|-------------|
| `pipeline.type` | `str` | `solver` / `planner_solver` / `planner_solver_verifier` |
| `pipeline.communication` | `str` | `natural` / `constrained` / `structured` |
| `communication.max_tokens` | `int` | Token budget for Constrained / Structured modes |
| `communication.num_steps` | `int` | Max steps to extract in Structured mode |

### Methodology 2 Only

| Key | Type | Description |
|-----|------|-------------|
| `pipeline.type` | `str` | `diverse_ensemble` / `solver` |
| `ensemble.strategies` | `list` | Any combination of `chain_of_thought`, `formula_first`, `backward` |

---

## 💡 Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Sequential agents with communication (M1)** | Role specialization allows each agent to focus on one cognitive task; the communication layer controls information quality between stages. |
| **Parallel agents, no communication (M2)** | Agents solve independently; no inter-agent message passing. This isolates strategy diversity as the sole variable and avoids error propagation between agents. |
| **`FINAL_ANSWER:` protocol** | Structured extraction tag prevents the model from producing open-ended prose. Fallback to last number handles non-compliant outputs gracefully. |
| **Distractor identification in Planner (M1)** | SVAMP problems deliberately include irrelevant numbers; making the Planner explicitly flag them is the primary mechanism for filtering noise before it reaches the Solver. |
| **`repetition_penalty=1.2` (M2)** | Breaks infinite repetition loops observed in `formula_first` strategy on some models. |
| **`float32` on CPU/MPS** | `float16` produces NaN values on non-CUDA devices. `float32` is enforced as the safe default. |
| **Tie-breaking by first strategy (M2)** | When the vote is split three ways, Chain-of-Thought is given priority — empirically the most reliable individual strategy across models. |
| **Lazy checkpoint saves** | `predictions.json` is written every 10 examples, ensuring no results are lost on crash or OOM. |
| **NATE metric (M2)** | Normalized Accuracy per Thousand Tokens — measures accuracy per unit of token cost, enabling fair efficiency comparisons across models. |

---

## 📖 Citation

If you use this code or findings in your research, please cite:

```bibtex
@inproceedings{sur2026slm,
  title     = {Structured Communication and Ensemble Diversity in SLM based Multi-Agent Systems: A Comparative Study},
  author    = {Sur, Sayanjib and Sil Sarma, Ankush and Singh, Pawan Kumar},
  booktitle = {Proceedings of the 10th International Conference on Computing, Communication, Control \& Automation (ICCUBEA)},
  year      = {2026},
  note      = {Submitted}
}
```

---

<div align="center">

Made with 🔬 at **Jadavpur University**

<br/>

[![GitHub](https://img.shields.io/badge/GitHub-Codewithsayanjib-181717?style=for-the-badge&logo=github)](https://github.com/Codewithsayanjib)

</div>
