<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=28&pause=1000&color=6C63FF&center=true&vCenter=true&width=1000&lines=SLM-Based+Multi-Agent+Frameworks;Methodology+1+%3A+Pipeline+Decomposition;Methodology+2+%3A+Diverse+Ensemble+Reasoning;SVAMP+Benchmark" alt="Typing SVG" />

<br/>

# 🧠 SLM-Based Multi-Agent Frameworks
## Methodology 2 : Diverse Ensemble Reasoning

<br/>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![HuggingFace](https://img.shields.io/badge/🤗_HuggingFace-Transformers-FF9D00?style=for-the-badge)](https://huggingface.co)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Dataset](https://img.shields.io/badge/Dataset-SVAMP-4CAF50?style=for-the-badge)](https://huggingface.co/datasets/ChilleD/SVAMP)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active_Research-blueviolet?style=for-the-badge)]()

<br/>

> **Can a diverse committee of small reasoning strategies outperform any single solver?**
> This methodology decomposes mathematical reasoning into a three-strategy parallel ensemble —
> *Chain-of-Thought*, *Formula-First*, and *Backward Reasoning* — aggregated via majority vote,
> evaluated across five Small Language Models on the SVAMP benchmark.

<br/>

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Methodology](#-methodology)
  - [Pipeline Decomposition](#pipeline-decomposition)
  - [Solver Strategies](#solver-strategies)
  - [Aggregation](#aggregation)
- [Supported Models](#-supported-models)
- [Supported Datasets](#-supported-datasets)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Configuration Reference](#-configuration-reference)
- [Key Design Decisions](#-key-design-decisions)
- [Citation](#-citation)

---

## 🔭 Overview

This repository contains **Methodology 1** of a broader research program on Small Language Model (SLM) based multi-agent frameworks for mathematical reasoning. The core idea is **pipeline decomposition**: instead of relying on a single reasoning path, the problem is handed simultaneously to three agents, each employing a cognitively distinct strategy. Their answers are aggregated via majority vote, producing an ensemble that is more robust than any individual solver.

**Key contributions of this methodology:**

- 🔀 A **strategy-diverse parallel ensemble** that eliminates single-path reasoning bias
- 📐 Three prompt-engineered agents: CoT, Formula-First, and Backward Reasoning
- 📊 Evaluation across **5 SLMs** ranging from 1.5B to 7B parameters on **SVAMP** (100 samples)
- 📈 Per-strategy accuracy breakdown and ensemble **disagreement analysis**
- ⚙️ A clean, YAML-driven, modular codebase that runs on CPU, CUDA, and Apple Silicon (MPS)

---

## 🏗️ Architecture

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

## 🧩 Methodology

### Pipeline Decomposition

The central premise of this methodology is that **no single reasoning style dominates across all problem types**. By decomposing the pipeline into parallel, cognitively distinct agents and reconciling their outputs through democratic voting, the ensemble captures a richer solution space than any individual solver.

The pipeline is implemented in two modes:

| Mode | Class | Description |
|------|-------|-------------|
| `solver` | `SolverOnlyPipeline` | **Baseline** — single Chain-of-Thought agent |
| `diverse_ensemble` | `DiverseEnsemblePipeline` | **Full methodology** — 3 parallel strategy-diverse agents |

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

## 🤖 Supported Models

Any `AutoModelForCausalLM`-compatible HuggingFace model is supported. The following were evaluated in this study:

| Model | Parameters | Device Support |
|-------|-----------|----------------|
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | 1.5B | CPU / MPS |
| `google/gemma-2-2b` | 2B | CPU / MPS / CUDA |
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

## 📊 Results

> Evaluated on **SVAMP** · 100 samples · `diverse_ensemble` pipeline · greedy decoding (`temperature=0.0`)

### Ensemble Accuracy

| Model | Params | Ensemble Acc | Total Tokens | Avg Tokens/Q |
|-------|--------|:------------:|:------------:|:------------:|
| DeepSeek-R1-Distill-Qwen-1.5B | 1.5B | 30.0% | 186,231 | 1,862 |
| Gemma2-2B | 2B | 68.0% | 77,872 | 779 |
| **LLaMA 3.2-3B** | **3B** | **81.0%** | **88,562** | **886** |
| Mistral-7B | 7B | 67.0% | 101,976 | 1,020 |
| Qwen2.5-7B | 7B | 24.0% | 191,172 | 1,912 |

### Per-Strategy Accuracy Breakdown

| Model | Chain-of-Thought | Formula-First | Backward |
|-------|:----------------:|:-------------:|:--------:|
| DeepSeek-R1-Distill-Qwen-1.5B | 20.0% | 22.0% | **51.0%** |
| Gemma2-2B | **64.0%** | 51.0% | 47.0% |
| LLaMA 3.2-3B | **77.0%** | 60.0% | 64.0% |
| Mistral-7B | **66.0%** | 45.0% | 47.0% |
| Qwen2.5-7B | 22.0% | **57.0%** | 13.0% |

### Ensemble Diversity Analysis

| Model | Disagreement Rate | Unique Correct Rate |
|-------|:-----------------:|:-------------------:|
| DeepSeek-R1-Distill-Qwen-1.5B | 99% | 31% |
| Gemma2-2B | 83% | 25% |
| LLaMA 3.2-3B | 61% | 12% |
| Mistral-7B | 88% | 27% |
| Qwen2.5-7B | 100% | 50% |

### 🔍 Key Observations

- 🏆 **LLaMA 3.2-3B achieves the highest ensemble accuracy (81%)** — outperforming both 7B models, demonstrating that parameter count alone does not determine reasoning quality.
- 🔄 **Backward Reasoning is the dominant strategy for DeepSeek-R1-1.5B (51%)** — the model's distilled reasoning style aligns naturally with goal-first decomposition.
- 📐 **Formula-First is the only viable strategy for Qwen2.5-7B (57%)** — the model shows near-random performance on CoT and Backward despite its size.
- 🤝 **Qwen2.5-7B has a 100% disagreement rate** — the three strategies never agree, and 50% of the time only one solver is correct. High diversity, low consensus.
- 📉 **Majority vote can be hurt by high disagreement** — when strategies rarely agree and errors are diverse (Qwen2.5-7B), voting collapses rather than helps.
- ✅ **Gemma2-2B punches above its weight** — at just 2B parameters, it achieves 68% accuracy using only 779 tokens per question on average.

---

## 📁 Project Structure

```
SLM-based-MultiAgent-Frameworks/
│
├── Pipeline_decomposition/          # Methodology 1 (this folder)
│   ├── slm_comm/
│   │   ├── agents.py               # ChainOfThought, FormulaFirst, Backward solvers
│   │   ├── pipeline.py             # SolverOnlyPipeline & DiverseEnsemblePipeline
│   │   ├── model.py                # HuggingFaceTextGenerator (CausalLM wrapper)
│   │   ├── metrics.py              # is_correct(), normalize_answer(), summarize_results()
│   │   ├── data.py                 # Dataset loaders: SVAMP / GSM8K / MATH
│   │   ├── config.py               # Dataclass configs + YAML loader + validator
│   │   ├── communication.py        # (Retained for reference; not active in this methodology)
│   │   └── utils.py                # set_seed(), save_json(), extract_gsm8k_answer()
│   │
│   ├── config.yaml                 # Experiment configuration
│   ├── run.py                      # Entry point
│   └── outputs/                    # Per-run prediction & summary JSONs
│
└── README.md
```

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/Codewithsayanjib/SLM-based-MultiAgent-Frameworks.git
cd SLM-based-MultiAgent-Frameworks/Pipeline_decomposition

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

### 1. Configure your experiment

Edit `config.yaml` to select your model, dataset, and pipeline:

```yaml
seed: 42
output_dir: outputs/run_01/llama3.2_3b

dataset:
  name: svamp          # svamp | gsm8k | math
  split: train
  max_samples: 100

model:
  name: meta-llama/Llama-3.2-3B
  device: mps          # mps | cpu | cuda
  dtype: float32       # float32 (CPU/MPS) | float16 (CUDA)
  load_in_4bit: false  # true for 7B+ on GPU

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

### 2. Run the experiment

```bash
python run.py --config config.yaml
```

### 3. View outputs

```
outputs/run_01/llama3.2_3b/
├── predictions.json     # Per-example: question, gold, all strategy outputs, final prediction
└── summary.json         # Accuracy, token cost, per-strategy breakdown, disagreement rate
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

### Run as Baseline (Single Solver)

```yaml
pipeline:
  type: solver    # single chain-of-thought agent only
```

---

## 🛠️ Configuration Reference

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
| `pipeline.type` | `str` | `diverse_ensemble` / `solver` |
| `ensemble.strategies` | `list` | Any combination of `chain_of_thought`, `formula_first`, `backward` |
| `generation.max_new_tokens` | `int` | Max tokens to generate per agent call |
| `generation.temperature` | `float` | Sampling temperature (`0.0` = greedy) |
| `generation.do_sample` | `bool` | Enable stochastic sampling |

---

## 💡 Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Parallel solvers, no communication** | Agents solve independently; no inter-agent message passing. This isolates strategy diversity as the sole variable and avoids error propagation between agents. |
| **`FINAL_ANSWER:` protocol** | Structured extraction tag prevents the model from producing open-ended prose. Fallback to last number handles non-compliant outputs gracefully. |
| **`repetition_penalty=1.2`** | Breaks infinite repetition loops observed in `formula_first` strategy on some models. |
| **`float32` on CPU/MPS** | `float16` produces NaN values on non-CUDA devices. `float32` is enforced as the safe default. |
| **Tie-breaking by first strategy** | When the vote is split three ways, Chain-of-Thought is given priority — empirically the most reliable individual strategy across models. |
| **Lazy checkpoint saves** | `predictions.json` is written every 10 examples, ensuring no results are lost on crash or OOM. |

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

**More methodologies coming soon** · Stay tuned for Methodology 1

<br/>

Made with 🔬 at **Jadavpur University**

<br/>

[![GitHub](https://img.shields.io/badge/GitHub-Codewithsayanjib-181717?style=for-the-badge&logo=github)](https://github.com/Codewithsayanjib)

</div>
