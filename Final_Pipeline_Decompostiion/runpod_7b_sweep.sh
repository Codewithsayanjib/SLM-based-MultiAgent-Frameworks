#!/usr/bin/env bash
# One-shot RunPod driver for the A1 7B sweep (Qwen2.5-7B + Mistral-7B),
# SVAMP N=300, fp16, all 3 communication modes — identical settings to the
# local small-model runs. Run this INSIDE a RunPod GPU pod's web terminal.
#
#   bash runpod_7b_sweep.sh
#
# It installs deps, logs into HF (for gated Mistral), runs the sweep to
# outputs_n300/, and tars the two 7B result folders for download.
set -euo pipefail

echo "==> GPU check"; nvidia-smi || { echo "No GPU visible — pick a GPU pod"; exit 1; }

echo "==> Installing dependencies"
pip install -q "transformers>=4.44" "datasets>=2.20" accelerate sentencepiece "huggingface_hub>=0.23"

cd "$(dirname "$0")"                       # -> Final_Pipeline_Decompostiion
echo "==> Working dir: $(pwd)"

echo "==> Hugging Face login (needed for gated Mistral-7B)"
if [ -n "${HF_TOKEN:-}" ]; then
  huggingface-cli login --token "$HF_TOKEN"
else
  echo "    No HF_TOKEN env set. You'll be prompted to paste a token."
  echo "    (Accept the Mistral license first: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)"
  huggingface-cli login
fi

echo "==> Running sweep (Qwen2.5-7B + Mistral-7B, N=300, fp16, cuda) — resumable"
python run_sweep.py --n 300 --models large --device cuda --dtype float16 --out-root outputs_n300

echo "==> Packaging results"
tar czf /workspace/outputs_n300_7b.tar.gz outputs_n300/qwen2.5_7b outputs_n300/mistral-7b
echo ""
echo "================= SUMMARIES ================="
for s in outputs_n300/qwen2.5_7b/*/summary.json outputs_n300/mistral-7b/*/summary.json; do
  echo "$s:"; cat "$s"; echo
done
echo "============================================="
echo ""
echo "DONE. Archive: /workspace/outputs_n300_7b.tar.gz"
echo "To pull it to your Mac, run on the pod:"
echo "    runpodctl send /workspace/outputs_n300_7b.tar.gz"
echo "then on your Mac:  runpodctl receive <the-code-it-prints>"
