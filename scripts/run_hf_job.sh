#!/usr/bin/env bash
# Launch a CyberSOC Arena GRPO training run on Hugging Face Jobs (T4-medium).
#
# Prerequisites:
#   1. `pip install -U huggingface_hub` and `hf auth login`.
#   2. You have HF compute credits (the OpenEnv hackathon ships $30 per person).
#   3. The CyberSOC Arena environment is pushed to https://huggingface.co/spaces/amit51/cybersoc-arena
#
# Estimated cost: ~$0.10 per run (about 10 minutes on a t4-medium).
#
# Output: trained LoRA adapter at amit51/cybersoc-arena-qwen0.5b-grpo
#         + training_log.json / loss_curve.png / reward_curve.png in /tmp/cybersoc_grpo on the job

set -euo pipefail

REPO_ID="${REPO_ID:-amit51/cybersoc-arena}"
PUSH_TO="${PUSH_TO:-amit51/cybersoc-arena-qwen0.5b-grpo}"
FLAVOR="${FLAVOR:-t4-medium}"
SCRIPT_URL="https://huggingface.co/spaces/${REPO_ID}/raw/main/scripts/train_hf_job.py"

echo "[run_hf_job] flavor=${FLAVOR}  script=${SCRIPT_URL}  push=${PUSH_TO}"

hf jobs uv run \
  --flavor "${FLAVOR}" \
  --secrets HF_TOKEN \
  "${SCRIPT_URL}" \
    --base-model Qwen/Qwen2.5-0.5B-Instruct \
    --num-prompts 160 \
    --num-generations 4 \
    --epochs 1 \
    --output-dir /tmp/cybersoc_grpo \
    --push-to-hub "${PUSH_TO}"

echo "[run_hf_job] launched; track via:  hf jobs ps  or  https://huggingface.co/jobs"
