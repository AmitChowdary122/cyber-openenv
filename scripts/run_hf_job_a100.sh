#!/usr/bin/env bash
# Launch a CyberSOC Arena GRPO training run on Hugging Face Jobs (A100 80 GB).
#
# Prerequisites
# -------------
#   1. `pip install -U huggingface_hub` and `hf auth login`.
#   2. You have HF compute credits (the OpenEnv hackathon ships $30 per person).
#   3. The CyberSOC Arena environment is pushed to
#        https://huggingface.co/spaces/amit51/cybersoc-arena
#
# What this does
# --------------
#   - Allocates a single A100 80 GB ($2.50/hr).
#   - Trains a Qwen2.5-1.5B-Instruct LoRA adapter for ~30 min using
#     trl.GRPOTrainer with the live CyberSOC Arena env as the reward source.
#   - Runs a held-out greedy rollout BEFORE and AFTER training across all 6
#     scenarios so the per-scenario before/after plot is meaningful.
#   - Pushes the LoRA adapter, training_log.json, eval_results.json, and three
#     PNG plots to a model repo so they survive after the HF Job's container
#     is torn down. Default repo: amit51/cybersoc-arena-qwen2.5-1.5b-grpo
#
# Estimated cost: ~$1.10 - $1.40 per run.
#
# Track the job in real time with:
#   hf jobs ps                       # list running jobs
#   hf jobs logs <JOB_ID> -f         # tail logs

set -euo pipefail

REPO_ID="${REPO_ID:-amit51/cybersoc-arena}"
PUSH_TO="${PUSH_TO:-amit51/cybersoc-arena-qwen2.5-1.5b-grpo}"
FLAVOR="${FLAVOR:-a100-large}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
NUM_PROMPTS="${NUM_PROMPTS:-320}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
EPOCHS="${EPOCHS:-2}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-192}"

SCRIPT_URL="https://huggingface.co/spaces/${REPO_ID}/raw/main/scripts/train_hf_job.py"

echo "[run_hf_job_a100]"
echo "  flavor       : ${FLAVOR}"
echo "  base model   : ${BASE_MODEL}"
echo "  num prompts  : ${NUM_PROMPTS}"
echo "  num gens     : ${NUM_GENERATIONS}"
echo "  epochs       : ${EPOCHS}"
echo "  max comp len : ${MAX_COMPLETION_LENGTH}"
echo "  push to      : ${PUSH_TO}"
echo "  script       : ${SCRIPT_URL}"
echo

hf jobs uv run \
  --flavor "${FLAVOR}" \
  --secrets HF_TOKEN \
  "${SCRIPT_URL}" \
    --base-model "${BASE_MODEL}" \
    --num-prompts "${NUM_PROMPTS}" \
    --num-generations "${NUM_GENERATIONS}" \
    --epochs "${EPOCHS}" \
    --max-completion-length "${MAX_COMPLETION_LENGTH}" \
    --output-dir /tmp/cybersoc_grpo \
    --push-to-hub "${PUSH_TO}"

echo
echo "[run_hf_job_a100] launched."
echo "Track job:  hf jobs ps   |   hf jobs logs <ID> -f"
echo "When done:  https://huggingface.co/${PUSH_TO}"
