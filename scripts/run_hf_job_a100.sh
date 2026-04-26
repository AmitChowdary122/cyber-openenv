#!/usr/bin/env bash
# Launch a CyberSOC Arena GRPO training run on Hugging Face Jobs.
#
# DEFAULT FLAVOR: l40sx1 (1x Nvidia L40S 48GB, $1.80/hr).
# We default to L40S over A100 because the A100 queue is regularly 30+ min
# during India daytime, while L40S almost always schedules within a few
# minutes. Qwen2.5-1.5B-Instruct fits comfortably in 48GB at bf16, and the
# strong training schedule below produces clean curves on this GPU.
#
# Override the flavor with FLAVOR=h200, FLAVOR=a100-large, etc. via env vars.
#
# Prerequisites
# -------------
#   1. `pip install -U huggingface_hub` and `hf auth login`.
#   2. You have HF compute credits (the OpenEnv hackathon ships $30 per person).
#   3. The CyberSOC Arena environment is pushed to
#        https://huggingface.co/spaces/amit51/cybersoc-arena
#      (so the script URL below resolves).
#
# What this does
# --------------
#   - Allocates a single L40S 48GB ($1.80/hr) by default.
#   - Trains a Qwen2.5-1.5B-Instruct LoRA adapter for ~40 min using
#     trl.GRPOTrainer with the live CyberSOC Arena env as the reward source.
#   - 480 prompts x 3 epochs x 8 generations gives ~360 logged training
#     steps -- enough to produce a visibly improving reward curve.
#   - Runs a held-out greedy rollout BEFORE and AFTER training across all 6
#     scenarios so the per-scenario before/after plot is meaningful.
#   - Pushes the LoRA adapter, training_log.json, eval_results.json, and
#     three PNG plots to a model repo so they survive after the HF Job's
#     container is torn down. Default repo:
#       amit51/cybersoc-arena-qwen2.5-1.5b-grpo
#
# Estimated cost on the $30 hackathon credit:
#   - L40S (default):  ~$1.20  (~40 min @ $1.80/hr)
#   - A100-large:      ~$1.30  (~30 min @ $2.50/hr, but queue can be slow)
#   - H200:            ~$1.50  (~20 min @ $5.00/hr, fastest if available)
#
# Track in real time:
#   hf jobs ps                         # list jobs
#   hf jobs logs <JOB_ID> -f           # tail logs
#   https://huggingface.co/jobs        # browser dashboard

set -euo pipefail

REPO_ID="${REPO_ID:-amit51/cybersoc-arena}"
PUSH_TO="${PUSH_TO:-amit51/cybersoc-arena-qwen2.5-1.5b-grpo}"
FLAVOR="${FLAVOR:-l40sx1}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
NUM_PROMPTS="${NUM_PROMPTS:-480}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
EPOCHS="${EPOCHS:-3}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-192}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"

SCRIPT_URL="https://huggingface.co/spaces/${REPO_ID}/raw/main/scripts/train_hf_job.py"

echo "[run_hf_job]"
echo "  flavor             : ${FLAVOR}"
echo "  base model         : ${BASE_MODEL}"
echo "  num prompts        : ${NUM_PROMPTS}"
echo "  num generations    : ${NUM_GENERATIONS}"
echo "  epochs             : ${EPOCHS}"
echo "  max completion len : ${MAX_COMPLETION_LENGTH}"
echo "  per-device batch   : ${PER_DEVICE_BATCH}"
echo "  grad accumulation  : ${GRAD_ACCUM}"
echo "  push to            : ${PUSH_TO}"
echo "  script URL         : ${SCRIPT_URL}"
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
    --per-device-batch-size "${PER_DEVICE_BATCH}" \
    --gradient-accumulation-steps "${GRAD_ACCUM}" \
    --output-dir /tmp/cybersoc_grpo \
    --push-to-hub "${PUSH_TO}"

echo
echo "[run_hf_job] launched."
echo "Track job:    hf jobs ps    |    hf jobs logs <ID> -f"
echo "Browser:      https://huggingface.co/jobs"
echo "When done:    https://huggingface.co/${PUSH_TO}"
