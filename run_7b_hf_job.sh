#!/usr/bin/env bash
# Launch the Qwen2.5-7B GRPO training run on HF Jobs A10G-small
# (~$1.00/hr, ~3-4 hrs total → ≈ $3–4).
#
# Usage:
#     export HF_TOKEN=hf_xxx               # MUST be a valid token
#     export WANDB_API_KEY=...             # optional; enables wandb
#     ./run_7b_hf_job.sh
#
# Tips:
#     - Refresh / verify HF_TOKEN at https://huggingface.co/settings/tokens
#     - Validate locally first:   hf auth whoami
#     - List GPU options:         hf jobs hardware
#     - Tail logs once launched:  hf jobs logs <job_id> --follow
set -euo pipefail

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: set HF_TOKEN before running."
    exit 1
fi

# Sanity check the token resolves to a user.
if ! hf auth whoami >/dev/null 2>&1; then
    echo "ERROR: hf auth whoami failed — HF_TOKEN is not valid."
    echo "       Refresh at https://huggingface.co/settings/tokens"
    exit 2
fi

EXTRA_ENV=()
if [[ -n "${WANDB_API_KEY:-}" ]]; then
    EXTRA_ENV+=(--env "WANDB_API_KEY=${WANDB_API_KEY}")
else
    echo "WARNING: WANDB_API_KEY is not set — wandb tracking will be disabled."
fi

echo "Submitting Qwen2.5-7B GRPO job on a10g-small..."
hf jobs uv run \
    --flavor a10g-small \
    --env "HF_TOKEN=${HF_TOKEN}" \
    "${EXTRA_ENV[@]}" \
    train_7b_hf_jobs.py
