#!/usr/bin/env bash
# Run the real Unsloth + GRPO training of Qwen2.5-0.5B-Instruct on an
# HF Jobs T4-small (~$0.40/hr). Expects HF_TOKEN to be exported.
#
# Usage:
#     export HF_TOKEN=hf_xxx
#     ./run_hf_job.sh
#
# When the run completes, fetch the artifacts back from the job and
# commit the contents of assets/ + runs/grpo/.
set -euo pipefail

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: set HF_TOKEN before running."
    exit 1
fi

hf jobs uv run \
    --flavor t4-small \
    --env HF_TOKEN="${HF_TOKEN}" \
    --secret HF_TOKEN="${HF_TOKEN}" \
    train_unsloth_grpo.py
