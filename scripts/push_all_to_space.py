"""One-shot uploader: push every submission asset to the HF Space + model repo.

Avoids PowerShell's shell-side wildcard expansion by doing the upload through
huggingface_hub.HfApi.upload_folder, which takes a clean allow_patterns list
on the Python side. Run from the repo root:

    python scripts/push_all_to_space.py

Requires: huggingface_hub, and `hf auth login` already done (write-scope token).
"""
from __future__ import annotations

import os
import sys

from huggingface_hub import HfApi, upload_file

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SPACE_REPO = "amit51/cybersoc-arena"
MODEL_REPO = "amit51/cybersoc-arena-qwen2.5-1.5b-grpo"

# Files / patterns to mirror to the HF Space. Folder upload is one round trip.
SPACE_PATTERNS = [
    # Core package
    "cybersoc_arena/__init__.py",
    "cybersoc_arena/rubric.py",
    "cybersoc_arena/web_ui.py",
    "cybersoc_arena/env.py",
    "cybersoc_arena/curriculum.py",
    "cybersoc_arena/server.py",
    "cybersoc_arena/client.py",
    "cybersoc_arena/models.py",
    "cybersoc_arena/actions.py",
    "cybersoc_arena/rewards.py",
    "cybersoc_arena/scenarios.py",
    "cybersoc_arena/observations.py",
    "cybersoc_arena/state.py",
    # Root shim files for the OpenEnv validator
    "__init__.py",
    "models.py",
    "client.py",
    # Scripts (training + plot regeneration)
    "scripts/train_hf_job.py",
    "scripts/run_hf_job_l40s.sh",
    "scripts/regenerate_plots.py",
    "scripts/plot_curriculum_full.py",
    "scripts/push_all_to_space.py",
    # All current plots
    "assets/*.png",
    # 5-minute Colab demo notebook
    "notebooks/CyberSOC_Arena_demo.ipynb",
    # Top-level docs
    "README.md",
    "BLOG.md",
    "CHANGELOG.md",
    # Configuration the Space needs
    "Dockerfile",
    "openenv.yaml",
    "pyproject.toml",
]

# Files to push to the model repo: the corrected L40S plots + the polished model card
MODEL_FILES = [
    ("assets/grpo_reward_curve.png", "grpo_reward_curve.png"),
    ("assets/grpo_loss_curve.png", "grpo_loss_curve.png"),
    ("assets/grpo_baseline_compare.png", "grpo_baseline_compare.png"),
    ("assets/model_card.md", "README.md"),
]


def main() -> int:
    api = HfApi()
    print(f"[push] mirroring repo root to space: {SPACE_REPO}")
    api.upload_folder(
        folder_path=REPO_ROOT,
        repo_id=SPACE_REPO,
        repo_type="space",
        allow_patterns=SPACE_PATTERNS,
        commit_message="ship: Rubric wrapper + 6-tier curriculum plot + winning BLOG.md + L40S plot relabels",
    )
    print(f"[push] space updated -> https://huggingface.co/spaces/{SPACE_REPO}")

    print(f"\n[push] uploading corrected line plots to model repo: {MODEL_REPO}")
    for local, remote in MODEL_FILES:
        full = os.path.join(REPO_ROOT, local)
        if not os.path.exists(full):
            print(f"  [skip] {local} not found locally")
            continue
        upload_file(
            path_or_fileobj=full,
            path_in_repo=remote,
            repo_id=MODEL_REPO,
            repo_type="model",
            commit_message=f"Update {remote} (L40S titles)",
        )
        print(f"  [ok] {local} -> {MODEL_REPO}/{remote}")

    print(f"[push] model repo updated -> https://huggingface.co/{MODEL_REPO}")
    print("\n[push] DONE. Submission surfaces:")
    print(f"  Space   : https://huggingface.co/spaces/{SPACE_REPO}")
    print(f"  Model   : https://huggingface.co/{MODEL_REPO}")
    print(f"  GitHub  : https://github.com/AmitChowdary122/cyber-openenv  (push via git)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
