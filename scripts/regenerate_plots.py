"""Regenerate the GRPO loss + reward plots with L40S titles.

Pulls `training_log.json` from the HF model repo
(amit51/cybersoc-arena-qwen2.5-1.5b-grpo), replots the two line charts with
the correct hardware label, and saves them next to it. Use this whenever
the headline plot titles need a refresh without re-running the 2-hour
training job.

Usage:
    python scripts/regenerate_plots.py \
        --repo-id amit51/cybersoc-arena-qwen2.5-1.5b-grpo \
        --out-dir assets

Then upload back to the model repo with:
    hf upload amit51/cybersoc-arena-qwen2.5-1.5b-grpo \
        assets/grpo_loss_curve.png grpo_loss_curve.png
    hf upload amit51/cybersoc-arena-qwen2.5-1.5b-grpo \
        assets/grpo_reward_curve.png grpo_reward_curve.png
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.request

import matplotlib.pyplot as plt


def fetch_training_log(repo_id: str, dest: str) -> None:
    url = f"https://huggingface.co/{repo_id}/resolve/main/training_log.json"
    print(f"[regen] downloading {url}")
    urllib.request.urlretrieve(url, dest)
    print(f"[regen] saved -> {dest}")


def plot_curves(log_path: str, out_dir: str, model_label: str, hardware: str) -> None:
    with open(log_path) as f:
        log = json.load(f)

    steps_l = [r["step"] for r in log if "loss" in r]
    losses = [r["loss"] for r in log if "loss" in r]
    rsteps = [r["step"] for r in log if "reward" in r]
    rewards = [r["reward"] for r in log if "reward" in r]

    os.makedirs(out_dir, exist_ok=True)

    if losses:
        fig, ax = plt.subplots(figsize=(8.4, 4.4))
        ax.plot(steps_l, losses, color="#7570b3", lw=1.6, label="GRPO loss")
        ax.set_xlabel("Training step (#)")
        ax.set_ylabel("Loss")
        ax.set_title(f"CyberSOC Arena - GRPO loss ({model_label}, {hardware})")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        path = os.path.join(out_dir, "grpo_loss_curve.png")
        fig.savefig(path, dpi=130)
        plt.close(fig)
        print(f"[regen] wrote {path}  (n={len(losses)} points)")

    if rewards:
        fig, ax = plt.subplots(figsize=(8.4, 4.4))
        ax.plot(rsteps, rewards, color="#1b9e77", lw=1.6, label="mean reward")
        ax.set_xlabel("Training step (#)")
        ax.set_ylabel("Mean per-step env reward")
        ax.set_title(f"CyberSOC Arena - GRPO reward ({model_label}, {hardware})")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        path = os.path.join(out_dir, "grpo_reward_curve.png")
        fig.savefig(path, dpi=130)
        plt.close(fig)
        print(f"[regen] wrote {path}  (n={len(rewards)} points)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="amit51/cybersoc-arena-qwen2.5-1.5b-grpo")
    ap.add_argument("--out-dir", default="assets")
    ap.add_argument("--model-label", default="Qwen2.5-1.5B-Instruct")
    ap.add_argument("--hardware", default="L40S")
    ap.add_argument("--log-path", default=None,
                    help="Optional local training_log.json (skip download)")
    args = ap.parse_args()

    log_path = args.log_path
    if log_path is None:
        log_path = os.path.join(args.out_dir, "training_log.json")
        os.makedirs(args.out_dir, exist_ok=True)
        fetch_training_log(args.repo_id, log_path)

    plot_curves(log_path, args.out_dir, args.model_label, args.hardware)
    print("[regen] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
