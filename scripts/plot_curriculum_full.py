"""Generate a clean 6-tier curriculum staircase plot.

Runs a synthetic-reward driver against `CurriculumEnv` so the agent
demonstrably crosses every promotion threshold and unlocks Tier 5
(APT hunter). The synthetic reward is a noisy linear schedule designed
to clear each threshold with margin -- this is a *visualization* of the
mechanism, not a training run. Real RL training drives the same
machinery via `record_episode_reward()` after every episode.

Saves two plots to assets/:
  * curriculum_progress.png     -- single-panel staircase: episode -> tier
  * curriculum_combined.png     -- two panels: rolling mean + tier staircase
"""
from __future__ import annotations

import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np

from cybersoc_arena import CurriculumEnv, TIERS


TIER_COLORS = ["#9ecae1", "#74c476", "#fd8d3c", "#9e9ac8", "#fb6a4a", "#525252"]


def synth_reward_schedule(num_episodes: int, seed: int = 7):
    rng = np.random.default_rng(seed)
    rewards = []
    phase_targets = [0.65, 0.85, 1.00, 1.10, 1.25, 1.35]
    eps_per_phase = max(60, num_episodes // len(phase_targets))
    for target in phase_targets:
        for _ in range(eps_per_phase):
            r = float(rng.normal(loc=target, scale=0.18))
            r = max(-2.0, min(2.0, r))
            rewards.append(r)
        if len(rewards) >= num_episodes:
            break
    return rewards[:num_episodes]


def run_curriculum_demo(num_episodes: int, seed: int):
    random.seed(seed)
    np.random.seed(seed)

    cenv = CurriculumEnv(window=20, ratchet=True)
    rewards = synth_reward_schedule(num_episodes, seed=seed)

    tier_history = []
    rolling_history = []
    promotions = []

    prev_tier = cenv.tier
    for ep, r in enumerate(rewards):
        cenv.record_episode_reward(r)
        tier_history.append(cenv.tier)
        win = list(getattr(cenv, "_recent_rewards", []))
        rolling = float(np.mean(win)) if win else 0.0
        rolling_history.append(rolling)
        if cenv.tier != prev_tier:
            promotions.append((ep, cenv.tier, TIERS[cenv.tier].name))
            prev_tier = cenv.tier

    return {
        "rewards": rewards,
        "tier_history": tier_history,
        "rolling_history": rolling_history,
        "promotions": promotions,
        "final_tier": cenv.tier,
    }


def plot_combined(out_path, data):
    eps = np.arange(len(data["rewards"]))
    tier_history = data["tier_history"]
    rolling = data["rolling_history"]
    promotions = data["promotions"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 6.0), sharex=True,
                                   gridspec_kw={"height_ratios": [1.0, 1.4]})

    ax1.plot(eps, rolling, color="#1b9e77", lw=1.6, label="Rolling mean (window=20)")
    for t in TIERS[:-1]:
        ax1.axhline(t.advance_threshold, ls="--", lw=0.9, color="#888", alpha=0.7)
        ax1.text(eps[-1] + 2, t.advance_threshold,
                 f"T{t.index+1} unlock @ {t.advance_threshold:+.2f}",
                 fontsize=8, va="center", color="#555")
    ax1.set_ylabel("Rolling mean episode reward")
    ax1.set_title("CyberSOC Arena - CurriculumEnv self-promotion across all 6 tiers")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.3)

    ax2.step(eps, tier_history, where="post", color="#525252", lw=1.8)
    ax2.fill_between(eps, tier_history, step="post", alpha=0.15, color="#525252")
    for ep, tier, name in promotions:
        ax2.axvline(ep, color=TIER_COLORS[tier], lw=1.0, alpha=0.8)
        ax2.text(ep + 2, tier - 0.15,
                 f"ep {ep}: -> Tier {tier} ({name})",
                 fontsize=8, color="#222", va="top")
    ax2.set_yticks(list(range(6)))
    ax2.set_yticklabels([f"T{t.index} {t.name}" for t in TIERS], fontsize=8)
    ax2.set_xlabel("Episode #")
    ax2.set_ylabel("Unlocked tier")
    ax2.set_ylim(-0.5, 5.7)
    ax2.grid(alpha=0.3, axis="x")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[curriculum] wrote {out_path}")


def plot_progress_only(out_path, data):
    eps = np.arange(len(data["rewards"]))
    tier_history = data["tier_history"]
    promotions = data["promotions"]

    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    ax.step(eps, tier_history, where="post", color="#525252", lw=1.8)
    ax.fill_between(eps, tier_history, step="post", alpha=0.15, color="#525252")
    for ep, tier, name in promotions:
        ax.axvline(ep, color=TIER_COLORS[tier], lw=1.1, alpha=0.85)
        ax.text(ep + 2, tier - 0.15,
                f"ep {ep}: -> T{tier} {name}",
                fontsize=8.5, color="#222", va="top")
    ax.set_yticks(list(range(6)))
    ax.set_yticklabels([f"T{t.index} {t.name}" for t in TIERS], fontsize=9)
    ax.set_xlabel("Episode #")
    ax.set_ylabel("Unlocked tier")
    ax.set_title("CyberSOC Arena - adaptive curriculum unlocks all 6 tiers")
    ax.set_ylim(-0.5, 5.7)
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[curriculum] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=480)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-dir", default="assets")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    data = run_curriculum_demo(args.episodes, args.seed)

    print(f"[curriculum] {len(data['promotions'])} promotions:")
    for ep, tier, name in data["promotions"]:
        print(f"  episode {ep:>4}: -> Tier {tier} ({name})")
    print(f"[curriculum] final tier: {data['final_tier']} ({TIERS[data['final_tier']].name})")

    plot_progress_only(os.path.join(args.out_dir, "curriculum_progress.png"), data)
    plot_combined(os.path.join(args.out_dir, "curriculum_combined.png"), data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
