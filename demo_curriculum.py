"""Demo: show CurriculumEnv tier unlocks over a sequence of synthetic episodes.

Drives CurriculumEnv with a sequence of scripted episode rewards (climbing as
the agent "improves") and prints each tier promotion along the way.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from typing import List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cybersoc_arena import CurriculumEnv  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=120)
    ap.add_argument("--out", default="training_runs/demo_curriculum.log")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    cenv = CurriculumEnv(window=10, promote_after=8, seed=args.seed)
    rng = random.Random(args.seed)

    lines: List[str] = []
    lines.append("CyberSOC Arena -- CurriculumEnv tier-unlock demo")
    lines.append(f"episodes={args.episodes}  window=10  promote_after=8  seed={args.seed}")
    lines.append("=" * 80)
    lines.append(f"  initial: tier={cenv.tier} ({cenv.tier_name}) pool={cenv.tier_pool}")

    for ep in range(1, args.episodes + 1):
        cenv.reset()
        # Synthetic reward growth: starts low, ramps up so each tier eventually
        # gets cleared. Adds stochastic noise to feel like real training.
        ramp = math.tanh((ep / args.episodes) * 3.5) * 1.5  # 0..~+1.5
        noise = rng.gauss(0.0, 0.35)
        ep_reward = ramp + noise
        cenv.record_episode_reward(ep_reward)
        if cenv.tier_changed:
            lines.append(
                f"  ep {ep:3d}: PROMOTED -> tier {cenv.tier} ({cenv.tier_name})  "
                f"pool size={len(cenv.tier_pool)}  rolling_mean={cenv.rolling_mean:+.2f}"
            )
        if ep % 20 == 0:
            lines.append(
                f"    ep {ep:3d}  tier={cenv.tier}  ep_reward={ep_reward:+.2f}  "
                f"rolling_mean={cenv.rolling_mean:+.2f}"
            )

    lines.append("")
    lines.append(f"FINAL: reached tier {cenv.tier} ({cenv.tier_name})")
    lines.append(f"  pool: {cenv.tier_pool}")

    text = "\n".join(lines)
    print(text)
    with open(args.out, "w") as f:
        f.write(text + "\n")
    print(f"\n[written] {args.out}")


if __name__ == "__main__":
    main()
