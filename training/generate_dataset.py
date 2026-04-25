"""Build a SFT training dataset from heuristic-agent rollouts.

For each episode, we record (observation_prompt, action_json) pairs and only
keep episodes where the heuristic actually succeeded (correct terminal). This
gives the LM clean expert demonstrations.

Output:
  runs/sft_dataset.jsonl

Each line:
{
  "prompt":     "<system prompt + rendered observation>",
  "completion": "<JSON action>",
  "scenario":   "phishing_lateral",
  "step":       3,
  "ep_reward":  1.7
}

Usage:
  python -m training.generate_dataset --episodes 600 --out runs/sft_dataset.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Dict, List

from baselines.heuristic_agent import HeuristicAgent
from baselines.trained_policy_agent import _SYSTEM_PROMPT
from cybersoc_arena.env import CyberSOCEnv
from cybersoc_arena.observations import render_observation_text
from cybersoc_arena.scenarios import SCENARIO_TYPES


def _format_prompt(obs: Dict[str, Any]) -> str:
    body = render_observation_text(obs)
    return (
        f"<|system|>\n{_SYSTEM_PROMPT}\n<|end|>\n"
        f"<|user|>\n{body}\n\nRespond with a single JSON action.\n<|end|>\n"
        f"<|assistant|>\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=600,
                        help="Total episodes to attempt across all scenarios")
    parser.add_argument("--out", type=str, default="runs/sft_dataset.jsonl")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--keep-failed", action="store_true",
                        help="Also keep episodes where heuristic failed (for variety)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rng = random.Random(args.seed)
    env = CyberSOCEnv()

    pairs: List[Dict[str, Any]] = []
    kept_ep = 0
    for ep in range(args.episodes):
        scenario = rng.choice(SCENARIO_TYPES)
        agent = HeuristicAgent()
        obs = env.reset(scenario_type=scenario, seed=args.seed + ep * 7919)
        ep_pairs: List[Dict[str, Any]] = []
        ep_reward = 0.0
        done = False
        while not done:
            action = agent.act(obs)
            prompt = _format_prompt(obs)
            ep_pairs.append({
                "prompt": prompt,
                "completion": json.dumps(action),
                "scenario": scenario,
                "step": env.state()["step"],
            })
            obs, reward, done, info = env.step(action)
            ep_reward += reward

        state = env.state()
        if state.get("terminal_correct") or args.keep_failed:
            for p in ep_pairs:
                p["ep_reward"] = ep_reward
            pairs.extend(ep_pairs)
            kept_ep += 1
        if (ep + 1) % 100 == 0:
            print(f"  generated ep {ep+1}/{args.episodes}, kept {kept_ep} episodes, {len(pairs)} pairs")

    with open(args.out, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    print(f"\nWrote {len(pairs)} (prompt, completion) pairs to {args.out}")
    print(f"  from {kept_ep}/{args.episodes} episodes")


if __name__ == "__main__":
    main()
