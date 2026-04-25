"""Lightweight evaluation harness: one trained checkpoint vs all baselines.

Identical interface to benchmark.py but emphasises before/after improvement
plots. Saves runs/eval_summary.json.

Usage:
  python evaluate.py --episodes 30 --checkpoint runs/policy_sft
"""

from __future__ import annotations

import argparse
import json
import os

from benchmark_cybersoc import aggregate, render_table, run_episode
from baselines import HeuristicAgent, RandomAgent, TrainedPolicyAgent, UntrainedPriorAgent
from cybersoc_arena.env import CyberSOCEnv
from cybersoc_arena.scenarios import SCENARIO_TYPES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--out", type=str, default="runs/eval_summary.json")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = CyberSOCEnv()
    agents = [
        RandomAgent(seed=args.seed),
        UntrainedPriorAgent(seed=args.seed),
        HeuristicAgent(),
        TrainedPolicyAgent(model_path=args.checkpoint),
    ]
    records = []
    for agent in agents:
        for s in SCENARIO_TYPES:
            for ep in range(args.episodes):
                seed = args.seed + ep * 7919 + hash(s + agent.name) % 99999
                records.append(run_episode(env, agent, s, seed=seed))
        print(f"  {agent.name} done")

    summary = aggregate(records)
    print(render_table(summary))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"records": records, "summary": summary}, f, indent=2)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
