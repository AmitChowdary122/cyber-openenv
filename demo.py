"""Interactive CLI demo for CyberSOC Arena.

Runs ONE randomly-selected scenario, prints the alert and asset inventory, and
either:
  - lets you (a human) pick actions step-by-step (default)
  - runs an automated baseline (--agent {random,heuristic,untrained,trained})

Use this to feel the loop and to capture trace screenshots for the README.

Usage:
  python demo.py
  python demo.py --agent heuristic --scenario phishing_lateral
  python demo.py --agent trained --checkpoint runs/policy_sft
"""

from __future__ import annotations

import argparse
import json

from baselines import HeuristicAgent, RandomAgent, TrainedPolicyAgent, UntrainedPriorAgent
from cybersoc_arena.actions import ALL_ACTIONS
from cybersoc_arena.env import CyberSOCEnv
from cybersoc_arena.observations import render_observation_text
from cybersoc_arena.scenarios import SCENARIO_TYPES


def _box(title: str, body: str):
    bar = "-" * max(40, min(80, len(title) + 4))
    print(bar)
    print(title)
    print(bar)
    print(body)
    print()


def human_action(obs):
    print()
    print("Pick an action_type:")
    for i, a in enumerate(sorted(ALL_ACTIONS), 1):
        print(f"  {i:2}. {a}")
    raw = input("> action (number, name, or full JSON): ").strip()
    if raw.startswith("{"):
        return json.loads(raw)
    try:
        idx = int(raw) - 1
        a = sorted(ALL_ACTIONS)[idx]
    except ValueError:
        a = raw
    target = input("  ip/host/entity (blank to skip): ").strip() or None
    summary = None
    if a in ("escalate_incident", "close_as_benign"):
        summary = input("  summary: ").strip()
    out = {"action_type": a}
    if target:
        if a in ("inspect_endpoint", "isolate_host"):
            out["host"] = target
        elif a == "correlate_events":
            out["entity"] = target
        else:
            out["ip"] = target
    if summary:
        out["summary"] = summary
    return out


def agent_for(name: str, checkpoint: str = None):
    if name == "random":
        return RandomAgent()
    if name == "heuristic":
        return HeuristicAgent()
    if name == "untrained":
        return UntrainedPriorAgent()
    if name == "trained":
        return TrainedPolicyAgent(model_path=checkpoint)
    raise ValueError(f"Unknown agent: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default=None, choices=[None, *SCENARIO_TYPES])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--agent", type=str, default=None,
                        choices=[None, "random", "heuristic", "untrained", "trained"],
                        help="Run an automated agent instead of human input.")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    env = CyberSOCEnv()
    obs = env.reset(scenario_type=args.scenario, seed=args.seed)

    state0 = env.state()
    _box(
        f"  CyberSOC Arena — scenario: {state0['scenario_type']}  "
        f"(is_benign={state0['is_benign']})",
        render_observation_text(obs),
    )

    auto = agent_for(args.agent, args.checkpoint) if args.agent else None
    total_reward = 0.0
    breakdowns = []
    while True:
        if auto:
            action = auto.act(obs)
            print(f"agent[{auto.name}] -> {json.dumps(action)}")
        else:
            action = human_action(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        breakdowns.append({"step": env.state()["step"], "reward": reward, "breakdown": info.get("breakdown", {})})
        print()
        print(f"  step reward = {reward:+.3f}    breakdown = {info.get('breakdown', {})}")
        print()
        print(render_observation_text(obs))
        if done:
            break

    final = env.state()
    _box(
        "  EPISODE END  ",
        (
            f"terminal action  : {final['terminal_action']}\n"
            f"terminal correct : {final['terminal_correct']}\n"
            f"steps used       : {final['step']} / {state0['scenario_type']} budget\n"
            f"total reward     : {total_reward:+.3f}\n"
            f"final summary    : {final.get('final_summary') or '(none)'}"
        ),
    )


if __name__ == "__main__":
    main()
