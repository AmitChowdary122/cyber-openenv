"""CyberSOC Arena benchmark runner.

Compares: random / untrained / heuristic / trained agents across all scenario
types. Outputs:
  runs/benchmark_results.json    — per-agent / per-scenario metrics
  runs/benchmark_table.txt       — pretty-printed table

Usage:
  python benchmark.py --episodes 50 --task all
  python benchmark.py --episodes 30 --task multi_stage_chain --checkpoint runs/policy_sft
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from typing import Any, Dict, List

from baselines import HeuristicAgent, RandomAgent, TrainedPolicyAgent, UntrainedPriorAgent
from cybersoc_arena.env import CyberSOCEnv
from cybersoc_arena.scenarios import SCENARIO_TYPES


def run_episode(env: CyberSOCEnv, agent, scenario_type: str, seed: int) -> Dict[str, Any]:
    obs = env.reset(scenario_type=scenario_type, seed=seed)
    if hasattr(agent, "_step"):       # reset stateful baselines
        agent._step = 0
    if hasattr(agent, "_asked"):
        agent._asked = set()
    total_reward = 0.0
    steps = 0
    info_last: Dict[str, Any] = {}
    done = False
    while not done:
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        info_last = info
    state = env.state()
    return {
        "agent": agent.name,
        "scenario_type": scenario_type,
        "total_reward": total_reward,
        "steps": steps,
        "terminal": info_last.get("terminal"),
        "correct": bool(state.get("terminal_correct") or info_last.get("correct")),
        "is_benign": state.get("is_benign"),
        "false_positive": _is_false_positive(info_last, state),
        "missed_attacker": _is_missed_attacker(info_last, state),
    }


def _is_false_positive(info: Dict[str, Any], state: Dict[str, Any]) -> bool:
    """Wrong isolate/identify/escalate on a benign scenario, or wrong target on a malicious one."""
    term = info.get("terminal")
    if state.get("is_benign"):
        return term in ("isolate_host", "identify_attacker", "escalate_incident")
    return term in ("isolate_host", "identify_attacker") and not info.get("correct", False)


def _is_missed_attacker(info: Dict[str, Any], state: Dict[str, Any]) -> bool:
    """Ran out of budget, or closed as benign, when the scenario was actually malicious."""
    if state.get("is_benign"):
        return False
    term = info.get("terminal")
    if term == "budget_exhausted":
        return True
    if term == "close_as_benign":
        return True
    return False


def aggregate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_agent: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        by_agent.setdefault(r["agent"], []).append(r)

    summary: Dict[str, Any] = {}
    for agent_name, rows in by_agent.items():
        per_scenario: Dict[str, Dict[str, float]] = {}
        for s in SCENARIO_TYPES:
            sub = [r for r in rows if r["scenario_type"] == s]
            if not sub:
                continue
            per_scenario[s] = {
                "n":             len(sub),
                "mean_reward":   statistics.mean(r["total_reward"] for r in sub),
                "success_rate":  sum(1 for r in sub if r["correct"]) / len(sub),
                "fp_rate":       sum(1 for r in sub if r["false_positive"]) / len(sub),
                "miss_rate":     sum(1 for r in sub if r["missed_attacker"]) / len(sub),
                "avg_steps":     statistics.mean(r["steps"] for r in sub),
            }
        summary[agent_name] = {
            "per_scenario": per_scenario,
            "overall": {
                "n":            len(rows),
                "mean_reward":  statistics.mean(r["total_reward"] for r in rows),
                "success_rate": sum(1 for r in rows if r["correct"]) / len(rows),
                "fp_rate":      sum(1 for r in rows if r["false_positive"]) / len(rows),
                "miss_rate":    sum(1 for r in rows if r["missed_attacker"]) / len(rows),
                "avg_steps":    statistics.mean(r["steps"] for r in rows),
            },
        }
    return summary


def render_table(summary: Dict[str, Any]) -> str:
    order = ["random", "untrained_prior", "heuristic", "trained_policy"]
    rows = []
    rows.append(f"{'Agent':<22} {'Mean R':>9} {'Success':>8} {'FP':>6} {'Miss':>6} {'Steps':>6}")
    rows.append("-" * 60)
    for ag in order:
        if ag not in summary:
            continue
        o = summary[ag]["overall"]
        rows.append(
            f"{ag:<22} {o['mean_reward']:>+9.3f} "
            f"{o['success_rate']*100:>7.1f}% "
            f"{o['fp_rate']*100:>5.1f}% "
            f"{o['miss_rate']*100:>5.1f}% "
            f"{o['avg_steps']:>6.1f}"
        )
    rows.append("")
    rows.append("Per-scenario success rate (%):")
    rows.append(f"{'Agent':<22}" + "".join(f"{s[:14]:>16}" for s in SCENARIO_TYPES))
    for ag in order:
        if ag not in summary:
            continue
        per = summary[ag]["per_scenario"]
        cells = "".join(
            f"{per.get(s, {}).get('success_rate', 0) * 100:>15.1f}%"
            for s in SCENARIO_TYPES
        )
        rows.append(f"{ag:<22}{cells}")
    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50,
                        help="Episodes per (agent, scenario) pair")
    parser.add_argument("--task", type=str, default="all",
                        choices=["all", *SCENARIO_TYPES])
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a fine-tuned model for trained_policy agent")
    parser.add_argument("--out", type=str, default="runs")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    scenarios = SCENARIO_TYPES if args.task == "all" else [args.task]

    agents = [
        RandomAgent(seed=args.seed),
        UntrainedPriorAgent(seed=args.seed),
        HeuristicAgent(),
        TrainedPolicyAgent(model_path=args.checkpoint),
    ]
    env = CyberSOCEnv()

    records: List[Dict[str, Any]] = []
    t0 = time.time()
    for agent in agents:
        for s in scenarios:
            for ep in range(args.episodes):
                seed = args.seed + ep * 7919 + hash(s) % 9973 + hash(agent.name) % 4093
                rec = run_episode(env, agent, s, seed=seed)
                records.append(rec)
        elapsed = time.time() - t0
        print(f"  [{elapsed:6.1f}s] {agent.name} done")

    summary = aggregate(records)
    table = render_table(summary)
    print()
    print(table)

    with open(os.path.join(args.out, "benchmark_results.json"), "w") as f:
        json.dump({"records": records, "summary": summary}, f, indent=2)
    with open(os.path.join(args.out, "benchmark_table.txt"), "w") as f:
        f.write(table + "\n")
    print(f"\nSaved -> {args.out}/benchmark_results.json")


if __name__ == "__main__":
    main()
