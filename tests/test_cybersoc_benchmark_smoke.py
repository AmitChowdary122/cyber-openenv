"""Benchmark + baseline smoke test — runs in seconds, no model loads."""

from baselines import HeuristicAgent, RandomAgent, UntrainedPriorAgent
from benchmark_cybersoc import aggregate, run_episode
from cybersoc_arena.env import CyberSOCEnv
from cybersoc_arena.scenarios import SCENARIO_TYPES


def test_each_baseline_completes_an_episode():
    env = CyberSOCEnv()
    for agent in (RandomAgent(seed=1), UntrainedPriorAgent(seed=1), HeuristicAgent()):
        rec = run_episode(env, agent, "phishing_lateral", seed=42)
        assert isinstance(rec["total_reward"], float)
        assert rec["agent"] == agent.name


def test_aggregate_produces_overall_metrics():
    env = CyberSOCEnv()
    agents = [RandomAgent(seed=0), HeuristicAgent()]
    records = []
    for agent in agents:
        for s in SCENARIO_TYPES[:2]:  # subset for speed
            for ep in range(3):
                records.append(run_episode(env, agent, s, seed=ep * 7 + hash(s) % 100))
    summary = aggregate(records)
    for ag in ("random", "heuristic"):
        o = summary[ag]["overall"]
        for k in ("mean_reward", "success_rate", "fp_rate", "miss_rate", "avg_steps"):
            assert k in o
