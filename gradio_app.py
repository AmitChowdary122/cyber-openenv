"""Gradio Space app for CyberSOC Arena.

Launches a chat-style UI where the user picks a scenario, watches the agent
take actions, and sees the running reward breakdown. Suitable for HF Spaces.

Usage:
  python app.py
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

import gradio as gr

from baselines import HeuristicAgent, RandomAgent, TrainedPolicyAgent, UntrainedPriorAgent
from cybersoc_arena.env import CyberSOCEnv
from cybersoc_arena.observations import render_observation_text
from cybersoc_arena.scenarios import SCENARIO_TYPES

AGENTS = {
    "Random":          lambda: RandomAgent(),
    "Untrained prior": lambda: UntrainedPriorAgent(),
    "Heuristic":       lambda: HeuristicAgent(),
    "Trained":         lambda: TrainedPolicyAgent(),
}


def run(scenario: str, agent_name: str, seed: int) -> Tuple[str, str, str]:
    env = CyberSOCEnv()
    obs = env.reset(scenario_type=scenario or None, seed=seed)
    agent = AGENTS[agent_name]()

    trace_lines: List[str] = [f"### Scenario: `{env.state()['scenario_type']}` (seed {seed})", ""]
    trace_lines.append(f"**Initial alert:** {obs['alert']['summary']}")
    trace_lines.append("")

    total = 0.0
    breakdown_rows: List[str] = []
    done = False
    step_idx = 0
    while not done:
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)
        total += reward
        step_idx += 1
        trace_lines.append(
            f"- **step {step_idx}** `{action.get('action_type')}` "
            f"target=`{action.get('ip') or action.get('host') or action.get('entity') or '-'}` "
            f"reward=**{reward:+.3f}**"
        )
        breakdown_rows.append(f"step {step_idx}: {info.get('breakdown', {})}")

    final = env.state()
    summary = (
        f"### Final outcome\n\n"
        f"- terminal action: `{final['terminal_action']}`\n"
        f"- correct: **{final['terminal_correct']}**\n"
        f"- steps: {final['step']}\n"
        f"- total reward: **{total:+.3f}**\n"
        f"- final summary: {final.get('final_summary') or '(none)'}\n"
    )

    return "\n".join(trace_lines), summary, "\n".join(breakdown_rows)


def build():
    with gr.Blocks(title="CyberSOC Arena") as demo:
        gr.Markdown(
            "# CyberSOC Arena\n"
            "An OpenEnv environment where an LLM agent triages SOC incidents under a step budget. "
            "Pick a scenario, pick an agent, watch the trace."
        )
        with gr.Row():
            scenario = gr.Dropdown(
                choices=[""] + list(SCENARIO_TYPES),
                value="",
                label="Scenario (blank = random)",
            )
            agent_name = gr.Dropdown(
                choices=list(AGENTS.keys()), value="Heuristic", label="Agent",
            )
            seed = gr.Number(value=42, label="Seed", precision=0)
        run_btn = gr.Button("Run episode", variant="primary")
        with gr.Row():
            trace = gr.Markdown(label="Action trace")
            outcome = gr.Markdown(label="Outcome")
        breakdown = gr.Textbox(label="Reward breakdown per step", lines=12)
        run_btn.click(run, [scenario, agent_name, seed], [trace, outcome, breakdown])
    return demo


if __name__ == "__main__":
    build().launch()
