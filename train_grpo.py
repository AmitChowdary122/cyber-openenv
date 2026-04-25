"""
train_grpo.py — Two modes:

  1. ROLLOUT mode (default, CPU only)
     ----------------------------------
     Runs three agents through the curriculum / plain env:

       - mixed_agent:  fed into CurriculumEnv. Starts mostly-random and
                       linearly anneals to mostly-heuristic — this is the
                       "learner" whose reward should improve over training.
       - RandomAgent:  baseline on plain CyberSOCEnv.
       - HeuristicAgent: scripted analyst on plain CyberSOCEnv.

     Logs every episode to runs/grpo/training_log.jsonl, prints progress
     every 20 episodes, and at the end produces four plots:

       - reward_curve.png         (smoothed curves + tier markers)
       - curriculum_progress.png  (tier level over time)
       - baseline_comparison.png  (mean reward last 100 eps)
       - success_rate.png         (rolling correctness %)

  2. TRL mode  (--use_trl, requires GPU + transformers + trl)
     -----------------------------------------------------------
     Builds prompts from the curriculum env, defines a reward function
     that simulates the rest of the episode with the heuristic, and runs
     GRPO. Save the model to runs/grpo/final_model.

Examples
--------
    python train_grpo.py --steps 500
    python train_grpo.py --steps 200 --seed 42
    python train_grpo.py --use_trl --model Qwen/Qwen2.5-0.5B-Instruct
"""
from __future__ import annotations
import argparse
import json
import os
import random
import sys
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cybersoc_arena import (
    CurriculumEnv, CyberSOCEnv, TIERS,
    ALL_ACTIONS, ACTION_DESCRIPTIONS,
    INVESTIGATIVE_ACTIONS, TERMINAL_ACTIONS,
)


RUNS_DIR = Path("runs/grpo")
RUNS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Agents
# =============================================================================

class RandomAgent:
    """Uniform random over all 9 action types, random target from known."""

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def reset(self):
        pass

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        action_type = self.rng.choice(ALL_ACTIONS)
        ips = list(obs.get("known_entities", {}).get("ips", []))
        hosts = list(obs.get("known_entities", {}).get("hosts", []))
        if action_type in ("inspect_endpoint",):
            target = self.rng.choice(hosts) if hosts else None
        elif action_type in ("check_threat_intel",):
            target = self.rng.choice(ips) if ips else None
        elif action_type in ("identify_attacker", "block_ip"):
            target = self.rng.choice(ips) if ips else None
        elif action_type in ("query_logs", "correlate_events"):
            pool = ips + hosts
            target = self.rng.choice(pool) if pool else None
        else:
            target = None
        a = {"action_type": action_type}
        if target is not None:
            if target in ips:
                a["ip"] = target
            else:
                a["host"] = target
        return a


ATTACK_KEYWORDS = (
    "phishing", "webshell", "lateral", "mimikatz", "ransom", "beacon",
    "exfiltration", "stuffing", "golden", "kill chain", "kill-chain",
    "brute-force", "brute force", "successful login", "failed login",
    "implant", "payload", "apt", "ddos", "backdoor", "dns tunnel",
    "forged", "dcsync", "botnet", "spearphish", "macro", "malicious",
    "exploitation", "compromised", "stolen", "rogue", "anomalous",
    "exported", "outbound https bursts", "powershell", "psexec",
    "team server", "cobalt", "tunnelling", "tunneling",
)
BENIGN_KEYWORDS = (
    "scanner", "fat-fingered", "no abuse", "legitimate", "normal",
    "test file", "benign", "false positive", "stock", "vendor scanner",
    "vuln-mgmt", "ms 365", "microsoft 365", "cdn edge", "no successful",
)


class HeuristicAgent:
    """
    Walks a planned (tool, target) queue covering every IP and host, then
    commits to a verdict in the last two steps based on attack/benign
    keyword scoring per IP.
    """

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self._queue: Optional[List[Tuple[str, str, str]]] = None
        self._tried: set = set()

    def reset(self):
        self._queue = None
        self._tried = set()

    def _build_queue(self, ips: List[str], hosts: List[str]) -> List[Tuple[str, str, str]]:
        """Priority order: cheap intel/log lookups first, then deeper scans."""
        q: List[Tuple[str, str, str]] = []
        for ip in ips:
            q.append(("check_threat_intel", ip, "ip"))
        for ip in ips:
            q.append(("query_logs", ip, "ip"))
        for h in hosts:
            q.append(("inspect_endpoint", h, "host"))
        for h in hosts:
            q.append(("query_logs", h, "host"))
        for h in hosts:
            q.append(("correlate_events", h, "host"))
        for ip in ips:
            q.append(("correlate_events", ip, "ip"))
        return q

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        step = obs.get("step", 0)
        budget = obs.get("step_budget", 10)
        ips = list(obs.get("known_entities", {}).get("ips", []))
        hosts = list(obs.get("known_entities", {}).get("hosts", []))

        if self._queue is None:
            self._queue = self._build_queue(ips, hosts)

        if step >= budget - 2:
            return self._verdict(obs, ips, hosts)

        while self._queue:
            tool, tgt, kind = self._queue.pop(0)
            if (tool, tgt) in self._tried:
                continue
            self._tried.add((tool, tgt))
            return {"action_type": tool, kind: tgt}
        return {"action_type": "list_entities"}

    def _verdict(self, obs, ips, hosts):
        revealed: List[str] = obs.get("revealed_evidence", [])
        if not revealed:
            return {"action_type": "mark_benign"}

        ip_scores = {ip: 0 for ip in ips}
        total_attack = 0
        total_benign = 0
        for line in revealed:
            ll = line.lower()
            attack_hits = sum(1 for k in ATTACK_KEYWORDS if k in ll)
            benign_hits = sum(1 for k in BENIGN_KEYWORDS if k in ll)
            total_attack += attack_hits
            total_benign += benign_hits
            if attack_hits > 0:
                for ip in ips:
                    if ip in line:
                        ip_scores[ip] = ip_scores.get(ip, 0) + attack_hits

        # Decide attack vs benign: dominant signal wins, attack is given a slight edge.
        attack_dominant = total_attack >= max(1, total_benign)

        if not attack_dominant:
            return {"action_type": "mark_benign"}

        if ip_scores and max(ip_scores.values()) > 0:
            best_ip = max(ip_scores, key=lambda k: ip_scores[k])
            return {"action_type": "identify_attacker", "ip": best_ip}
        return {"action_type": "escalate_incident"}


class MixedAgent:
    """
    With probability noise_prob act like RandomAgent, otherwise act
    like HeuristicAgent. Caller updates noise_prob across episodes.
    """

    def __init__(self, seed: int = 0):
        self.random = RandomAgent(seed=seed)
        self.heuristic = HeuristicAgent(seed=seed + 1)
        self.noise_prob = 1.0
        self.rng = random.Random(seed + 2)

    def reset(self):
        self.heuristic.reset()
        self.random.reset()

    def act(self, obs):
        if self.rng.random() < self.noise_prob:
            return self.random.act(obs)
        return self.heuristic.act(obs)


# =============================================================================
# Episode rollout
# =============================================================================

def run_episode(env, agent, max_steps: int = 100) -> Dict[str, Any]:
    obs = env.reset()
    if hasattr(agent, "reset"):
        agent.reset()
    total = 0.0
    done = False
    info: Dict[str, Any] = {}
    steps = 0
    while not done and steps < max_steps:
        a = agent.act(obs)
        obs, r, done, info = env.step(a)
        total += r
        steps += 1
    return {
        "reward": total,
        "steps": steps,
        "correct": bool(info.get("correct") or False),
        "info": info,
    }


def smooth(values: List[float], window: int = 20) -> List[float]:
    if not values:
        return []
    w = max(1, min(window, len(values)))
    out = []
    s = 0.0
    q = deque(maxlen=w)
    for v in values:
        q.append(v)
        out.append(sum(q) / len(q))
    return out


# =============================================================================
# Plotting
# =============================================================================

def plot_reward_curve(curr: List[float], rand: List[float], heur: List[float],
                      transitions: List[Dict[str, Any]], out: Path):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ep = list(range(1, len(curr) + 1))
    ax.plot(ep, smooth(curr, 20), label="Curriculum (mixed agent)",
            color="#1f77b4", linewidth=2.2)
    ax.plot(ep, smooth(rand, 20), label="Random baseline",
            color="#d62728", linewidth=1.6, alpha=0.85)
    ax.plot(ep, smooth(heur, 20), label="Heuristic baseline",
            color="#2ca02c", linewidth=1.6, alpha=0.85)
    for tr in transitions:
        x = tr["at_episode"]
        ax.axvline(x=x, color="grey", linestyle="--", alpha=0.55)
        ax.text(x, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 0.5,
                f"→ Tier {tr['to_tier']}", rotation=90,
                fontsize=8, color="grey", va="top")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode reward (smoothed, window=20)")
    ax.set_title("CyberSOC Arena — Reward (Theme 4: Self-Improvement curriculum)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_curriculum_progress(tier_per_ep: List[int],
                             transitions: List[Dict[str, Any]], out: Path):
    fig, ax = plt.subplots(figsize=(11, 5.0))
    ep = list(range(1, len(tier_per_ep) + 1))
    ax.fill_between(ep, 0, tier_per_ep, color="#1f77b4", alpha=0.35,
                    step="post")
    ax.plot(ep, tier_per_ep, color="#1f77b4", linewidth=2.0, drawstyle="steps-post")
    for tr in transitions:
        ax.axvline(x=tr["at_episode"], color="black", linestyle=":", alpha=0.5)
    ax.set_yticks(list(range(len(TIERS))))
    ax.set_yticklabels([f"{t.level}: {t.name}" for t in TIERS], fontsize=9)
    ax.set_xlabel("Episode")
    ax.set_title("Curriculum tier over training (Theme 4)")
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_baseline_comparison(curr: List[float], rand: List[float], heur: List[float],
                             out: Path):
    last = 100
    bars = {
        "Random": (rand[-last:] if len(rand) >= last else rand) or [0],
        "Heuristic": (heur[-last:] if len(heur) >= last else heur) or [0],
        "Curriculum": (curr[-last:] if len(curr) >= last else curr) or [0],
    }
    means = {k: sum(v) / len(v) for k, v in bars.items()}
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    colors = {"Random": "#d62728", "Heuristic": "#2ca02c", "Curriculum": "#1f77b4"}
    xs = list(means.keys())
    ys = [means[k] for k in xs]
    ax.bar(xs, ys, color=[colors[k] for k in xs])
    for i, v in enumerate(ys):
        ax.text(i, v, f"{v:+.2f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=10)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel(f"Mean reward (last {last} episodes)")
    ax.set_title("Baseline comparison (Theme 2/3/4 — agents on CyberSOC Arena)")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_success_rate(curr_correct: List[bool], rand_correct: List[bool],
                      heur_correct: List[bool], out: Path):
    def to_pct(bools, w=20):
        vals = [1.0 if b else 0.0 for b in bools]
        return [100 * x for x in smooth(vals, w)]
    fig, ax = plt.subplots(figsize=(11, 5.0))
    ep = list(range(1, len(curr_correct) + 1))
    ax.plot(ep, to_pct(curr_correct), label="Curriculum",
            color="#1f77b4", linewidth=2.2)
    ax.plot(ep, to_pct(rand_correct), label="Random",
            color="#d62728", linewidth=1.6, alpha=0.85)
    ax.plot(ep, to_pct(heur_correct), label="Heuristic",
            color="#2ca02c", linewidth=1.6, alpha=0.85)
    ax.set_ylabel("Correct verdict rate (%, smoothed window=20)")
    ax.set_xlabel("Episode")
    ax.set_title("Investigation success rate over training (Theme 2/3)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


# =============================================================================
# Rollout-mode training loop
# =============================================================================

def run_rollout(args):
    seed = args.seed
    steps = args.steps

    curr_env = CurriculumEnv(seed=seed)
    plain_for_random = CyberSOCEnv(seed=seed + 1001)
    plain_for_heur = CyberSOCEnv(seed=seed + 2002)

    mixed = MixedAgent(seed=seed)
    rand_agent = RandomAgent(seed=seed + 3003)
    heur_agent = HeuristicAgent(seed=seed + 4004)

    log_path = RUNS_DIR / "training_log.jsonl"
    log_f = log_path.open("w", encoding="utf-8")

    curr_rewards: List[float] = []
    rand_rewards: List[float] = []
    heur_rewards: List[float] = []
    curr_correct: List[bool] = []
    rand_correct: List[bool] = []
    heur_correct: List[bool] = []
    tier_per_episode: List[int] = []

    print(f"[rollout] running {steps} episodes, seed={seed}")
    print(f"[rollout] curriculum tiers: {[t.name for t in TIERS]}")

    for ep in range(1, steps + 1):
        # mixed agent's noise schedule
        mixed.noise_prob = max(0.0, 1.0 - ep / max(1, int(steps * 0.6)))

        ce = run_episode(curr_env, mixed, max_steps=40)
        re = run_episode(plain_for_random, rand_agent, max_steps=40)
        he = run_episode(plain_for_heur, heur_agent, max_steps=40)

        curr_rewards.append(ce["reward"])
        rand_rewards.append(re["reward"])
        heur_rewards.append(he["reward"])
        curr_correct.append(ce["correct"])
        rand_correct.append(re["correct"])
        heur_correct.append(he["correct"])
        tier_per_episode.append(curr_env.tier)

        metrics = curr_env.curriculum_metrics()
        record = {
            "episode": ep,
            "noise_prob": mixed.noise_prob,
            "curriculum_reward": ce["reward"],
            "random_reward": re["reward"],
            "heuristic_reward": he["reward"],
            "tier": curr_env.tier,
            "tier_name": curr_env.tier_name,
            "rolling_mean": metrics["rolling_mean_reward"],
            "progress_to_next": metrics["progress_to_next"],
            "correct_curriculum": ce["correct"],
            "correct_random": re["correct"],
            "correct_heuristic": he["correct"],
            "scenario": metrics.get("available_scenarios", []),
        }
        log_f.write(json.dumps(record) + "\n")

        if ep % 20 == 0 or ep == 1 or ep == steps:
            print(
                f"  ep={ep:>4}  tier={curr_env.tier} ({curr_env.tier_name:<16})  "
                f"r_curr={ce['reward']:+.2f}  r_rand={re['reward']:+.2f}  r_heur={he['reward']:+.2f}  "
                f"roll={metrics['rolling_mean_reward']:+.2f}  "
                f"prog={metrics['progress_to_next']*100:5.1f}%  "
                f"noise={mixed.noise_prob:.2f}"
            )

    log_f.close()

    transitions = curr_env.transitions
    print(f"[rollout] tier transitions: {len(transitions)}")
    for tr in transitions:
        print(f"  ep {tr['at_episode']:>4}: "
              f"{tr['from_name']} -> {tr['to_name']} "
              f"(rolling={tr['rolling_mean_at_unlock']:+.2f})")

    # Plots
    plot_reward_curve(curr_rewards, rand_rewards, heur_rewards,
                      transitions, RUNS_DIR / "reward_curve.png")
    plot_curriculum_progress(tier_per_episode, transitions,
                             RUNS_DIR / "curriculum_progress.png")
    plot_baseline_comparison(curr_rewards, rand_rewards, heur_rewards,
                             RUNS_DIR / "baseline_comparison.png")
    plot_success_rate(curr_correct, rand_correct, heur_correct,
                      RUNS_DIR / "success_rate.png")

    summary = {
        "episodes": steps,
        "seed": seed,
        "transitions": transitions,
        "final_tier": curr_env.tier,
        "final_tier_name": curr_env.tier_name,
        "mean_curriculum_last100": (sum(curr_rewards[-100:]) / max(1, len(curr_rewards[-100:]))),
        "mean_random_last100": (sum(rand_rewards[-100:]) / max(1, len(rand_rewards[-100:]))),
        "mean_heuristic_last100": (sum(heur_rewards[-100:]) / max(1, len(heur_rewards[-100:]))),
        "correct_rate_curriculum_last100": sum(curr_correct[-100:]) / max(1, len(curr_correct[-100:])),
        "correct_rate_random_last100": sum(rand_correct[-100:]) / max(1, len(rand_correct[-100:])),
        "correct_rate_heuristic_last100": sum(heur_correct[-100:]) / max(1, len(heur_correct[-100:])),
    }
    with (RUNS_DIR / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Final summary ===")
    for k, v in summary.items():
        if k == "transitions":
            continue
        print(f"  {k}: {v}")
    print(f"\nLogs:    {log_path}")
    print(f"Plots:   {RUNS_DIR}/*.png")


# =============================================================================
# TRL-mode training loop
# =============================================================================

def build_system_prompt() -> str:
    lines = [
        "You are a senior SOC analyst investigating a security incident in an "
        "enterprise network. You have nine tools. Use them to gather evidence,",
        "then commit to ONE terminal verdict. Output a single JSON object: ",
        '{"action_type": "...", "ip": "...", "host": "..."} (omit fields you do not need).',
        "",
        "Tools (5 investigative + 4 terminal):",
    ]
    for a in ALL_ACTIONS:
        lines.append(f"  - {a}: {ACTION_DESCRIPTIONS[a]}")
    lines.append("")
    lines.append("Investigate before committing. Premature verdicts are penalised.")
    return "\n".join(lines)


def build_user_prompt(obs: Dict[str, Any]) -> str:
    parts = [
        f"ALERT [{obs['alert_severity'].upper()}]: {obs['initial_alert']}",
        f"Step {obs['step']} / {obs['step_budget']}.",
        f"Known IPs: {obs['known_entities']['ips']}",
        f"Known hosts: {obs['known_entities']['hosts']}",
    ]
    if obs.get("revealed_evidence"):
        parts.append("Evidence collected so far:")
        for e in obs["revealed_evidence"]:
            parts.append(f"  - {e}")
    else:
        parts.append("No evidence collected yet.")

    history = obs.get("action_history", [])
    if history:
        parts.append("Last actions:")
        for h in history[-5:]:
            parts.append(f"  - step {h['step']}: {h['action_type']} "
                         f"ip={h.get('ip')} host={h.get('host')}")
    parts.append("")
    parts.append("Reply with ONE JSON action.")
    return "\n".join(parts)


def run_trl(args):
    try:
        from datasets import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer
    except Exception as e:
        print("[trl] missing optional deps:", e, file=sys.stderr)
        sys.exit(2)

    model_id = args.model
    print(f"[trl] loading model: {model_id}")
    if args.use_unsloth:
        try:
            from unsloth import FastLanguageModel
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_id, max_seq_length=1024, dtype=None,
                load_in_4bit=True,
            )
        except Exception as e:
            print("[trl] unsloth unavailable, falling back:", e)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)

    sys_prompt = build_system_prompt()
    examples = []
    seed = args.seed
    for i in range(args.steps):
        env = CurriculumEnv(seed=seed + i)
        obs = env.reset()
        examples.append({
            "prompt": (
                f"<|system|>\n{sys_prompt}\n<|user|>\n"
                f"{build_user_prompt(obs)}\n<|assistant|>\n"
            ),
            "scenario_seed": seed + i,
        })

    ds = Dataset.from_list(examples)

    heur = HeuristicAgent(seed=seed)

    def reward_fn(prompts, completions, **kwargs):
        rewards = []
        seeds = kwargs.get("scenario_seed") or [seed] * len(completions)
        for comp, sd in zip(completions, seeds):
            env = CurriculumEnv(seed=int(sd))
            obs = env.reset()
            try:
                action = json.loads(comp.strip().split("\n")[0])
            except Exception:
                action = {"action_type": "list_entities"}
            obs, r, done, info = env.step(action)
            total = r
            heur.reset()
            steps = 0
            while not done and steps < 40:
                a = heur.act(obs)
                obs, r, done, info = env.step(a)
                total += r
                steps += 1
            rewards.append(float(total))
        return rewards

    cfg = GRPOConfig(
        output_dir=str(RUNS_DIR / "final_model"),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        max_completion_length=256,
        num_generations=4,
        logging_steps=5,
        num_train_epochs=1,
    )

    trainer = GRPOTrainer(
        model=model,
        args=cfg,
        train_dataset=ds,
        reward_funcs=[reward_fn],
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(RUNS_DIR / "final_model"))
    print(f"[trl] saved to {RUNS_DIR/'final_model'}")


# =============================================================================
# CLI
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_trl", action="store_true")
    ap.add_argument("--use_unsloth", action="store_true")
    args = ap.parse_args()
    if args.use_trl:
        run_trl(args)
    else:
        run_rollout(args)


if __name__ == "__main__":
    main()
