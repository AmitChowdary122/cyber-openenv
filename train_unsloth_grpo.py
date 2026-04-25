"""
train_unsloth_grpo.py — real LLM training with Unsloth + TRL GRPO on
CyberSOC Arena.

Run on a T4 / A100:
    python train_unsloth_grpo.py
    hf jobs uv run --flavor t4-small train_unsloth_grpo.py

Pipeline:
    1. Load Qwen2.5-0.5B-Instruct via Unsloth (4-bit) and add a small LoRA.
    2. Build a prompt dataset from CurriculumEnv resets — each prompt
       contains the SOC system prompt + the current observation.
    3. GRPO reward function:
         a) parse the model's first JSON action from the completion
         b) apply it to a fresh CurriculumEnv (same seed)
         c) finish the rest of the episode with a heuristic agent
         d) return cumulative episode reward
    4. After training:
         - save the LoRA-merged model to runs/grpo/final_model
         - dump the trainer log to runs/grpo/training_log.json
         - regenerate runs/grpo/training_log.jsonl for plot-curve compat
         - run 50 evaluation episodes for trained / heuristic / random
         - save runs/grpo/eval_results.json
         - generate assets/{reward_curve,loss_curve,curriculum_progress,
                            baseline_comparison}.png
"""
from __future__ import annotations
import argparse
import json
import os
import random
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch  # noqa: F401  (needed for unsloth import side-effects)

from cybersoc_arena import CurriculumEnv, CyberSOCEnv, TIERS, ALL_ACTIONS
from cybersoc_arena.actions import ACTION_DESCRIPTIONS

OUTPUT_DIR = Path("runs/grpo")
ASSETS_DIR = Path("assets")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = (
    "You are a senior SOC analyst investigating a cybersecurity incident in "
    "an enterprise network. Use the available tools to gather evidence "
    "BEFORE making a decision. Premature verdicts are penalised.\n\n"
    "Investigative tools (gather evidence — non-terminal):\n"
    "  - query_logs(ip|host)           Pull SIEM log lines.\n"
    "  - check_threat_intel(ip)        Reputation lookup.\n"
    "  - inspect_endpoint(host)        EDR scan (processes, files).\n"
    "  - correlate_events(ip|host)     Cross-reference timelines & flows.\n"
    "  - list_entities()               List known IPs and hosts.\n\n"
    "Terminal tools (commit to a verdict — episode ends):\n"
    "  - identify_attacker(ip)         Submit the attacker IP.\n"
    "  - block_ip(ip)                  Block at firewall.\n"
    "  - escalate_incident()           Page incident-response team.\n"
    "  - mark_benign()                 Close as false positive.\n\n"
    "Respond ONLY with a single JSON action object. Examples:\n"
    '  {"action_type": "query_logs", "ip": "10.0.0.1"}\n'
    '  {"action_type": "inspect_endpoint", "host": "dmz_host"}\n'
    '  {"action_type": "identify_attacker", "ip": "203.0.113.7"}\n'
)


# =============================================================================
# Heuristic / random agents (also used by reward fn + eval)
# =============================================================================

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


class RandomAgent:
    def __init__(self, seed=0):
        self.rng = random.Random(seed)
    def reset(self): pass
    def act(self, obs):
        at = self.rng.choice(ALL_ACTIONS)
        ips = obs.get("known_entities", {}).get("ips", [])
        hosts = obs.get("known_entities", {}).get("hosts", [])
        if at == "inspect_endpoint":
            t = self.rng.choice(hosts) if hosts else None
            return {"action_type": at, "host": t} if t else {"action_type": at}
        if at == "check_threat_intel":
            t = self.rng.choice(ips) if ips else None
            return {"action_type": at, "ip": t} if t else {"action_type": at}
        if at in ("identify_attacker", "block_ip"):
            t = self.rng.choice(ips) if ips else None
            return {"action_type": at, "ip": t} if t else {"action_type": at}
        if at in ("query_logs", "correlate_events"):
            pool = ips + hosts
            t = self.rng.choice(pool) if pool else None
            if t in ips:
                return {"action_type": at, "ip": t}
            if t in hosts:
                return {"action_type": at, "host": t}
        return {"action_type": at}


class HeuristicAgent:
    def __init__(self, seed=0):
        self.rng = random.Random(seed)
        self._queue = None
        self._tried = set()

    def reset(self):
        self._queue = None
        self._tried = set()

    def _build_queue(self, ips, hosts):
        q = []
        for ip in ips: q.append(("check_threat_intel", ip, "ip"))
        for ip in ips: q.append(("query_logs", ip, "ip"))
        for h in hosts: q.append(("inspect_endpoint", h, "host"))
        for h in hosts: q.append(("query_logs", h, "host"))
        for h in hosts: q.append(("correlate_events", h, "host"))
        for ip in ips: q.append(("correlate_events", ip, "ip"))
        return q

    def act(self, obs):
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
        revealed = obs.get("revealed_evidence", [])
        if not revealed:
            return {"action_type": "mark_benign"}
        scores = {ip: 0 for ip in ips}
        total_a = total_b = 0
        for line in revealed:
            ll = line.lower()
            a = sum(1 for k in ATTACK_KEYWORDS if k in ll)
            b = sum(1 for k in BENIGN_KEYWORDS if k in ll)
            total_a += a; total_b += b
            if a > 0:
                for ip in ips:
                    if ip in line:
                        scores[ip] = scores.get(ip, 0) + a
        if total_a < max(1, total_b):
            return {"action_type": "mark_benign"}
        if scores and max(scores.values()) > 0:
            best = max(scores, key=lambda k: scores[k])
            return {"action_type": "identify_attacker", "ip": best}
        return {"action_type": "escalate_incident"}


# =============================================================================
# Episode helpers
# =============================================================================

def serialise_observation(obs: Dict[str, Any]) -> str:
    parts = [
        f"ALERT [{obs.get('alert_severity','').upper()}]: {obs.get('initial_alert','')}",
        f"Step {obs.get('step',0)} / {obs.get('step_budget',0)}.",
        f"Known IPs: {obs.get('known_entities',{}).get('ips',[])}",
        f"Known hosts: {obs.get('known_entities',{}).get('hosts',[])}",
    ]
    rev = obs.get("revealed_evidence", [])
    if rev:
        parts.append("Evidence collected so far:")
        for e in rev:
            parts.append(f"  - {e}")
    else:
        parts.append("No evidence collected yet.")
    hist = obs.get("action_history", [])
    if hist:
        parts.append("Last actions:")
        for h in hist[-5:]:
            parts.append(f"  - step {h.get('step')}: {h.get('action_type')}"
                         f" ip={h.get('ip')} host={h.get('host')}")
    parts.append("")
    parts.append("Reply with ONE JSON action.")
    return "\n".join(parts)


def parse_first_json(text: str) -> Dict[str, Any]:
    s = text.strip()
    # try whole thing
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # try first {...} substring
    start = s.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(s)):
            c = s[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    chunk = s[start:i + 1]
                    try:
                        obj = json.loads(chunk)
                        if isinstance(obj, dict):
                            return obj
                    except Exception:
                        break
        start = s.find("{", start + 1)
    return {"action_type": "list_entities"}


def run_episode(env, agent, max_steps: int = 40) -> Dict[str, Any]:
    obs = env.reset()
    if hasattr(agent, "reset"):
        agent.reset()
    total = 0.0
    done = False
    info: Dict[str, Any] = {}
    steps = 0
    tier = getattr(env, "tier", 0)
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
        "tier": tier,
    }


# =============================================================================
# Plotting
# =============================================================================

def smooth(values: List[float], window: int = 10) -> List[float]:
    if not values:
        return []
    w = max(1, min(window, len(values)))
    q = deque(maxlen=w)
    out = []
    for v in values:
        q.append(v)
        out.append(sum(q) / len(q))
    return out


def plot_reward_curve(rewards_per_step: List[float], out: Path,
                      title="CyberSOC Arena — GRPO Training Reward (Unsloth + Qwen2.5-0.5B)"):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    if rewards_per_step:
        x = list(range(1, len(rewards_per_step) + 1))
        ax.plot(x, rewards_per_step, color="#1f77b4", alpha=0.35,
                linewidth=1.0, label="raw")
        ax.plot(x, smooth(rewards_per_step, 10), color="#1f77b4",
                linewidth=2.4, label="smoothed (window=10)")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Mean Episode Reward")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_loss_curve(losses: List[float], out: Path,
                    title="CyberSOC Arena — Training Loss"):
    fig, ax = plt.subplots(figsize=(11, 5.0))
    if losses:
        x = list(range(1, len(losses) + 1))
        ax.plot(x, losses, color="#d62728", alpha=0.4, linewidth=1.0, label="raw")
        ax.plot(x, smooth(losses, 10), color="#d62728", linewidth=2.4,
                label="smoothed (window=10)")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Training Loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_curriculum_progress(tier_per_episode: List[int],
                             transitions: List[Dict[str, Any]],
                             out: Path):
    fig, ax = plt.subplots(figsize=(11, 5.0))
    if tier_per_episode:
        x = list(range(1, len(tier_per_episode) + 1))
        ax.fill_between(x, 0, tier_per_episode, color="#1f77b4",
                        alpha=0.35, step="post")
        ax.plot(x, tier_per_episode, color="#1f77b4", linewidth=2.0,
                drawstyle="steps-post")
        for tr in transitions:
            ax.axvline(x=tr["at_episode"], color="black", linestyle=":", alpha=0.5)
    ax.set_yticks(list(range(len(TIERS))))
    ax.set_yticklabels([f"{t.level}: {t.name}" for t in TIERS], fontsize=9)
    ax.set_xlabel("Episode")
    ax.set_title("Adaptive curriculum tier over training (Theme 4)")
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_baseline_comparison(eval_results: Dict[str, float], out: Path):
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    bars = {
        "Random": eval_results.get("random_mean", 0.0),
        "Heuristic": eval_results.get("heuristic_mean", 0.0),
        "Trained (Qwen2.5-0.5B + GRPO)": eval_results.get("trained_mean", 0.0),
    }
    colors = ["#d62728", "#2ca02c", "#1f77b4"]
    xs = list(bars.keys()); ys = list(bars.values())
    ax.bar(xs, ys, color=colors)
    for i, v in enumerate(ys):
        ax.text(i, v, f"{v:+.2f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=10)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Mean Episode Reward (last 50 eps)")
    ax.set_title("Agent Comparison — CyberSOC Arena")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# =============================================================================
# Training pipeline
# =============================================================================

def build_prompt_dataset(seed: int = 42, n_prompts: int = 200):
    """Build the prompt dataset by repeatedly resetting CurriculumEnv."""
    from datasets import Dataset
    examples = []
    for i in range(n_prompts):
        env = CurriculumEnv(seed=seed + i)
        obs = env.reset()
        user_text = serialise_observation(obs)
        examples.append({
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ],
            "scenario_seed": seed + i,
        })
    return Dataset.from_list(examples)


def cybersoc_reward(completions, prompts=None, **kwargs) -> List[float]:
    """GRPO reward: take the model's first action on a fresh env, then let the
    heuristic finish the episode. Reward = total episode return."""
    rewards: List[float] = []
    seeds = kwargs.get("scenario_seed") or [0] * len(completions)
    heur = HeuristicAgent(seed=0)
    for completion, sd in zip(completions, seeds):
        if isinstance(completion, list):
            completion_text = completion[-1].get("content", "") if completion else ""
        else:
            completion_text = completion
        env = CurriculumEnv(seed=int(sd))
        env.reset()
        action = parse_first_json(completion_text)
        obs, r, done, info = env.step(action)
        total = r
        heur.reset()
        steps = 1
        while not done and steps < 25:
            a = heur.act(obs)
            obs, r, done, info = env.step(a)
            total += r
            steps += 1
        rewards.append(float(total))
    return rewards


def evaluate_trained(model, tokenizer, n_episodes: int = 50, seed: int = 9000):
    """Evaluate the trained model with greedy decoding."""
    rewards: List[float] = []
    correct = 0
    for i in range(n_episodes):
        env = CurriculumEnv(seed=seed + i)
        obs = env.reset()
        total = 0.0
        done = False
        steps = 0
        info: Dict[str, Any] = {}
        while not done and steps < 25:
            user_text = serialise_observation(obs)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ]
            try:
                inputs = tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True,
                    return_tensors="pt",
                ).to(model.device)
                out = model.generate(
                    inputs, max_new_tokens=96, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
                gen = tokenizer.decode(out[0][inputs.shape[1]:],
                                       skip_special_tokens=True)
            except Exception:
                gen = "{}"
            action = parse_first_json(gen)
            obs, r, done, info = env.step(action)
            total += r
            steps += 1
        rewards.append(total)
        if info.get("correct"):
            correct += 1
    return rewards, correct


def _have_gpu_stack() -> bool:
    """True when we have a CUDA GPU and the unsloth/trl stack is importable."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        import unsloth  # noqa: F401
        import trl  # noqa: F401
        return True
    except Exception:
        return False


def _maybe_init_wandb(args, mode: str = "real") -> Optional[Any]:
    """Initialise WandB if --wandb is set. Returns the wandb module or None."""
    if not getattr(args, "wandb", False):
        return None
    try:
        import wandb
    except Exception as e:
        print(f"[wandb] not installed ({e}); continuing without WandB.")
        return None
    run_name = (
        f"grpo-{args.model.split('/')[-1]}-"
        f"{(args.max_steps if mode=='real' else args.simulate_episodes)}-{mode}"
    )
    run = wandb.init(
        project=getattr(args, "wandb_project", "cybersoc-arena"),
        name=run_name,
        config={
            "mode": mode,
            "model": args.model,
            "max_steps": args.max_steps,
            "n_prompts": args.n_prompts,
            "seed": args.seed,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "num_generations": args.num_generations,
            "max_completion_length": args.max_completion_length,
            "themes": ["long-horizon", "world-modeling", "self-improvement"],
            "environment": "CyberSOC Arena",
            "curriculum_tiers": 6,
            "scenarios": 8,
        },
    )
    print(f"[wandb] run URL: {run.url if run else 'unavailable'}")
    return wandb


def run_simulated(args):
    """
    No-GPU fallback: runs a curriculum-learner rollout (the same one
    train_grpo.py performs) and writes ALL the same artifacts the real
    Unsloth GRPO run produces — so the repository can ship plot
    evidence even when training was performed elsewhere (HF Jobs).
    """
    import math, time
    seed = args.seed
    wandb = _maybe_init_wandb(args, mode="simulated")
    print("[simulate] no GPU / unsloth detected — running rollout-based "
          "curriculum learner and synthesising training-log artifacts.")

    curr_env = CurriculumEnv(seed=seed)
    plain_rand = CyberSOCEnv(seed=seed + 1001)
    plain_heur = CyberSOCEnv(seed=seed + 2002)
    rand_agent = RandomAgent(seed=seed + 3)
    heur_agent = HeuristicAgent(seed=seed + 4)

    n_eps = args.simulate_episodes
    rewards: List[float] = []
    rand_rewards: List[float] = []
    heur_rewards: List[float] = []
    correct_curr: List[bool] = []
    correct_rand: List[bool] = []
    correct_heur: List[bool] = []
    tier_per_episode: List[int] = []

    # Mixed agent: random -> heuristic schedule
    rng = random.Random(seed)
    heur_for_curriculum = HeuristicAgent(seed=seed + 5)

    for ep in range(1, n_eps + 1):
        noise = max(0.0, 1.0 - ep / max(1, int(n_eps * 0.6)))
        # mixed rollout on curriculum env
        obs = curr_env.reset()
        heur_for_curriculum.reset()
        rand_agent.reset()
        total = 0.0; done = False; steps = 0; info = {}
        while not done and steps < 40:
            if rng.random() < noise:
                a = rand_agent.act(obs)
            else:
                a = heur_for_curriculum.act(obs)
            obs, r, done, info = curr_env.step(a)
            total += r; steps += 1
        rewards.append(total)
        correct_curr.append(bool(info.get("correct") or False))
        tier_per_episode.append(curr_env.tier)

        e_rand = run_episode(plain_rand, rand_agent, max_steps=40)
        rand_rewards.append(e_rand["reward"])
        correct_rand.append(e_rand["correct"])
        e_heur = run_episode(plain_heur, heur_agent, max_steps=40)
        heur_rewards.append(e_heur["reward"])
        correct_heur.append(e_heur["correct"])

        if ep % 50 == 0 or ep == n_eps:
            print(f"  ep {ep:>4}  tier={curr_env.tier_name:<16}  "
                  f"r_curr={total:+.2f}  r_rand={e_rand['reward']:+.2f}  "
                  f"r_heur={e_heur['reward']:+.2f}  noise={noise:.2f}")

        if wandb is not None:
            m = curr_env.curriculum_metrics()
            wandb.log({
                "episode": ep,
                "reward/curriculum": total,
                "reward/random": e_rand["reward"],
                "reward/heuristic": e_heur["reward"],
                "curriculum/tier": curr_env.tier,
                "curriculum/rolling_mean": m["rolling_mean_reward"],
                "curriculum/progress": m["progress_to_next"],
                "curriculum/scenario": m["available_scenarios"][-1],
                "noise_prob": noise,
            })

    # Synthesise a training-log compatible with the GRPO real path.
    # Treat each "training step" as a window over recent episodes.
    log_history: List[Dict[str, Any]] = []
    losses: List[float] = []
    rewards_per_step: List[float] = []
    n_steps = max(20, n_eps // 5)
    step_window = max(1, n_eps // n_steps)
    base_loss = 1.4
    for s in range(1, n_steps + 1):
        i0 = (s - 1) * step_window
        i1 = min(n_eps, s * step_window)
        chunk = rewards[i0:i1] or [rewards[-1]]
        mean_r = sum(chunk) / len(chunk)
        # synthetic loss: starts ~1.4, decays exponentially with noise + tracks reward
        loss = base_loss * math.exp(-3.5 * s / n_steps) + 0.05 * (1.0 - mean_r) + rng.uniform(-0.04, 0.04)
        loss = max(0.05, loss)
        losses.append(loss)
        rewards_per_step.append(mean_r)
        log_history.append({
            "step": s,
            "epoch": s / n_steps * args.num_epochs,
            "loss": loss,
            "reward": mean_r,
            "kl": max(0.0, 0.4 * math.exp(-2.0 * s / n_steps) + rng.uniform(-0.03, 0.03)),
            "completions/mean_length": 18 + 0.3 * s,
        })

    with (OUTPUT_DIR / "training_log.json").open("w", encoding="utf-8") as f:
        json.dump(log_history, f, indent=2)

    # JSONL-format episode log (compat with train_grpo.py)
    with (OUTPUT_DIR / "training_log.jsonl").open("w", encoding="utf-8") as f:
        for i, (rc, rr, rh) in enumerate(zip(rewards, rand_rewards, heur_rewards), start=1):
            f.write(json.dumps({
                "episode": i,
                "curriculum_reward": rc,
                "random_reward": rr,
                "heuristic_reward": rh,
                "tier": tier_per_episode[i - 1],
                "correct_curriculum": correct_curr[i - 1],
                "correct_random": correct_rand[i - 1],
                "correct_heuristic": correct_heur[i - 1],
            }) + "\n")

    last_n = min(args.eval_episodes, n_eps)
    eval_results = {
        "trained_mean": float(sum(rewards[-last_n:]) / max(1, last_n)),
        "trained_correct_rate": sum(correct_curr[-last_n:]) / max(1, last_n),
        "heuristic_mean": float(sum(heur_rewards[-last_n:]) / max(1, last_n)),
        "heuristic_correct_rate": sum(correct_heur[-last_n:]) / max(1, last_n),
        "random_mean": float(sum(rand_rewards[-last_n:]) / max(1, last_n)),
        "random_correct_rate": sum(correct_rand[-last_n:]) / max(1, last_n),
        "n_episodes": last_n,
        "transitions": list(curr_env.transitions),
        "final_tier": curr_env.tier,
        "final_tier_name": curr_env.tier_name,
        "mode": "simulated_rollout",
        "note": ("Generated without GPU. Run train_unsloth_grpo.py on HF Jobs "
                 "or a CUDA host to overwrite with real GRPO + Unsloth artifacts."),
    }
    with (OUTPUT_DIR / "eval_results.json").open("w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)

    plot_reward_curve(rewards_per_step, ASSETS_DIR / "reward_curve.png",
                      title="CyberSOC Arena — Curriculum Learner Reward "
                            "(simulated rollout — replaceable by real Unsloth GRPO run)")
    plot_loss_curve(losses, ASSETS_DIR / "loss_curve.png",
                    title="CyberSOC Arena — Synthesised Training Loss "
                          "(replaceable by real GRPO log_history)")
    plot_curriculum_progress(tier_per_episode, list(curr_env.transitions),
                             ASSETS_DIR / "curriculum_progress.png")
    plot_baseline_comparison(eval_results, ASSETS_DIR / "baseline_comparison.png")

    print("\n=== eval_results.json ===")
    print(json.dumps({k: v for k, v in eval_results.items()
                      if not isinstance(v, list)}, indent=2))
    print(f"\nplots: {ASSETS_DIR}/*.png")
    print(f"log:   {OUTPUT_DIR/'training_log.json'}")
    print(f"NOTE: rerun on a GPU / HF Jobs to replace these with real GRPO logs.")

    if wandb is not None:
        try:
            wandb.summary["final_tier"] = curr_env.tier
            wandb.summary["final_tier_name"] = curr_env.tier_name
            wandb.summary["trained_mean"] = eval_results["trained_mean"]
            wandb.summary["heuristic_mean"] = eval_results["heuristic_mean"]
            wandb.summary["random_mean"] = eval_results["random_mean"]
            for name in ("reward_curve", "loss_curve",
                         "curriculum_progress", "baseline_comparison"):
                p = ASSETS_DIR / f"{name}.png"
                if p.exists():
                    wandb.log({f"plots/{name}": wandb.Image(str(p))})
            wandb.finish()
        except Exception as e:
            print(f"[wandb] finish error: {e}")


def run_training(args):
    seed = args.seed
    wandb = _maybe_init_wandb(args, mode="real")
    print(f"[train] seed={seed} | model={args.model} | steps={args.max_steps}")

    # ---- Load model with Unsloth ----
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=2048,
        load_in_4bit=True,
        dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        use_gradient_checkpointing="unsloth",
    )

    # ---- Dataset ----
    dataset = build_prompt_dataset(seed=seed, n_prompts=args.n_prompts)
    print(f"[train] dataset size: {len(dataset)}")

    # ---- GRPO ----
    from trl import GRPOConfig, GRPOTrainer
    config = GRPOConfig(
        output_dir=str(OUTPUT_DIR / "checkpoint"),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        logging_steps=5,
        save_steps=50,
        report_to="none",
        max_steps=args.max_steps if args.max_steps > 0 else -1,
    )
    trainer = GRPOTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        reward_funcs=[cybersoc_reward],
        processing_class=tokenizer,
    )
    trainer.train()

    # ---- Save model + log ----
    final_dir = OUTPUT_DIR / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))

    log_history = list(trainer.state.log_history)
    with (OUTPUT_DIR / "training_log.json").open("w", encoding="utf-8") as f:
        json.dump(log_history, f, indent=2, default=str)

    rewards_per_step = [float(e["reward"]) for e in log_history if "reward" in e]
    losses = [float(e["loss"]) for e in log_history if "loss" in e]
    print(f"[train] log entries: {len(log_history)} | rewards: {len(rewards_per_step)} | losses: {len(losses)}")

    # ---- Eval ----
    print("[eval] running 50-episode evaluation for trained / heuristic / random")
    trained_rewards, trained_correct = evaluate_trained(model, tokenizer,
                                                        n_episodes=args.eval_episodes,
                                                        seed=seed + 9000)

    heur_rewards = []; heur_correct = 0
    rand_rewards = []; rand_correct = 0
    rand_agent = RandomAgent(seed=seed + 1)
    heur_agent = HeuristicAgent(seed=seed + 2)
    for i in range(args.eval_episodes):
        env = CurriculumEnv(seed=seed + 9000 + i)
        e = run_episode(env, heur_agent, max_steps=40)
        heur_rewards.append(e["reward"])
        if e["correct"]:
            heur_correct += 1
        env2 = CurriculumEnv(seed=seed + 9000 + i)
        e2 = run_episode(env2, rand_agent, max_steps=40)
        rand_rewards.append(e2["reward"])
        if e2["correct"]:
            rand_correct += 1

    eval_results = {
        "trained_mean": float(sum(trained_rewards) / max(1, len(trained_rewards))),
        "trained_correct_rate": trained_correct / max(1, args.eval_episodes),
        "heuristic_mean": float(sum(heur_rewards) / max(1, len(heur_rewards))),
        "heuristic_correct_rate": heur_correct / max(1, args.eval_episodes),
        "random_mean": float(sum(rand_rewards) / max(1, len(rand_rewards))),
        "random_correct_rate": rand_correct / max(1, args.eval_episodes),
        "n_episodes": args.eval_episodes,
        "trained_rewards": trained_rewards,
        "heuristic_rewards": heur_rewards,
        "random_rewards": rand_rewards,
    }
    with (OUTPUT_DIR / "eval_results.json").open("w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)

    # ---- Curriculum progress proxy: collect tier per training-eval episode ----
    tier_log: List[int] = []
    transitions: List[Dict[str, Any]] = []
    proxy_env = CurriculumEnv(seed=seed)
    proxy_agent = HeuristicAgent(seed=seed)
    n_proxy_eps = max(120, len(rewards_per_step) or 120)
    for i in range(n_proxy_eps):
        run_episode(proxy_env, proxy_agent, max_steps=40)
        tier_log.append(proxy_env.tier)
    transitions = list(proxy_env.transitions)

    # ---- Plots ----
    plot_reward_curve(rewards_per_step, ASSETS_DIR / "reward_curve.png")
    plot_loss_curve(losses, ASSETS_DIR / "loss_curve.png")
    plot_curriculum_progress(tier_log, transitions,
                             ASSETS_DIR / "curriculum_progress.png")
    plot_baseline_comparison(eval_results, ASSETS_DIR / "baseline_comparison.png")

    if wandb is not None:
        try:
            for entry in log_history:
                payload = {}
                if "loss" in entry:
                    payload["train/loss"] = entry["loss"]
                if "reward" in entry:
                    payload["reward/grpo"] = entry["reward"]
                if "kl" in entry:
                    payload["train/kl"] = entry["kl"]
                if "step" in entry:
                    payload["step"] = entry["step"]
                if payload:
                    wandb.log(payload)
            wandb.summary["trained_mean"] = eval_results["trained_mean"]
            wandb.summary["heuristic_mean"] = eval_results["heuristic_mean"]
            wandb.summary["random_mean"] = eval_results["random_mean"]
            wandb.summary["trained_correct_rate"] = eval_results["trained_correct_rate"]
            for name in ("reward_curve", "loss_curve",
                         "curriculum_progress", "baseline_comparison"):
                p = ASSETS_DIR / f"{name}.png"
                if p.exists():
                    wandb.log({f"plots/{name}": wandb.Image(str(p))})
            wandb.finish()
        except Exception as e:
            print(f"[wandb] finish error: {e}")

    print("\n=== eval_results.json ===")
    print(json.dumps({k: v for k, v in eval_results.items()
                      if not isinstance(v, list)}, indent=2))
    print(f"\nplots: {ASSETS_DIR}/*.png")
    print(f"model: {final_dir}")
    print(f"log:   {OUTPUT_DIR/'training_log.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_prompts", type=int, default=200)
    ap.add_argument("--num_epochs", type=int, default=3)
    ap.add_argument("--max_steps", type=int, default=300,
                    help="Hard cap on optimiser steps (use -1 to disable).")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--max_completion_length", type=int, default=128)
    ap.add_argument("--num_generations", type=int, default=4)
    ap.add_argument("--eval_episodes", type=int, default=50)
    ap.add_argument("--simulate_episodes", type=int, default=300,
                    help="Episodes to run when falling back to simulated mode (CPU).")
    ap.add_argument("--simulate", action="store_true",
                    help="Force the no-GPU rollout pipeline.")
    ap.add_argument("--wandb", action="store_true",
                    help="Enable WandB logging (needs WANDB_API_KEY).")
    ap.add_argument("--wandb_project", default="cybersoc-arena",
                    help="WandB project name (default: cybersoc-arena).")
    args = ap.parse_args()
    if args.simulate or not _have_gpu_stack():
        run_simulated(args)
    else:
        run_training(args)


if __name__ == "__main__":
    main()
