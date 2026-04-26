"""Real REINFORCE training on CyberSOC Arena (CPU, numpy-only).

Why this exists
---------------
The hackathon judging guide is explicit: "Evidence that you actually trained;
at minimum, loss and reward plots from a real run." This script runs a real
on-policy REINFORCE loop against the live CurriculumEnv, with a tiny
softmax policy parameterised in numpy. No GPU, torch, or Unsloth needed,
so the curves are reproducible from a clean checkout in under a minute.

Design
------
Action space is collapsed to a small *meta* policy of 4 strategic moves:
    0  INVESTIGATE   - cycle through investigative tools on the next entity
    1  CORRELATE     - run correlate_events on the most-mentioned entity
    2  IDENTIFY      - identify_attacker on the IP most-referenced across
                       attacker-style findings
    3  CLOSE_BENIGN  - close_as_benign (right answer for benign scenarios)
The heuristic target-picker handles "WHICH IP/host" while the policy learns
"WHEN to invest in more evidence vs WHEN to commit, and which way to commit."
This keeps the learning problem small enough to solve on CPU, while still
producing measurable improvement over a random meta-policy.

The HF Jobs script (scripts/run_hf_job_a100.sh -> scripts/train_hf_job.py)
provides the second, larger-scale training run with TRL's GRPOTrainer +
Qwen2.5-1.5B-Instruct + LoRA on an L40S 48GB. Both train on the same env,
so the reward curves are directly comparable.

Output
------
training_runs/reinforce/
    training_log.json     - per-episode reward, loss, evidence, tier
    eval_results.json     - mean reward & success rate (random vs trained)
    policy.npz            - trained policy parameters
assets/
    reward_curve.png      - smoothed reward vs episode (random + trained)
    loss_curve.png        - policy gradient loss vs training step
    curriculum_progress.png - curriculum tier over episodes
    baseline_comparison.png - bar chart of mean rewards per scenario
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import sys
from collections import Counter, deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cybersoc_arena import (  # noqa: E402
    CurriculumEnv,
    CyberAction,
    CyberObservation,
    CyberSOCEnv,
    SCENARIO_TYPES,
)


# ─────────────────────────────────────────────────────────────────────────────
# Meta-action space (4 high-level moves)
# ─────────────────────────────────────────────────────────────────────────────
META_ACTIONS = ["INVESTIGATE", "CORRELATE", "IDENTIFY", "CLOSE_BENIGN"]
META_DIM = len(META_ACTIONS)

# Cycle of investigative tools tried in INVESTIGATE
INVESTIGATIVE_CYCLE = [
    "query_logs",
    "inspect_endpoint",
    "check_threat_intel",
    "investigate_ip",
]

SEVERITY_MAP = {"info": 0.0, "low": 0.25, "medium": 0.5, "high": 0.75, "critical": 1.0}

# Substrings that suggest an attacker-style finding vs a decoy
ATTACKER_KEYWORDS = (
    "exploit", "exfil", "exfiltrat", "phish", "ransom", "kerberos",
    "lateral", "stager", "rundll32", "powershell", "webshell", "persistence",
    "outbound", "self-signed", "ja3", "credentials stuffed", "credential stuffing",
    "1234567",
)
DECOY_KEYWORDS = (
    "scanner", "shodan", "censys", "authorized red-team", "red-team", "red team",
    "backup", "expected", "no follow-on", "dhcp", "ldap",
)
IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


# ─────────────────────────────────────────────────────────────────────────────
# Evidence parsing helpers
# ─────────────────────────────────────────────────────────────────────────────
def _evidence_ip_mentions(evidence_findings: List[str]) -> Counter:
    """Count how often each IP appears in attacker-flavored findings."""
    c: Counter = Counter()
    for f in evidence_findings:
        text = f.lower()
        attacker_score = sum(1 for kw in ATTACKER_KEYWORDS if kw in text)
        decoy_score = sum(1 for kw in DECOY_KEYWORDS if kw in text)
        weight = max(0, attacker_score - decoy_score)
        if weight <= 0:
            continue
        for ip in IPV4_RE.findall(f):
            c[ip] += weight
    return c


def _entity_mentions(evidence_findings: List[str]) -> Counter:
    """Count distinct entity (ip+host-token) mentions for CORRELATE picker."""
    c: Counter = Counter()
    tokens_re = re.compile(r"\b((?:\d{1,3}\.){3}\d{1,3}|(?:edge|ws|fs|db|proxy|host|api)-\d{2,4})\b",
                           re.IGNORECASE)
    for f in evidence_findings:
        for tok in tokens_re.findall(f):
            c[tok] += 1
    return c


# ─────────────────────────────────────────────────────────────────────────────
# Feature extractor
# ─────────────────────────────────────────────────────────────────────────────
def featurize(obs: CyberObservation) -> np.ndarray:
    sev = SEVERITY_MAP.get(obs.alert.severity.lower(), 0.5)
    progress = obs.step / max(1, obs.step_budget)
    remaining = obs.remaining_steps / max(1, obs.step_budget)
    n_visible_ips = min(20, len(obs.asset_inventory.visible_ips)) / 20.0
    n_hosts = min(10, len(obs.asset_inventory.hosts)) / 10.0
    n_evidence = min(10, obs.evidence_count) / 10.0

    findings = [e.finding for e in obs.evidence_collected]
    ip_mentions = _evidence_ip_mentions(findings)

    # Strongest signal we have for the policy: max count of attacker-flavored
    # mentions of any single IP. Crosses thresholds when investigation has
    # actually landed on the real attacker.
    top_attacker_mentions = max(ip_mentions.values()) if ip_mentions else 0
    second_attacker_mentions = sorted(ip_mentions.values(), reverse=True)[1] \
        if len(ip_mentions) >= 2 else 0
    n_attacker_evidence = sum(1 for v in ip_mentions.values() if v > 0)

    # Whether the alert text mentions known benign keywords (helps benign close)
    alert_text = obs.alert.summary.lower()
    benign_alert = float(any(kw in alert_text for kw in ("scan", "scanner", "internet")))

    # How many distinct (action_type, target) pairs have been used
    used = set()
    for h in obs.action_history:
        used.add((h.action_type, h.target.lower()))
    diversity = len(used) / max(1, len(obs.action_history) or 1)
    n_actions_taken = min(20, len(obs.action_history)) / 20.0

    return np.array([
        sev,
        progress,
        remaining,
        n_visible_ips,
        n_hosts,
        n_evidence,
        diversity,
        n_actions_taken,
        min(5, top_attacker_mentions) / 5.0,
        min(5, second_attacker_mentions) / 5.0,
        min(8, n_attacker_evidence) / 8.0,
        benign_alert,
        1.0,  # bias
    ], dtype=np.float32)


FEATURE_DIM = 13


# ─────────────────────────────────────────────────────────────────────────────
# Target picker for each meta-action
# ─────────────────────────────────────────────────────────────────────────────
def _least_used(items: List[str], history_targets: List[str],
                rng: random.Random) -> str:
    if not items:
        return ""
    counts = Counter(history_targets)
    sorted_items = sorted(items, key=lambda x: (counts.get(x, 0), rng.random()))
    return sorted_items[0]


def meta_to_action(meta: int, obs: CyberObservation,
                   rng: random.Random,
                   investigate_cycle_idx: int) -> Tuple[CyberAction, int]:
    """Translate a meta-action to a concrete CyberAction.

    Returns (action, next_investigate_cycle_idx).
    """
    visible_ips = list(obs.asset_inventory.visible_ips)
    hosts = list(obs.asset_inventory.hosts)
    history_ip_targets = [
        h.target for h in obs.action_history
        if h.action_type in ("query_logs", "investigate_ip", "check_threat_intel",
                              "identify_attacker")
    ]
    history_host_targets = [
        h.target for h in obs.action_history
        if h.action_type in ("inspect_endpoint", "isolate_host")
    ]

    if meta == 0:  # INVESTIGATE
        tool = INVESTIGATIVE_CYCLE[investigate_cycle_idx % len(INVESTIGATIVE_CYCLE)]
        nxt = (investigate_cycle_idx + 1) % len(INVESTIGATIVE_CYCLE)
        if tool == "inspect_endpoint":
            target = _least_used(hosts, history_host_targets, rng)
            return CyberAction(action_type="inspect_endpoint", host=target), nxt
        # IP-targeted investigative tool
        target = _least_used(visible_ips, history_ip_targets, rng)
        kw = "ip"
        return CyberAction(action_type=tool, **{kw: target}), nxt

    findings = [e.finding for e in obs.evidence_collected]

    if meta == 1:  # CORRELATE
        ents = _entity_mentions(findings)
        if ents:
            entity = ents.most_common(1)[0][0]
        elif visible_ips:
            entity = visible_ips[0]
        else:
            entity = ""
        return CyberAction(action_type="correlate_events", entity=entity), \
            investigate_cycle_idx

    if meta == 2:  # IDENTIFY top IP
        ip_mentions = _evidence_ip_mentions(findings)
        if ip_mentions:
            top_ip = ip_mentions.most_common(1)[0][0]
        elif visible_ips:
            top_ip = visible_ips[0]
        else:
            top_ip = ""
        return CyberAction(action_type="identify_attacker", ip=top_ip), \
            investigate_cycle_idx

    # CLOSE_BENIGN
    return CyberAction(
        action_type="close_as_benign",
        summary="Internet scanner / authorized red-team activity, no impact.",
    ), investigate_cycle_idx


# ─────────────────────────────────────────────────────────────────────────────
# Tiny softmax policy
# ─────────────────────────────────────────────────────────────────────────────
class SoftmaxPolicy:
    def __init__(self, n_features: int = FEATURE_DIM, n_actions: int = META_DIM,
                 seed: int = 0, lr: float = 0.05):
        rng = np.random.RandomState(seed)
        self.W = rng.randn(n_actions, n_features).astype(np.float32) * 0.05
        self.b = np.zeros(n_actions, dtype=np.float32)
        self.lr = lr

    def logits(self, phi: np.ndarray) -> np.ndarray:
        return self.W @ phi + self.b

    def probs(self, phi: np.ndarray) -> np.ndarray:
        z = self.logits(phi)
        z -= z.max()
        e = np.exp(z)
        return e / e.sum()

    def sample(self, phi: np.ndarray, rng: random.Random) -> int:
        p = self.probs(phi)
        r = rng.random()
        cum = 0.0
        for i, pi in enumerate(p):
            cum += float(pi)
            if r <= cum:
                return i
        return len(p) - 1

    def grad_log_pi(self, phi: np.ndarray, action_idx: int):
        p = self.probs(phi)
        one_hot = np.zeros_like(p)
        one_hot[action_idx] = 1.0
        delta = (one_hot - p)
        dW = np.outer(delta, phi).astype(np.float32)
        db = delta.astype(np.float32)
        return dW, db


# ─────────────────────────────────────────────────────────────────────────────
# Episode runners
# ─────────────────────────────────────────────────────────────────────────────
def run_random_meta_episode(env, rng: random.Random,
                            seed: Optional[int] = None) -> Tuple[float, bool]:
    obs = env.reset(seed=seed)
    cycle = 0
    total = 0.0
    while not obs.done:
        meta = rng.randint(0, META_DIM - 1)
        a, cycle = meta_to_action(meta, obs, rng, cycle)
        obs = env.step(a)
        total += obs.reward
    correct = bool(env.state.terminal_correct)
    return total, correct


def run_policy_episode(env, policy: SoftmaxPolicy, rng: random.Random,
                       seed: Optional[int] = None,
                       greedy: bool = False) -> Tuple[float, List, bool]:
    obs = env.reset(seed=seed)
    cycle = 0
    traj = []
    total = 0.0
    while not obs.done:
        phi = featurize(obs)
        if greedy:
            meta = int(np.argmax(policy.probs(phi)))
        else:
            meta = policy.sample(phi, rng)
        a, cycle = meta_to_action(meta, obs, rng, cycle)
        obs = env.step(a)
        traj.append((phi, meta, float(obs.reward)))
        total += obs.reward
    correct = bool(env.state.terminal_correct)
    return total, traj, correct


def reinforce_update(policy: SoftmaxPolicy, traj, gamma: float = 0.97,
                     baseline: float = 0.0,
                     entropy_coef: float = 0.01) -> float:
    G = 0.0
    returns = []
    for _, _, r in reversed(traj):
        G = r + gamma * G
        returns.append(G)
    returns.reverse()
    returns = np.array(returns, dtype=np.float32)
    advs = returns - baseline

    total_dW = np.zeros_like(policy.W)
    total_db = np.zeros_like(policy.b)
    surrogate_loss = 0.0
    for (phi, aidx, _r), adv in zip(traj, advs):
        dW, db = policy.grad_log_pi(phi, aidx)
        total_dW += dW * adv
        total_db += db * adv
        # Entropy regulariser to avoid premature collapse
        p = policy.probs(phi)
        ent = -float(np.sum(p * np.log(p + 1e-9)))
        # Entropy gradient: d/dW(-sum p log p)
        # For softmax, gradient is messy; approximate by perturbing toward uniform.
        # Just add a small uniform pull on dW
        uniform_pull = np.outer(
            (np.ones_like(p) / len(p) - p), phi
        ).astype(np.float32)
        total_dW += entropy_coef * uniform_pull
        log_pi = math.log(max(1e-9, float(p[aidx])))
        surrogate_loss += -log_pi * float(adv) - entropy_coef * ent

    n = max(1, len(traj))
    policy.W += policy.lr * (total_dW / n)
    policy.b += policy.lr * (total_db / n)
    return float(surrogate_loss / n)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main(
    n_episodes: int = 1200,
    eval_episodes: int = 60,
    out_dir: str = "training_runs/reinforce",
    assets_dir: str = "assets",
    seed: int = 7,
    use_curriculum: bool = True,
    eval_greedy: bool = True,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(assets_dir, exist_ok=True)

    rng = random.Random(seed)
    np.random.seed(seed)

    if use_curriculum:
        train_env = CurriculumEnv(window=20, promote_after=12, seed=seed + 1)
    else:
        train_env = CyberSOCEnv()
    policy = SoftmaxPolicy(seed=seed + 2, lr=0.08)

    rolling_reward: deque = deque(maxlen=30)
    log: List[Dict[str, Any]] = []
    print(f"[train] {n_episodes} episodes against "
          f"{'CurriculumEnv' if use_curriculum else 'CyberSOCEnv'}")

    for ep in range(1, n_episodes + 1):
        ep_seed = seed + 1000 + ep
        total, traj, correct = run_policy_episode(train_env, policy, rng, seed=ep_seed)
        if use_curriculum:
            train_env.record_episode_reward(total)
        baseline = float(np.mean(rolling_reward)) if rolling_reward else 0.0
        loss = reinforce_update(policy, traj, gamma=0.97, baseline=baseline)
        rolling_reward.append(total)

        tier = train_env.tier if use_curriculum else 0
        tier_name = train_env.tier_name if use_curriculum else "(no curriculum)"
        scen = train_env.state.scenario_type if use_curriculum else "(mixed)"
        log.append({
            "episode": ep,
            "reward": total,
            "loss": loss,
            "rolling_reward_30": float(np.mean(rolling_reward)),
            "tier": tier,
            "tier_name": tier_name,
            "evidence": train_env.state.evidence_count,
            "terminal_correct": correct,
            "scenario": scen,
        })
        if ep % 100 == 0 or ep == 1:
            print(f"  ep {ep:4d} | reward={total:+.3f} loss={loss:+.4f} "
                  f"roll30={np.mean(rolling_reward):+.3f} tier={tier} "
                  f"({tier_name})")

    # ─── Evaluation: random vs trained, all 6 scenarios, eval_episodes/scenario
    print(f"\n[eval] {eval_episodes} episodes random vs trained per scenario")
    eval_rng = random.Random(seed + 9999)
    random_rewards: List[float] = []
    trained_rewards: List[float] = []
    random_correct = 0
    trained_correct = 0
    per_scenario: Dict[str, Dict[str, List[float]]] = {
        s: {"random": [], "trained": []} for s in SCENARIO_TYPES
    }
    n_per_scen = max(1, eval_episodes // len(SCENARIO_TYPES))
    for scen in SCENARIO_TYPES:
        for k in range(n_per_scen):
            s_seed = seed + 50_000 + hash((scen, k)) % 100_000
            # random
            e_r = CyberSOCEnv()
            tot_r, ok_r = run_random_meta_episode(
                e_r, random.Random(s_seed * 13 + 1),
                seed=s_seed,
            )
            # NOTE: run_random_meta_episode uses default scenario.
            # Force scenario via a targeted reset:
            e_r2 = CyberSOCEnv()
            obs_r = e_r2.reset(seed=s_seed, scenario_type=scen)
            cycle = 0
            tot_r = 0.0
            rng_r = random.Random(s_seed * 13 + 1)
            while not obs_r.done:
                meta = rng_r.randint(0, META_DIM - 1)
                a, cycle = meta_to_action(meta, obs_r, rng_r, cycle)
                obs_r = e_r2.step(a)
                tot_r += obs_r.reward
            ok_r = bool(e_r2.state.terminal_correct)
            random_rewards.append(tot_r)
            per_scenario[scen]["random"].append(tot_r)
            if ok_r:
                random_correct += 1

            # trained
            e_t = CyberSOCEnv()
            obs_t = e_t.reset(seed=s_seed, scenario_type=scen)
            cycle = 0
            tot_t = 0.0
            rng_t = random.Random(s_seed * 17 + 3)
            while not obs_t.done:
                phi = featurize(obs_t)
                if eval_greedy:
                    meta = int(np.argmax(policy.probs(phi)))
                else:
                    meta = policy.sample(phi, rng_t)
                a, cycle = meta_to_action(meta, obs_t, rng_t, cycle)
                obs_t = e_t.step(a)
                tot_t += obs_t.reward
            ok_t = bool(e_t.state.terminal_correct)
            trained_rewards.append(tot_t)
            per_scenario[scen]["trained"].append(tot_t)
            if ok_t:
                trained_correct += 1

    n_total = len(random_rewards)
    eval_summary = {
        "n_eval_episodes": n_total,
        "random_mean_reward": float(np.mean(random_rewards)),
        "random_success_rate": random_correct / n_total,
        "trained_mean_reward": float(np.mean(trained_rewards)),
        "trained_success_rate": trained_correct / n_total,
        "per_scenario": {
            s: {
                "random_mean": float(np.mean(per_scenario[s]["random"])) if per_scenario[s]["random"] else 0.0,
                "trained_mean": float(np.mean(per_scenario[s]["trained"])) if per_scenario[s]["trained"] else 0.0,
                "n_episodes": len(per_scenario[s]["random"]),
            }
            for s in SCENARIO_TYPES
        },
    }

    with open(os.path.join(out_dir, "training_log.json"), "w") as f:
        json.dump(log, f, indent=2)
    with open(os.path.join(out_dir, "eval_results.json"), "w") as f:
        json.dump(eval_summary, f, indent=2)
    np.savez(os.path.join(out_dir, "policy.npz"), W=policy.W, b=policy.b)

    print()
    print("=== EVALUATION ===")
    print(f"  Random meta-policy:  mean reward = {eval_summary['random_mean_reward']:+.3f}  "
          f"success = {eval_summary['random_success_rate']:.1%}")
    print(f"  REINFORCE-trained:   mean reward = {eval_summary['trained_mean_reward']:+.3f}  "
          f"success = {eval_summary['trained_success_rate']:.1%}")
    lift = eval_summary['trained_mean_reward'] - eval_summary['random_mean_reward']
    print(f"  Reward lift: {lift:+.3f}")

    # ─── Plots ──────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eps = np.array([r["episode"] for r in log])
    rewards = np.array([r["reward"] for r in log])
    losses = np.array([r["loss"] for r in log])
    tiers = np.array([r["tier"] for r in log])

    def smooth(arr, k=30):
        if len(arr) < k:
            return arr
        kernel = np.ones(k) / k
        return np.convolve(arr, kernel, mode="valid")

    # Reward curve
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.plot(eps, rewards, color="lightgray", lw=0.7, label="raw episode reward")
    sm = smooth(rewards)
    sm_x = eps[len(eps) - len(sm):]
    ax.plot(sm_x, sm, color="#2c7fb8", lw=2.6, label="smoothed (window=30)")
    ax.axhline(eval_summary["random_mean_reward"], color="#d95f02",
               lw=1.4, linestyle="--",
               label=f"random eval baseline ({eval_summary['random_mean_reward']:+.2f})")
    ax.axhline(eval_summary["trained_mean_reward"], color="#1b9e77",
               lw=1.4, linestyle="--",
               label=f"trained eval ({eval_summary['trained_mean_reward']:+.2f})")
    ax.set_xlabel("Training episode (#)")
    ax.set_ylabel("Episode cumulative reward")
    ax.set_title("CyberSOC Arena - REINFORCE training reward curve")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(assets_dir, "reward_curve.png"), dpi=130)
    plt.close(fig)

    # Loss curve
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.plot(eps, losses, color="#7570b3", lw=0.6, alpha=0.5, label="raw surrogate loss")
    if len(losses) >= 30:
        sm_l = smooth(losses)
        ax.plot(sm_x, sm_l, color="#7570b3", lw=2.4, label="smoothed (window=30)")
    ax.set_xlabel("Training episode (#)")
    ax.set_ylabel("REINFORCE surrogate loss (lower = better gradient signal)")
    ax.set_title("CyberSOC Arena - policy gradient loss")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(assets_dir, "loss_curve.png"), dpi=130)
    plt.close(fig)

    # Curriculum progression
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.step(eps, tiers, where="post", color="#d95f02", lw=2.0)
    ax.set_xlabel("Training episode (#)")
    ax.set_ylabel("Curriculum tier (unlocked scenarios)")
    ax.set_yticks(range(6))
    ax.set_yticklabels(["0  Novice", "1  Junior", "2  Mid",
                         "3  Senior", "4  Lead", "5  APT-hunter"])
    ax.set_title("CyberSOC Arena - adaptive curriculum tier over training")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(assets_dir, "curriculum_progress.png"), dpi=130)
    plt.close(fig)

    # Per-scenario baseline comparison
    fig, ax = plt.subplots(figsize=(9.5, 4.7))
    x = np.arange(len(SCENARIO_TYPES))
    rand_means = [eval_summary["per_scenario"][s]["random_mean"] for s in SCENARIO_TYPES]
    train_means = [eval_summary["per_scenario"][s]["trained_mean"] for s in SCENARIO_TYPES]
    w = 0.36
    ax.bar(x - w / 2, rand_means, w, label="Random meta-policy", color="#d95f02")
    ax.bar(x + w / 2, train_means, w, label="REINFORCE-trained", color="#1b9e77")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", "\n") for s in SCENARIO_TYPES],
                       rotation=0, fontsize=9)
    ax.set_ylabel("Mean episode reward")
    ax.set_title(f"Mean reward per scenario  (n={n_per_scen} eval episodes per bar)")
    ax.axhline(0, color="black", lw=0.8)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(assets_dir, "baseline_comparison.png"), dpi=130)
    plt.close(fig)

    print()
    print("=== ARTIFACTS ===")
    for p in [
        f"{out_dir}/training_log.json",
        f"{out_dir}/eval_results.json",
        f"{out_dir}/policy.npz",
        f"{assets_dir}/reward_curve.png",
        f"{assets_dir}/loss_curve.png",
        f"{assets_dir}/curriculum_progress.png",
        f"{assets_dir}/baseline_comparison.png",
    ]:
        print(f"  - {p}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=1200)
    p.add_argument("--eval", type=int, default=60)
    p.add_argument("--out-dir", default="training_runs/reinforce")
    p.add_argument("--assets-dir", default="assets")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--no-curriculum", action="store_true")
    p.add_argument("--stochastic-eval", action="store_true",
                   help="Sample from policy at eval time instead of greedy argmax.")
    args = p.parse_args()
    main(
        n_episodes=args.episodes,
        eval_episodes=args.eval,
        out_dir=args.out_dir,
        assets_dir=args.assets_dir,
        seed=args.seed,
        use_curriculum=not args.no_curriculum,
        eval_greedy=not args.stochastic_eval,
    )
