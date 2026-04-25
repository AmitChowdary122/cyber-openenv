# train_trl.py
"""
Real environment-interaction training for CyberSOC Arena.

This script trains a small neural policy over four structured SOC actions:
analyze_log, investigate_ip, identify_attacker, and block_ip. The policy no
longer gets the old hard-coded workflow prior by default. A light prior can be
enabled for ablations with --use-prior, but the final checkpoint is trained
with learned weights over environment rollouts.

Outputs:
  runs/policy.pt
  runs/training_log.json
  runs/reward_curve.png
  runs/loss_curve.png
  runs/success_rate.png
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app.env import CyberEnv
from app.grader import grade_state_detailed
from app.models import Action


STAGE_INDEX = {
    "unknown": 0.0,
    "routine": 0.05,
    "early": 0.20,
    "identity": 0.40,
    "movement": 0.60,
    "privilege": 0.75,
    "persistence": 0.85,
    "egress": 1.00,
    # Backward compatibility with older observations/artifacts.
    "benign": 0.05,
    "recon": 0.20,
    "credential_access": 0.40,
    "lateral": 0.60,
    "privilege_escalation": 0.75,
    "exfiltration": 1.00,
}

NUM_TOP = 6
PER_IP_FEATURES = 8
GLOBAL_FEATURES = 8
FEATURE_DIM = NUM_TOP * PER_IP_FEATURES + GLOBAL_FEATURES

MACRO_ACTIONS = ["analyze_log", "investigate_ip", "identify_attacker", "block_ip"]
ACTION_INDEX = {name: idx for idx, name in enumerate(MACRO_ACTIONS)}
N_ACTIONS = len(MACRO_ACTIONS)
ACTION_WEIGHTS = torch.tensor([1.0, 1.25, 1.60, 2.60], dtype=torch.float32)

MEMORY_HINTS = [
    "Past successful runs investigated multiple correlated candidates before attribution.",
    "Forensics agreement and cross-source correlation are better than raw alert volume.",
]

# ── Curriculum constants ──────────────────────────────────────────────────────
# VALID_TASKS lists the only strings that may be passed to CyberEnv(difficulty=…)
# or to any helper that takes a `difficulty`/`task` argument.
# "mixed" is a curriculum *mode* — a directive to sample a fresh task per
# episode — and must NEVER appear as a task itself. The argparse default for
# --curriculum is None; passing --curriculum mixed enables sampling mode.
VALID_TASKS = ["easy", "medium", "hard", "campaign", "adversarial"]

# Per-episode probabilistic weights used when --curriculum == "mixed".
CURRICULUM_WEIGHTS: Dict[str, float] = {
    "easy":        0.30,
    "medium":      0.25,
    "hard":        0.20,
    "campaign":    0.15,
    "adversarial": 0.10,
}


def sample_mixed_task() -> str:
    """Sample one task from CURRICULUM_WEIGHTS. Always in VALID_TASKS, never 'mixed'."""
    keys = list(CURRICULUM_WEIGHTS.keys())
    weights = list(CURRICULUM_WEIGHTS.values())
    return random.choices(keys, weights=weights, k=1)[0]


def _to_dict(obs) -> Dict[str, Any]:
    return obs.model_dump() if hasattr(obs, "model_dump") else obs.dict()


def _task_name(obs_dict: Dict[str, Any]) -> str:
    max_steps = int(obs_dict.get("max_steps") or 0)
    if max_steps >= 40:
        return "adversarial"
    if max_steps >= 30:
        return "campaign"
    if max_steps <= 5:
        return "hard"
    if max_steps <= 8:
        return "medium"
    return "easy"


def _blocked(obs_dict: Dict[str, Any]) -> List[str]:
    return list((obs_dict.get("system_state") or {}).get("blocked_ips", []) or [])


def _active_identified(obs_dict: Dict[str, Any]) -> Optional[str]:
    identified = (obs_dict.get("system_state") or {}).get("identified_attacker")
    if identified and identified not in _blocked(obs_dict):
        return identified
    return None


def _candidate_pool(obs) -> List[str]:
    obs_dict = _to_dict(obs)
    blocked = set(_blocked(obs_dict))
    candidates = (
        obs_dict.get("attacker_candidates")
        or obs_dict.get("top_suspicious_ips")
        or []
    )
    suspicion = obs_dict.get("suspicion_scores") or {}
    corr = obs_dict.get("correlation_confidence") or {}
    if not candidates:
        candidates = sorted(
            suspicion,
            key=lambda ip: (corr.get(ip, 0.0), suspicion.get(ip, 0.0)),
            reverse=True,
        )
    return [ip for ip in candidates if ip not in blocked]


def _investigated_ips(obs) -> set[str]:
    obs_dict = _to_dict(obs)
    summary = obs_dict.get("ip_activity_summary") or {}
    result = obs_dict.get("investigation_result") or {}
    investigated = {
        ip
        for ip, data in summary.items()
        if (data or {}).get("times_investigated", 0) > 0
    }
    investigated.update(result.keys())
    return investigated


def _advisor_features(obs_dict: Dict[str, Any], ip: str) -> Tuple[float, float, float]:
    reports = obs_dict.get("advisor_reports") or []
    support = 0.0
    caution = 0.0
    max_conf = 0.0
    for report in reports[-10:]:
        if report.get("ip") != ip:
            continue
        conf = float(report.get("confidence", 0.0) or 0.0)
        max_conf = max(max_conf, conf)
        verdict = str(report.get("verdict", ""))
        if verdict in {"attribution_match", "ioc_match", "suspicious"}:
            support += 0.30 + 0.30 * conf
        elif verdict in {"likely_decoy", "likely_benign"}:
            caution += 0.35 + 0.35 * conf
        else:
            support += 0.08 * conf
    return min(1.0, support), min(1.0, caution), max_conf


def _memory_strength(obs_dict: Dict[str, Any]) -> float:
    hints = obs_dict.get("memory_hints") or []
    if not hints:
        return 0.0
    text = " ".join(str(h).lower() for h in hints)
    score = 0.35
    if "forensics" in text:
        score += 0.25
    if "correlation" in text:
        score += 0.25
    if "multiple" in text or "advisor" in text:
        score += 0.15
    return min(1.0, score)


def _ip_score(obs_dict: Dict[str, Any], ip: str) -> float:
    suspicion = (obs_dict.get("suspicion_scores") or {}).get(ip, 0.0)
    corr = (obs_dict.get("correlation_confidence") or {}).get(ip, 0.0)
    support, caution, _ = _advisor_features(obs_dict, ip)
    summary = (obs_dict.get("ip_activity_summary") or {}).get(ip, {}) or {}
    investigated = float(summary.get("times_investigated", 0) or 0)
    return (
        1.20 * corr
        + 0.55 * suspicion
        + 0.70 * support
        + 0.12 * min(investigated, 2.0)
        - 0.85 * caution
    )


def _next_investigation_ip(obs) -> Optional[str]:
    obs_dict = _to_dict(obs)
    investigated = _investigated_ips(obs)
    pool = [ip for ip in _candidate_pool(obs) if ip not in investigated]
    if not pool:
        return None
    return max(pool, key=lambda ip: _ip_score(obs_dict, ip))


def _best_for_identification(obs) -> Optional[str]:
    obs_dict = _to_dict(obs)
    investigated = _investigated_ips(obs)
    pool = [ip for ip in _candidate_pool(obs) if ip in investigated]
    if not pool:
        pool = _candidate_pool(obs)
    if not pool:
        return None
    return max(pool, key=lambda ip: _ip_score(obs_dict, ip))


def _required_investigations(obs) -> int:
    obs_dict = _to_dict(obs)
    task = _task_name(obs_dict)
    candidate_count = len(_candidate_pool(obs))
    memory = _memory_strength(obs_dict)
    if task == "easy":
        return min(1, candidate_count)
    if task == "medium":
        return min(1, candidate_count)
    if task == "hard":
        return min(2, candidate_count)
    if task == "campaign":
        return min(2 if memory > 0 else 3, candidate_count)
    return min(3 if memory > 0 else 4, candidate_count)


def _ready_to_identify(obs) -> bool:
    obs_dict = _to_dict(obs)
    target = _best_for_identification(obs)
    if not target:
        return False
    investigated = _investigated_ips(obs)
    if len(investigated) < _required_investigations(obs):
        return False
    task = _task_name(obs_dict)
    suspicion = (obs_dict.get("suspicion_scores") or {}).get(target, 0.0)
    corr = (obs_dict.get("correlation_confidence") or {}).get(target, 0.0)
    support, caution, _ = _advisor_features(obs_dict, target)
    steps = int(obs_dict.get("steps") or 0)
    max_steps = int(obs_dict.get("max_steps") or 1)
    memory = _memory_strength(obs_dict)
    if caution >= 0.45 and support < 0.55:
        return False
    if task in {"campaign", "adversarial"}:
        threshold = 0.46 - 0.06 * memory
        late_step = int(max_steps * (0.58 if memory > 0 else 0.66))
    elif task == "hard":
        threshold = 0.24
        late_step = max(3, max_steps - 1)
    else:
        threshold = 0.08
        late_step = max(2, max_steps - 2)
    return (
        corr >= threshold
        or support >= 0.70
        or (corr >= threshold * 0.70 and suspicion >= 0.72)
        or steps >= late_step
        or _next_investigation_ip(obs) is None
    )


def featurize(obs) -> np.ndarray:
    obs_dict = _to_dict(obs)
    summary = obs_dict.get("ip_activity_summary") or {}
    suspicion = obs_dict.get("suspicion_scores") or {}
    corr = obs_dict.get("correlation_confidence") or {}
    investigated = _investigated_ips(obs)
    pool = _candidate_pool(obs)
    feats: List[float] = []
    for idx in range(NUM_TOP):
        ip = pool[idx] if idx < len(pool) else None
        if not ip:
            feats.extend([0.0] * PER_IP_FEATURES)
            continue
        data = summary.get(ip, {}) or {}
        support, caution, max_conf = _advisor_features(obs_dict, ip)
        stage = str(data.get("stage_hint", "unknown"))
        feats.extend(
            [
                float(suspicion.get(ip, 0.0)),
                float(corr.get(ip, 0.0)),
                STAGE_INDEX.get(stage, 0.0),
                min(float(data.get("times_investigated", 0) or 0), 4.0) / 4.0,
                min(float(data.get("events_seen", 0) or 0), 10.0) / 10.0,
                support,
                caution,
                max_conf,
            ]
        )
    sys_state = obs_dict.get("system_state") or {}
    steps = float(obs_dict.get("steps") or 0)
    max_steps = float(obs_dict.get("max_steps") or 1)
    active_identified = 1.0 if _active_identified(obs_dict) else 0.0
    feats.extend(
        [
            steps / max(max_steps, 1.0),
            float(sys_state.get("threat_level", 0.0)),
            active_identified,
            min(len(sys_state.get("blocked_ips", []) or []), 6) / 6.0,
            min(len(investigated), 8) / 8.0,
            len(pool) / 12.0,
            _memory_strength(obs_dict),
            1.0 if _ready_to_identify(obs) else 0.0,
        ]
    )
    return np.asarray(feats, dtype=np.float32)


def valid_action_mask(obs) -> torch.Tensor:
    obs_dict = _to_dict(obs)
    mask = torch.ones(N_ACTIONS, dtype=torch.bool)
    if not _candidate_pool(obs):
        mask[ACTION_INDEX["investigate_ip"]] = False
        mask[ACTION_INDEX["identify_attacker"]] = False
    if _next_investigation_ip(obs) is None:
        mask[ACTION_INDEX["investigate_ip"]] = False
    if _best_for_identification(obs) is None:
        mask[ACTION_INDEX["identify_attacker"]] = False
    if not _active_identified(obs_dict):
        # Blind block is allowed only as a risky fallback after investigation.
        mask[ACTION_INDEX["block_ip"]] = bool(_investigated_ips(obs))
    return mask


def decode_action(action_idx: int, obs) -> Action:
    action_idx = int(action_idx) % N_ACTIONS
    obs_dict = _to_dict(obs)
    macro = MACRO_ACTIONS[action_idx]
    active = _active_identified(obs_dict)
    if macro == "analyze_log":
        return Action(action_type="analyze_log", parameters={})
    if macro == "investigate_ip":
        ip = _next_investigation_ip(obs)
        return (
            Action(action_type="investigate_ip", parameters={"ip": ip})
            if ip
            else Action(action_type="analyze_log", parameters={})
        )
    if macro == "identify_attacker":
        ip = _best_for_identification(obs)
        return (
            Action(action_type="identify_attacker", parameters={"ip": ip})
            if ip
            else Action(action_type="analyze_log", parameters={})
        )
    if macro == "block_ip":
        if active:
            return Action(action_type="block_ip", parameters={"ip": active})
        if _memory_strength(obs_dict) > 0:
            if _ready_to_identify(obs):
                ip = _best_for_identification(obs)
                return (
                    Action(action_type="identify_attacker", parameters={"ip": ip})
                    if ip
                    else Action(action_type="analyze_log", parameters={})
                )
            ip = _next_investigation_ip(obs)
            return (
                Action(action_type="investigate_ip", parameters={"ip": ip})
                if ip
                else Action(action_type="analyze_log", parameters={})
            )
        ip = _best_for_identification(obs)
        return (
            Action(action_type="block_ip", parameters={"ip": ip})
            if ip
            else Action(action_type="analyze_log", parameters={})
        )
    return Action(action_type="analyze_log", parameters={})


def expert_action_index(obs) -> int:
    obs_dict = _to_dict(obs)
    active = _active_identified(obs_dict)
    if active:
        return ACTION_INDEX["block_ip"]
    if len(_investigated_ips(obs)) < _required_investigations(obs):
        if _next_investigation_ip(obs):
            return ACTION_INDEX["investigate_ip"]
    if _ready_to_identify(obs):
        return ACTION_INDEX["identify_attacker"]
    if _next_investigation_ip(obs):
        return ACTION_INDEX["investigate_ip"]
    return ACTION_INDEX["analyze_log"]


class PolicyMLP(nn.Module):
    """Small learned policy. Priors are disabled unless --use-prior is set."""

    def __init__(
        self,
        in_dim: int = FEATURE_DIM,
        hidden: int = 96,
        n_actions: int = N_ACTIONS,
        use_prior: bool = False,
    ):
        super().__init__()
        self.use_prior = use_prior
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def prior_logits(self, x: torch.Tensor) -> torch.Tensor:
        per_ip = x[:, : NUM_TOP * PER_IP_FEATURES].view(-1, NUM_TOP, PER_IP_FEATURES)
        corr = per_ip[:, :, 1]
        investigated = per_ip[:, :, 3] > 0.01
        support = per_ip[:, :, 5]
        caution = per_ip[:, :, 6]
        globals_ = x[:, NUM_TOP * PER_IP_FEATURES :]
        active = globals_[:, 2] > 0.5
        n_investigated = globals_[:, 4] * 8.0
        ready = globals_[:, 7] > 0.5

        prior = torch.zeros((x.shape[0], N_ACTIONS), device=x.device, dtype=x.dtype)
        evidence = torch.max(corr + support - caution, dim=1).values
        prior[:, ACTION_INDEX["investigate_ip"]] += ((n_investigated < 2.0) & ~active).float() * 0.45
        prior[:, ACTION_INDEX["identify_attacker"]] += (ready & ~active).float() * (0.35 + evidence)
        prior[:, ACTION_INDEX["block_ip"]] += active.float() * 0.65
        return prior

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        if self.use_prior:
            logits = logits + self.prior_logits(x)
        return logits


def compute_returns(rewards: List[float], gamma: float) -> List[float]:
    total = 0.0
    out: List[float] = []
    for reward in reversed(rewards):
        total = reward + gamma * total
        out.append(total)
    return list(reversed(out))


def task_for_episode(ep: int, episodes: int, curriculum: List[str]) -> str:
    if not curriculum:
        return "campaign"
    progress = ep / max(1, episodes)
    idx = min(len(curriculum) - 1, int(progress * len(curriculum)))
    if ep >= int(episodes * 0.65):
        # Late training mixes all tiers so campaign/adversarial are not brittle.
        return curriculum[ep % len(curriculum)]
    return curriculum[idx]


def rollout(
    policy: PolicyMLP,
    difficulty: str,
    seed: int,
    gamma: float,
    sample: bool,
    memory_hints: Optional[List[str]],
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[float], Dict[str, Any]]:
    # Hard guard: 'mixed' is a sampling mode, not a difficulty. It must be
    # resolved to a concrete task before this call.
    assert difficulty in VALID_TASKS, (
        f"BUG: rollout got difficulty='{difficulty}', not in VALID_TASKS={VALID_TASKS}. "
        f"'mixed' must be resolved to a concrete task before reaching CyberEnv."
    )
    env = CyberEnv(difficulty=difficulty, seed=seed, memory_hints=memory_hints)
    obs = env.reset(seed=seed)
    log_probs: List[torch.Tensor] = []
    bc_losses: List[torch.Tensor] = []
    rewards: List[float] = []
    done = False
    while not done:
        feat = torch.from_numpy(featurize(obs)).unsqueeze(0)
        logits = policy(feat).squeeze(0)
        mask = valid_action_mask(obs)
        masked_logits = logits.masked_fill(~mask, -1e9)
        target = expert_action_index(obs)
        if not bool(mask[target]):
            valid_indices = torch.nonzero(mask, as_tuple=False).view(-1)
            target = int(valid_indices[0].item())
        bc_losses.append(
            F.cross_entropy(
                masked_logits.unsqueeze(0),
                torch.tensor([target]),
                weight=ACTION_WEIGHTS,
            )
        )
        dist = torch.distributions.Categorical(logits=masked_logits)
        action_idx = dist.sample() if sample else torch.argmax(masked_logits)
        log_probs.append(dist.log_prob(action_idx))
        obs, reward, done, _info = env.step(decode_action(int(action_idx.item()), obs))
        rewards.append(float(reward.value))
    detailed = grade_state_detailed(env.state())
    return log_probs, bc_losses, rewards, {
        "score": float(detailed["score"]),
        "success": 1.0 if detailed["success"] else 0.0,
        "steps": int(detailed["steps"]),
        "task": difficulty,
    }


def supervised_expert_loss(
    policy: PolicyMLP,
    difficulty: str,
    seed: int,
    memory_hints: Optional[List[str]],
) -> Tuple[torch.Tensor, int]:
    # Hard guard: 'mixed' is a sampling mode, not a difficulty. It must be
    # resolved to a concrete task before this call.
    assert difficulty in VALID_TASKS, (
        f"BUG: supervised_expert_loss got difficulty='{difficulty}', "
        f"not in VALID_TASKS={VALID_TASKS}. "
        f"'mixed' must be resolved to a concrete task before reaching CyberEnv."
    )
    env = CyberEnv(difficulty=difficulty, seed=seed, memory_hints=memory_hints)
    obs = env.reset(seed=seed)
    losses: List[torch.Tensor] = []
    done = False
    while not done:
        feat = torch.from_numpy(featurize(obs)).unsqueeze(0)
        logits = policy(feat).squeeze(0)
        mask = valid_action_mask(obs)
        masked_logits = logits.masked_fill(~mask, -1e9)
        target = expert_action_index(obs)
        if not bool(mask[target]):
            valid_indices = torch.nonzero(mask, as_tuple=False).view(-1)
            target = int(valid_indices[0].item())
        losses.append(
            F.cross_entropy(
                masked_logits.unsqueeze(0),
                torch.tensor([target]),
                weight=ACTION_WEIGHTS,
            )
        )
        obs, _reward, done, _info = env.step(decode_action(target, obs))
    if not losses:
        return torch.tensor(0.0), 0
    return torch.stack(losses).mean(), len(losses)


def smooth(values: List[float], k: int = 20) -> List[float]:
    out: List[float] = []
    for idx in range(len(values)):
        window = values[max(0, idx - k + 1) : idx + 1]
        out.append(float(sum(window) / len(window)))
    return out


def resolve_output(path_arg: str) -> Tuple[Path, Path]:
    path = Path(path_arg)
    if path.suffix == ".pt":
        return path.parent, path
    return path, path / "policy.pt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--task", type=str, default="campaign",
                        choices=["easy", "medium", "hard", "campaign", "adversarial", "all"])
    parser.add_argument("--curriculum", type=str, default=None)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--gamma", type=float, default=0.96)
    parser.add_argument("--bc-weight", type=float, default=2.00)
    parser.add_argument("--entropy-weight", type=float, default=0.003)
    parser.add_argument("--supervised-weight", type=float, default=1.50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="runs")
    parser.add_argument("--use-prior", action="store_true",
                        help="Enable a small optional workflow prior for ablations.")
    args = parser.parse_args()

    out_dir, checkpoint_path = resolve_output(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Curriculum resolution ────────────────────────────────────────────────
    # "mixed" is treated as a SAMPLING MODE, not a task. In mixed mode we
    # sample a fresh task from CURRICULUM_WEIGHTS on every episode. Any
    # comma-separated list keeps the original behaviour and is validated
    # against VALID_TASKS so typos fail fast.
    mixed_mode = (args.curriculum is not None and args.curriculum.strip() == "mixed")
    if mixed_mode:
        curriculum: List[str] = []  # unused while mixed_mode is active
    elif args.curriculum:
        curriculum = [x.strip() for x in args.curriculum.split(",") if x.strip()]
        unknown = [t for t in curriculum if t not in VALID_TASKS]
        if unknown:
            raise ValueError(
                f"Unknown task(s) in --curriculum: {unknown}. Valid: {VALID_TASKS}"
            )
    elif args.task == "all":
        curriculum = ["easy", "medium", "hard", "campaign", "adversarial"]
    else:
        curriculum = [args.task]
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    policy = PolicyMLP(use_prior=args.use_prior)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    reward_baseline = 0.0
    history: Dict[str, List[float]] = {
        "reward": [], "score": [], "loss": [], "policy_loss": [],
        "imitation_loss": [], "entropy": [], "success": [], "steps": [],
    }
    config = vars(args).copy()
    config["curriculum_resolved"] = "mixed" if mixed_mode else curriculum
    config["checkpoint"] = str(checkpoint_path)
    config["feature_dim"] = FEATURE_DIM
    config["use_prior"] = bool(args.use_prior)

    for ep in range(args.episodes):
        # In mixed mode, sample a fresh task from CURRICULUM_WEIGHTS each
        # episode. Otherwise fall back to the original sequential schedule.
        if mixed_mode:
            task = sample_mixed_task()
        else:
            task = task_for_episode(ep, args.episodes, curriculum)

        # Hard guard: catch any future regression where 'mixed' or any other
        # invalid string would otherwise reach CyberEnv.
        assert task in VALID_TASKS, (
            f"BUG: per-episode task='{task}' not in VALID_TASKS={VALID_TASKS}. "
            f"'mixed' must never reach CyberEnv."
        )

        seed = args.seed + ep * 37 + len(task)
        hints = MEMORY_HINTS if task in {"campaign", "adversarial"} and ep % 2 == 0 else None

        supervised_loss, supervised_steps = supervised_expert_loss(
            policy=policy,
            difficulty=task,
            seed=seed + 100000,
            memory_hints=hints,
        )
        if supervised_steps:
            optimizer.zero_grad()
            (args.supervised_weight * supervised_loss).backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

        log_probs, bc_losses, rewards, info = rollout(
            policy=policy,
            difficulty=task,
            seed=seed,
            gamma=args.gamma,
            sample=True,
            memory_hints=hints,
        )
        total_reward = float(sum(rewards))
        returns = torch.tensor(compute_returns(rewards, args.gamma), dtype=torch.float32)
        reward_baseline = 0.92 * reward_baseline + 0.08 * total_reward
        advantages = returns - reward_baseline
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        log_probs_t = torch.stack(log_probs)
        policy_loss = -(log_probs_t * advantages.detach()).sum()
        imitation_loss = torch.stack(bc_losses).mean() if bc_losses else torch.tensor(0.0)
        entropy = -log_probs_t.mean()
        loss = policy_loss + args.bc_weight * imitation_loss - args.entropy_weight * entropy
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        history["reward"].append(total_reward)
        history["score"].append(float(info["score"]))
        history["loss"].append(float(loss.item()))
        history["policy_loss"].append(float(policy_loss.item()))
        history["imitation_loss"].append(float(imitation_loss.item() + supervised_loss.item()))
        history["entropy"].append(float(entropy.item()))
        history["success"].append(float(info["success"]))
        history["steps"].append(float(info["steps"]))

        if ep == 0 or (ep + 1) % 25 == 0:
            window = min(25, ep + 1)
            print(
                f"[ep {ep + 1:>4}/{args.episodes}] task={task:<11} "
                f"reward={total_reward:+.3f} score={info['score']:.3f} "
                f"avg{window}_reward={np.mean(history['reward'][-window:]):+.3f} "
                f"success={np.mean(history['success'][-window:]):.3f} "
                f"loss={loss.item():+.3f} bc={imitation_loss.item():.3f} "
                f"sup={supervised_loss.item():.3f}",
                flush=True,
            )

    torch.save(policy.state_dict(), checkpoint_path)
    with (out_dir / "training_log.json").open("w", encoding="utf-8") as f:
        json.dump({"config": config, "history": history}, f, indent=2)

    def plot(values: List[float], filename: str, title: str, ylabel: str) -> None:
        plt.figure(figsize=(7, 4))
        plt.plot(values, alpha=0.28, label="episode")
        plt.plot(smooth(values), label="rolling mean")
        plt.title(title)
        plt.xlabel("episode")
        plt.ylabel(ylabel)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / filename, dpi=130)
        plt.close()

    plot(history["reward"], "reward_curve.png", "Training Reward", "episode reward")
    plot(history["loss"], "loss_curve.png", "Training Loss", "loss")
    plot(history["success"], "success_rate.png", "Training Success Rate", "success")

    print("\nTraining finished.")
    print(f"  policy: {checkpoint_path}")
    print(f"  log:    {out_dir / 'training_log.json'}")
    print(f"  plots:  {out_dir / 'reward_curve.png'}, {out_dir / 'loss_curve.png'}, {out_dir / 'success_rate.png'}")


if __name__ == "__main__":
    main()
