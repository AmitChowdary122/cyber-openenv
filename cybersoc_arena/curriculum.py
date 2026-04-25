"""
CurriculumEnv — Theme 4 (Self-Improvement) wrapper around CyberSOCEnv.

Six tiers, from "Novice Analyst" up to "Elite Hunter". Each tier has its own
scenario pool, an unlock_threshold (mean reward over a rolling window) and a
window size. When the rolling mean over the last `window` episodes meets the
threshold, the curriculum auto-advances to the next tier — unlocking harder
scenarios. The Elite tier (which contains the long_horizon_apt 20-step
scenario) is gated behind mastery of every other scenario type.

The wrapper is API-compatible with CyberSOCEnv (same reset/step/state) and
injects a small `curriculum` block into observations so the agent (and plots)
can see what level it is at.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional
import random

from .env import CyberSOCEnv, _HAS_OPENENV, _OpenEnvBase  # type: ignore


@dataclass
class Tier:
    level: int
    name: str
    description: str
    scenarios: List[str]
    unlock_threshold: Optional[float]
    window: int


TIERS: List[Tier] = [
    Tier(0, "Novice Analyst",
         "Triage the easy false-positive scanner alerts.",
         ["benign_scan"],
         unlock_threshold=0.50, window=8),
    Tier(1, "Junior Analyst",
         "Tell credential stuffing apart from noisy logins.",
         ["benign_scan", "credential_stuffing"],
         unlock_threshold=0.60, window=12),
    Tier(2, "Senior Analyst",
         "Trace phishing into lateral movement on file servers.",
         ["benign_scan", "credential_stuffing", "phishing_lateral"],
         unlock_threshold=0.68, window=15),
    Tier(3, "Threat Hunter",
         "Catch outbound exfiltration of customer data to staging C2.",
         ["benign_scan", "credential_stuffing",
          "phishing_lateral", "data_exfiltration"],
         unlock_threshold=0.74, window=20),
    Tier(4, "APT Hunter",
         "Untangle multi-stage Cobalt-Strike chains across VPN, jump, app.",
         ["benign_scan", "credential_stuffing", "phishing_lateral",
          "data_exfiltration", "multi_stage_chain"],
         unlock_threshold=0.80, window=20),
    Tier(5, "Elite Hunter",
         "Top tier: APT-41 kill chains, LockBit ransomware deployments, "
         "and SolarWinds-style supply-chain compromises — with adversarial "
         "decoys mixed in.",
         ["benign_scan", "credential_stuffing", "phishing_lateral",
          "data_exfiltration", "multi_stage_chain", "long_horizon_apt",
          "ransomware_deployment", "supply_chain_attack"],
         unlock_threshold=None, window=25),
]


# Convincing-looking external IPs used by the Elite-tier adversarial mode to
# slip extra decoys into the observation. These ranges are cribbed from real
# threat-intel feeds (TOR exit ranges, bulletproof hosters) — not necessarily
# malicious in this context, but visually indistinguishable from attackers.
_ADVERSARIAL_DECOY_PREFIXES = ("185.220.101", "185.220.102", "23.129.64",
                               "194.165.16", "45.142.213", "91.92.250")


def _adversarial_decoys(rng: random.Random, n: int = 2) -> List[str]:
    out: List[str] = []
    for _ in range(n):
        prefix = rng.choice(_ADVERSARIAL_DECOY_PREFIXES)
        out.append(f"{prefix}.{rng.randint(2, 250)}")
    return out


class CurriculumEnv(_OpenEnvBase):
    """
    Drop-in replacement for CyberSOCEnv that progresses through tiers as
    the agent's rolling-mean reward crosses each unlock threshold.

    Like :class:`CyberSOCEnv`, this inherits from
    :class:`openenv.core.Environment` when OpenEnv is installed.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self,
                 start_tier: int = 0,
                 seed: Optional[int] = None,
                 auto_advance: bool = True,
                 adversarial: bool = False):
        if _HAS_OPENENV:
            super().__init__()
        self._tier: int = max(0, min(start_tier, len(TIERS) - 1))
        self._rng: random.Random = random.Random(seed)
        self._auto_advance: bool = auto_advance
        self._adversarial: bool = adversarial

        # one rolling window per tier (so changing tiers re-collects)
        self._windows: Dict[int, Deque[float]] = {
            t.level: deque(maxlen=t.window) for t in TIERS
        }
        self._episode_count: int = 0
        self._tier_transitions: List[Dict[str, Any]] = []
        self._episode_cumulative_reward: float = 0.0
        self._current_scenario: Optional[str] = None

        # underlying env (we drive its scenario selection per-episode)
        env_seed = self._rng.randint(0, 1 << 30)
        self._env: CyberSOCEnv = CyberSOCEnv(seed=env_seed)

    # ----------------------------------------------------------------- API
    def reset(self, seed: Optional[int] = None,
              episode_id: Optional[str] = None,
              **kwargs: Any) -> Dict[str, Any]:
        if seed is not None:
            self._rng = random.Random(seed)
        self._episode_id = episode_id
        tier = TIERS[self._tier]
        scenario_type = self._rng.choice(tier.scenarios)
        self._current_scenario = scenario_type
        ep_seed = self._rng.randint(0, 1 << 30)
        obs = self._env.reset(scenario_type=scenario_type, seed=ep_seed)
        self._episode_cumulative_reward = 0.0

        # Adversarial mode: at the Elite tier, slip in 2 extra plausible-
        # looking external decoy IPs the agent has never heard of. They have
        # NO evidence attached, so any tool call against them yields "nothing
        # of interest" — a tax on impulsive triagers, a cheap ignore for
        # patient ones.
        adv_active = self._adversarial and self._tier >= len(TIERS) - 1
        if adv_active:
            extra = _adversarial_decoys(self._rng, n=2)
            existing = list(obs.get("known_entities", {}).get("ips", []))
            for ip in extra:
                if ip not in existing:
                    existing.append(ip)
            obs.setdefault("known_entities", {"ips": [], "hosts": []})
            obs["known_entities"]["ips"] = existing
            obs["adversarial_mode"] = True
            obs["adversarial_decoys"] = list(extra)
            self._adv_decoys = list(extra)
        else:
            obs["adversarial_mode"] = False
            self._adv_decoys = []

        obs["curriculum"] = self._curriculum_obs()
        return obs

    def step(self, raw_action: Any, timeout_s: Optional[float] = None,
             **kwargs: Any):
        obs, reward, done, info = self._env.step(raw_action)
        self._episode_cumulative_reward += reward

        if done:
            self._on_episode_end()

        cur_obs = self._curriculum_obs()
        obs["curriculum"] = cur_obs

        # Carry adversarial state into every step's observation so the agent
        # can see it throughout the episode (it's only added on reset by the
        # underlying CyberSOCEnv).
        adv_active = self._adversarial and self._tier >= len(TIERS) - 1
        obs["adversarial_mode"] = bool(adv_active)
        if adv_active and getattr(self, "_adv_decoys", None):
            existing = list(obs.get("known_entities", {}).get("ips", []))
            for ip in self._adv_decoys:
                if ip not in existing:
                    existing.append(ip)
            obs["known_entities"]["ips"] = existing
            obs["adversarial_decoys"] = list(self._adv_decoys)

        info = dict(info)
        info["curriculum_tier"] = self._tier
        info["curriculum_tier_name"] = TIERS[self._tier].name
        info["curriculum_scenario"] = self._current_scenario
        info["curriculum_progress"] = cur_obs.get("progress", 0.0)
        info["curriculum_rolling_mean"] = cur_obs.get("rolling_mean", 0.0)
        info["adversarial_mode"] = bool(adv_active)
        return obs, reward, done, info

    @property
    def state(self) -> Dict[str, Any]:
        s = dict(self._env.state)
        s["curriculum_tier"] = self._tier
        s["curriculum_tier_name"] = TIERS[self._tier].name
        s["curriculum_scenario"] = self._current_scenario
        s["curriculum_episode_count"] = self._episode_count
        s["curriculum_at_max_tier"] = self._tier >= len(TIERS) - 1
        s["episode_id"] = getattr(self, "_episode_id", None)
        return s

    def get_metadata(self):
        return self._env.get_metadata()

    # ---------------------------------------------------------- internals
    def _on_episode_end(self) -> None:
        self._episode_count += 1
        win = self._windows[self._tier]
        win.append(self._episode_cumulative_reward)

        if not self._auto_advance:
            return
        tier = TIERS[self._tier]
        if tier.unlock_threshold is None:
            return  # already at the top
        if len(win) < win.maxlen:
            return
        rolling_mean = sum(win) / len(win)
        if rolling_mean >= tier.unlock_threshold and self._tier < len(TIERS) - 1:
            transition = {
                "from_tier": self._tier,
                "to_tier": self._tier + 1,
                "from_name": TIERS[self._tier].name,
                "to_name": TIERS[self._tier + 1].name,
                "at_episode": self._episode_count,
                "rolling_mean_at_unlock": rolling_mean,
                "unlock_threshold": tier.unlock_threshold,
            }
            self._tier_transitions.append(transition)
            self._tier += 1
            # clear new tier's window so it starts fresh
            self._windows[self._tier].clear()

    def _curriculum_obs(self) -> Dict[str, Any]:
        tier = TIERS[self._tier]
        win = self._windows[self._tier]
        rolling_mean = (sum(win) / len(win)) if win else 0.0
        thr = tier.unlock_threshold
        if thr is None:
            progress = 1.0
        else:
            # progress to threshold from a baseline of 0.0
            progress = max(0.0, min(1.0, rolling_mean / thr)) if thr > 0 else 1.0
        return {
            "tier": self._tier,
            "tier_name": tier.name,
            "rolling_mean": float(rolling_mean),
            "threshold": thr,
            "progress": float(progress),
            "current_scenario": self._current_scenario,
        }

    # --------------------------------------------------------- public dump
    def curriculum_metrics(self) -> Dict[str, Any]:
        tier = TIERS[self._tier]
        win = self._windows[self._tier]
        rolling_mean = (sum(win) / len(win)) if win else 0.0
        thr = tier.unlock_threshold
        progress = 1.0 if thr is None else (
            max(0.0, min(1.0, rolling_mean / thr)) if thr > 0 else 1.0
        )
        return {
            "tier_level": self._tier,
            "tier_name": tier.name,
            "tier_description": tier.description,
            "available_scenarios": list(tier.scenarios),
            "episode_count": self._episode_count,
            "rolling_mean_reward": float(rolling_mean),
            "unlock_threshold": thr,
            "progress_to_next": float(progress),
            "window_size": len(win),
            "window_capacity": win.maxlen,
            "transitions": list(self._tier_transitions),
            "at_max_tier": self._tier >= len(TIERS) - 1,
            "adversarial_mode": bool(self._adversarial
                                     and self._tier >= len(TIERS) - 1),
            "adversarial_decoys": list(getattr(self, "_adv_decoys", []) or []),
        }

    def force_tier(self, tier: int) -> None:
        self._tier = max(0, min(tier, len(TIERS) - 1))

    # --------------------------------------------------------- passthrough
    @property
    def last_reward(self) -> float:
        return self._env.last_reward

    @property
    def tier(self) -> int:
        return self._tier

    @property
    def tier_name(self) -> str:
        return TIERS[self._tier].name

    @property
    def episode_count(self) -> int:
        return self._episode_count

    @property
    def transitions(self) -> List[Dict[str, Any]]:
        return list(self._tier_transitions)
