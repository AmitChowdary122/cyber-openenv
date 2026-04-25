"""CurriculumEnv: an adaptive-difficulty wrapper for CyberSOC Arena.

Implements OpenEnv **Theme 4 (Self-Improvement)**: the wrapped environment
keeps a rolling mean of recent episode rewards and *unlocks* harder scenario
archetypes once the agent demonstrates mastery of easier ones. The agent
therefore drives its own capability growth — exactly the "recursive skill
amplification" framing the hackathon brief calls out.

Tier ladder
-----------
Each tier has a **scenario pool** (which archetypes can be sampled) and a
**reward threshold** (rolling mean over the last N episodes). When the agent
clears a threshold, the next tier unlocks; if performance regresses below the
previous threshold for a sustained window, the curriculum can drop a tier
back. A monotone ratchet (``ratchet=True``) prevents drops once mastery is
demonstrated — useful for benchmarking final policies.

Tier 0 — *Novice analyst*           pool: ``benign_scan``                                              threshold: +0.50
Tier 1 — *Junior responder*         pool: + ``phishing_lateral``                                      threshold: +0.70
Tier 2 — *Mid-level analyst*        pool: + ``credential_stuffing``                                   threshold: +0.85
Tier 3 — *Senior responder*         pool: + ``data_exfiltration``                                     threshold: +0.95
Tier 4 — *Incident lead*            pool: + ``multi_stage_chain``                                     threshold: +1.05
Tier 5 — *APT hunter*               pool: + ``long_horizon_apt`` (full mix, including 20-step APT)   threshold: --

Usage
-----
::

    from cybersoc_arena import CurriculumEnv

    env = CurriculumEnv(window=20)
    for episode in range(500):
        obs = env.reset()
        done = False
        while not done:
            action = my_policy(obs)
            obs = env.step(action)
            done = obs.done
        env.record_episode_reward(obs.reward)
        if env.tier_changed:
            print(f"Episode {episode}: unlocked tier {env.tier} "
                  f"({env.tier_name})")
"""

from __future__ import annotations

import dataclasses
import random
from collections import deque
from typing import Any, Deque, Dict, List, Optional

from cybersoc_arena.env import CyberSOCEnv
from cybersoc_arena.models import CyberAction, CyberObservation, CyberState


@dataclasses.dataclass(frozen=True)
class Tier:
    """One step on the curriculum ladder."""

    index: int
    name: str
    scenario_pool: List[str]
    advance_threshold: float
    description: str


# Ordered easiest -> hardest. The pool of each tier *includes* all previous.
TIERS: List[Tier] = [
    Tier(
        index=0,
        name="Novice analyst",
        scenario_pool=["benign_scan"],
        advance_threshold=0.50,
        description="Distinguish benign internet noise from real incidents.",
    ),
    Tier(
        index=1,
        name="Junior responder",
        scenario_pool=["benign_scan", "phishing_lateral"],
        advance_threshold=0.70,
        description="Add user-targeted phishing with lateral movement.",
    ),
    Tier(
        index=2,
        name="Mid-level analyst",
        scenario_pool=[
            "benign_scan", "phishing_lateral", "credential_stuffing",
        ],
        advance_threshold=0.85,
        description="Pick the real attacker IP out of a flood of failed logins.",
    ),
    Tier(
        index=3,
        name="Senior responder",
        scenario_pool=[
            "benign_scan", "phishing_lateral", "credential_stuffing",
            "data_exfiltration",
        ],
        advance_threshold=0.95,
        description="Catch slow, low-volume covert egress hidden in TLS noise.",
    ),
    Tier(
        index=4,
        name="Incident lead",
        scenario_pool=[
            "benign_scan", "phishing_lateral", "credential_stuffing",
            "data_exfiltration", "multi_stage_chain",
        ],
        advance_threshold=1.05,
        description="Full short kill chain: recon -> exploit -> persist -> exfil.",
    ),
    Tier(
        index=5,
        name="APT hunter",
        scenario_pool=[
            "benign_scan", "phishing_lateral", "credential_stuffing",
            "data_exfiltration", "multi_stage_chain", "long_horizon_apt",
        ],
        advance_threshold=float("inf"),  # terminal tier
        description="20-step long-horizon APT across 5 hosts, with decoys.",
    ),
]


class CurriculumEnv:
    """Adaptive-difficulty wrapper around :class:`CyberSOCEnv`.

    Acts like a regular OpenEnv environment to the outside world (``reset()``
    / ``step()`` / ``state``) but, between episodes, samples the next
    scenario from the currently unlocked tier's pool.

    Parameters
    ----------
    window : int
        How many recent episode rewards to roll over for the threshold check.
        Default 10. Larger window = slower, more confident promotions.
    promote_after : int
        Minimum number of episodes at the current tier before a promotion
        becomes possible. Avoids flapping right after a threshold change.
    ratchet : bool
        If True (default) a tier never drops back. Useful for clean training
        curves. Set False to allow demotion on sustained regressions.
    seed : int, optional
        Master seed for the inter-episode scenario sampler.
    """

    def __init__(
        self,
        window: int = 10,
        promote_after: int = 5,
        ratchet: bool = True,
        seed: Optional[int] = None,
    ):
        self._env = CyberSOCEnv()
        self._window = window
        self._promote_after = promote_after
        self._ratchet = ratchet
        self._rng = random.Random(seed)

        self._tier_idx: int = 0
        self._episodes_at_tier: int = 0
        self._recent_rewards: Deque[float] = deque(maxlen=window)
        self._tier_history: List[Dict[str, Any]] = []
        self._last_tier_change: Optional[int] = None
        self._episode_count: int = 0

    # ─── Curriculum metadata accessors ─────────────────────────────────────
    @property
    def tier(self) -> int:
        return self._tier_idx

    @property
    def tier_name(self) -> str:
        return TIERS[self._tier_idx].name

    @property
    def tier_pool(self) -> List[str]:
        return list(TIERS[self._tier_idx].scenario_pool)

    @property
    def rolling_mean(self) -> float:
        if not self._recent_rewards:
            return 0.0
        return sum(self._recent_rewards) / len(self._recent_rewards)

    @property
    def tier_changed(self) -> bool:
        """True if the most recent record_episode_reward() flipped a tier."""
        return self._last_tier_change == self._episode_count

    @property
    def history(self) -> List[Dict[str, Any]]:
        return list(self._tier_history)

    # ─── Gym-style API (delegates to the inner env) ────────────────────────
    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        scenario_type: Optional[str] = None,
        **kwargs: Any,
    ) -> CyberObservation:
        """Begin an episode using a scenario sampled from the current tier."""
        if scenario_type is None:
            scenario_type = self._rng.choice(self.tier_pool)
        self._env.set_curriculum_tag(self._tier_idx, self.tier_name)
        return self._env.reset(
            seed=seed, episode_id=episode_id, scenario_type=scenario_type, **kwargs
        )

    def step(
        self,
        action: CyberAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> CyberObservation:
        return self._env.step(action, timeout_s=timeout_s, **kwargs)

    @property
    def state(self) -> CyberState:
        return self._env.state

    # ─── Curriculum bookkeeping ────────────────────────────────────────────
    def record_episode_reward(self, episode_reward: float) -> None:
        """Append the just-finished episode's *cumulative* reward.

        Call this once per episode (after ``done=True``). It updates the
        rolling mean, may unlock the next tier, and writes a row into
        ``self.history``.
        """
        self._episode_count += 1
        self._episodes_at_tier += 1
        self._recent_rewards.append(float(episode_reward))
        prev_tier = self._tier_idx

        # Promotion check
        if (
            self._tier_idx < len(TIERS) - 1
            and self._episodes_at_tier >= self._promote_after
            and len(self._recent_rewards) >= max(3, self._window // 2)
            and self.rolling_mean >= TIERS[self._tier_idx].advance_threshold
        ):
            self._tier_idx += 1
            self._episodes_at_tier = 0
            self._recent_rewards.clear()
            self._last_tier_change = self._episode_count

        # Demotion check (only if not ratcheted)
        elif (
            not self._ratchet
            and self._tier_idx > 0
            and self._episodes_at_tier >= self._promote_after
            and len(self._recent_rewards) >= max(3, self._window // 2)
            and self.rolling_mean < TIERS[self._tier_idx - 1].advance_threshold * 0.6
        ):
            self._tier_idx -= 1
            self._episodes_at_tier = 0
            self._recent_rewards.clear()
            self._last_tier_change = self._episode_count

        self._tier_history.append(
            {
                "episode": self._episode_count,
                "tier": self._tier_idx,
                "tier_name": self.tier_name,
                "rolling_mean": self.rolling_mean,
                "promoted": self._tier_idx > prev_tier,
                "demoted": self._tier_idx < prev_tier,
                "episode_reward": float(episode_reward),
            }
        )


__all__ = ["CurriculumEnv", "Tier", "TIERS"]
