"""
OpenEnv server wrapper around :class:`cybersoc_arena.CurriculumEnv`.

The underlying CurriculumEnv is a gym-style env returning
``(obs_dict, reward, done, info)``. This wrapper re-exposes it as a
strict :class:`openenv.core.Environment` returning
:class:`models.CyberSOCObservation` (which carries ``done`` / ``reward``
inline) so that the openenv HTTP / WebSocket bridge can serialise it.
"""
from uuid import uuid4
from typing import Any, Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from cybersoc_arena import CurriculumEnv

try:
    from ..models import CyberSOCAction, CyberSOCObservation
except ImportError:  # when imported as `from models import ...`
    from models import CyberSOCAction, CyberSOCObservation  # type: ignore


class CyberSOCEnvironment(Environment):
    """OpenEnv-compatible front-end for CyberSOC Arena."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, seed: Optional[int] = None, start_tier: int = 0):
        super().__init__()
        self._seed = seed
        self._start_tier = start_tier
        self._env = CurriculumEnv(seed=seed, start_tier=start_tier)
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._last_obs_dict: dict = {}
        self._cum_reward: float = 0.0

    # ------------------------------------------------------------------ API
    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> CyberSOCObservation:
        if seed is not None:
            self._seed = seed
            self._env = CurriculumEnv(seed=seed, start_tier=self._start_tier)
        obs = self._env.reset(seed=seed, episode_id=episode_id)
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        self._last_obs_dict = obs
        self._cum_reward = 0.0
        return self._to_obs(obs, reward=0.0, done=False)

    def step(
        self,
        action: CyberSOCAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> CyberSOCObservation:  # type: ignore[override]
        a = {
            "action_type": action.action_type,
            "ip": action.ip,
            "host": action.host,
            "target": action.target,
        }
        obs_dict, reward, done, info = self._env.step(a)
        self._state.step_count += 1
        self._last_obs_dict = obs_dict
        self._cum_reward += reward
        return self._to_obs(obs_dict, reward=reward, done=done, extra=info)

    @property
    def state(self) -> State:
        return self._state

    # ---------------------------------------------------------- helpers
    def _to_obs(self, d: dict, reward: float = 0.0, done: bool = False,
                extra: Optional[dict] = None) -> CyberSOCObservation:
        meta = {
            "cum_reward": self._cum_reward,
            "tier": self._env.tier,
            "tier_name": self._env.tier_name,
            "episode_count": self._env.episode_count,
        }
        if extra:
            meta.update({k: v for k, v in extra.items()
                         if k.startswith("curriculum_") or k in ("verdict", "correct",
                                                                 "evidence_collected",
                                                                 "confirming_evidence")})
        return CyberSOCObservation(
            done=done,
            reward=float(reward),
            metadata=meta,
            scenario_type=d.get("scenario_type", ""),
            initial_alert=d.get("initial_alert", ""),
            alert_severity=d.get("alert_severity", ""),
            step=int(d.get("step", 0)),
            step_budget=int(d.get("step_budget", 0)),
            revealed_evidence=list(d.get("revealed_evidence", [])),
            evidence_count=int(d.get("evidence_count", 0)),
            min_evidence_for_verdict=int(d.get("min_evidence_for_verdict", 3)),
            known_entities=dict(d.get("known_entities", {"ips": [], "hosts": []})),
            action_history=list(d.get("action_history", [])),
            last_action_result=d.get("last_action_result", ""),
            available_actions=list(d.get("available_actions", [])),
            is_terminal=bool(d.get("is_terminal", False)),
            curriculum=dict(d.get("curriculum", {})),
        )
