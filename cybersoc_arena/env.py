"""CyberSOCEnv: an OpenEnv-compliant Tier-2 SOC investigation environment.

Inherits from openenv.core.env_server.Environment so that create_fastapi_app
exposes the standard Gym-style HTTP surface (/reset, /step, /state, /health,
/docs, /web) plus a WebSocket session at /ws.

Internal mechanics (scenarios, evidence trails, dense reward shaping,
repeat-penalty bookkeeping) live in their own modules:

  * cybersoc_arena.scenarios   - procedural scenario generators
  * cybersoc_arena.observations - WorldState -> dict observation builder
  * cybersoc_arena.rewards      - dense + bounded reward shaping
  * cybersoc_arena.state        - mutable per-episode bookkeeping
  * cybersoc_arena.actions      - action parser + INVESTIGATIVE/TERMINAL sets
  * cybersoc_arena.models       - Pydantic Action/Observation/State

Episode ends when (a) the agent invokes any TERMINAL action, or (b) the step
budget is exhausted. Reward is dense (per-step shaping + a sharp terminal
signal) and bounded to [-2, 2] per step so a single bad action cannot poison
a batch.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, Optional

from cybersoc_arena.actions import (
    INVESTIGATIVE_ACTIONS,
    TERMINAL_ACTIONS,
    Action as _RawAction,
    parse_action,
)
from cybersoc_arena.models import (
    ActionHistoryEntry,
    AlertView,
    AssetInventory,
    CyberAction,
    CyberObservation,
    CyberState,
    EvidenceItem,
)
from cybersoc_arena.observations import build_observation
from cybersoc_arena.rewards import StepReward, investigative_reward, terminal_reward
from cybersoc_arena.scenarios import generate_scenario
from cybersoc_arena.state import WorldState

try:
    from openenv.core.env_server import Environment as _OpenEnvBase
except ImportError:  # pragma: no cover  --- openenv-core not installed
    _OpenEnvBase = object  # type: ignore[assignment,misc]


class CyberSOCEnv(_OpenEnvBase):
    """Realistic Tier-2 SOC analyst environment for LLM agents.

    Parameters
    ----------
    scenario_type : str, optional
        Force every episode to use this scenario archetype. Default: random.
        Valid values: see cybersoc_arena.scenarios.SCENARIO_TYPES.
    seed : int, optional
        Default RNG seed. Per-episode override available via reset(seed=...).
    transform : openenv.core.Transform, optional
        Optional observation transform (passed through to the OpenEnv base).
    rubric : openenv.core.Rubric, optional
        Optional reward rubric (passed through to the OpenEnv base). When
        None (default) the dense in-house reward shaper in
        cybersoc_arena.rewards is used.

    Methods (Gym-style, OpenEnv-conformant)
    ---------------------------------------
    reset(seed=None, episode_id=None, scenario_type=None) -> CyberObservation
    step(action: CyberAction) -> CyberObservation
    state (property) -> CyberState
    """

    # Allow many concurrent WebSocket sessions per server instance. Multi-step
    # episodes use the /ws endpoint (or the CyberSOCAsyncClient subclass of
    # openenv.core.EnvClient). Stateless HTTP /reset+/step is for one-shot
    # evaluations only -- the OpenEnv server creates and closes a fresh env
    # per HTTP request, so to drive a full episode prefer the WebSocket path.
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(
        self,
        scenario_type: Optional[str] = None,
        seed: Optional[int] = None,
        transform: Any = None,
        rubric: Any = None,
    ):
        if _OpenEnvBase is not object:
            super().__init__(transform=transform, rubric=rubric)
        self._default_scenario_type = scenario_type
        self._default_seed = seed
        self._world: Optional[WorldState] = None
        self._episode_id: Optional[str] = None
        self.last_reward: Optional[StepReward] = None
        self._curriculum_tier: Optional[int] = None
        self._curriculum_tier_name: Optional[str] = None

    # ------------------------------------------------------------------ Gym API
    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        scenario_type: Optional[str] = None,
        **kwargs: Any,
    ) -> CyberObservation:
        """Begin a new SOC investigation episode."""
        st = scenario_type if scenario_type is not None else self._default_scenario_type
        sd = seed if seed is not None else self._default_seed
        scenario = generate_scenario(scenario_type=st, seed=sd)
        self._world = WorldState(scenario=scenario)
        self._episode_id = episode_id or f"ep-{uuid.uuid4().hex[:12]}"
        self.last_reward = None
        return self._build_obs(reward=None, info={"episode_id": self._episode_id})

    def step(
        self,
        action: CyberAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> CyberObservation:
        """Execute one SOC tool invocation and return the next observation."""
        if self._world is None:
            raise RuntimeError("Call reset() before step().")
        if self._world.done:
            raise RuntimeError("Episode is already done. Call reset().")

        # Coerce CyberAction -> internal dataclass via the tolerant parser.
        try:
            if isinstance(action, CyberAction):
                act = parse_action(action.model_dump(exclude_none=True))
            else:
                act = parse_action(action)
        except ValueError as e:
            self._world.step += 1
            penalty = StepReward(value=-0.1, breakdown={"malformed_action": -0.1})
            self.last_reward = penalty
            self._world.action_history.append(("malformed", "", False))
            done = self._budget_exhausted()
            if done:
                final = self._end_with_budget_exhaustion(penalty)
                return self._build_obs(reward=final, info={"error": str(e)})
            return self._build_obs(
                reward=penalty,
                info={"error": str(e), "breakdown": penalty.breakdown},
            )

        self._world.step += 1
        target_str = act.ip or act.host or act.entity or ""
        self._world.action_history.append((act.action_type, target_str, True))

        # Investigative branch
        if act.action_type in INVESTIGATIVE_ACTIONS:
            reward = self._handle_investigative(act)
            done = self._budget_exhausted()
            if done:
                term = terminal_reward(None, self._world)
                combined_value = reward.value + term.value
                breakdown = {**reward.breakdown, **term.breakdown}
                self._world.done = True
                self._world.terminal_action = "budget_exhausted"
                self._world.terminal_correct = False
                final_reward = StepReward(value=combined_value, breakdown=breakdown)
                self.last_reward = final_reward
                return self._build_obs(
                    reward=final_reward,
                    info={"breakdown": breakdown, "terminal": "budget_exhausted"},
                )
            self.last_reward = reward
            return self._build_obs(
                reward=reward, info={"breakdown": reward.breakdown}
            )

        # Terminal branch
        if act.action_type in TERMINAL_ACTIONS:
            reward = terminal_reward(act, self._world)
            self._world.done = True
            self._world.terminal_action = act.action_type
            self._world.terminal_correct = self._was_correct(act)
            if act.action_type in ("escalate_incident", "close_as_benign"):
                self._world.final_summary = act.summary
            self.last_reward = reward
            return self._build_obs(
                reward=reward,
                info={
                    "breakdown": reward.breakdown,
                    "terminal": act.action_type,
                    "correct": self._world.terminal_correct,
                },
            )

        raise ValueError(f"Unhandled action_type: {act.action_type}")

    @property
    def state(self) -> CyberState:
        """Snapshot of episode metadata. Never leaks the hidden attacker IP."""
        if self._world is None:
            return CyberState(
                episode_id="(none)",
                scenario_type="(none)",
                is_benign=False,
                step=0,
                step_budget=0,
                remaining_steps=0,
                done=False,
                evidence_count=0,
                attacker_evidence_count=0,
                curriculum_tier=self._curriculum_tier,
                curriculum_tier_name=self._curriculum_tier_name,
            )
        sc = self._world.scenario
        return CyberState(
            episode_id=self._episode_id or "(unknown)",
            scenario_type=sc.scenario_type,
            is_benign=sc.is_benign,
            step=self._world.step,
            step_budget=sc.step_budget,
            remaining_steps=self._world.remaining_steps,
            done=self._world.done,
            terminal_action=self._world.terminal_action,
            terminal_correct=self._world.terminal_correct,
            evidence_count=len(self._world.revealed_evidence),
            attacker_evidence_count=self._world.attacker_evidence_collected,
            final_summary=self._world.final_summary,
            curriculum_tier=self._curriculum_tier,
            curriculum_tier_name=self._curriculum_tier_name,
        )

    # ------------------------------------------------------------- Convenience
    def get_world_state(self) -> Optional[WorldState]:
        """Internal-only accessor (used by CurriculumEnv + tests)."""
        return self._world

    def set_curriculum_tag(self, tier: Optional[int], tier_name: Optional[str]) -> None:
        """Used by CurriculumEnv to label the current episode's tier."""
        self._curriculum_tier = tier
        self._curriculum_tier_name = tier_name

    # ------------------------------------------------------------------ Internals
    def _handle_investigative(self, act: _RawAction) -> StepReward:
        target = act.ip or act.host or act.entity or ""
        new_idxs = self._world.evidence_for_action(act.action_type, target)
        for i in new_idxs:
            self._world.revealed_evidence.append(i)
        if act.ip:
            target_set = (
                self._world.investigated_ips
                if act.action_type == "investigate_ip"
                else self._world.queried_ips
                if act.action_type == "query_logs"
                else self._world.threat_intel_ips
                if act.action_type == "check_threat_intel"
                else self._world.correlated_entities
            )
            target_set.add(act.ip)
        elif act.host:
            self._world.inspected_hosts.add(act.host)
        elif act.entity:
            self._world.correlated_entities.add(act.entity)
        return investigative_reward(act, self._world, new_idxs)

    def _budget_exhausted(self) -> bool:
        return self._world.step >= self._world.scenario.step_budget

    def _end_with_budget_exhaustion(self, base: StepReward) -> StepReward:
        term = terminal_reward(None, self._world)
        combined = base.value + term.value
        breakdown = {**base.breakdown, **term.breakdown}
        self._world.done = True
        self._world.terminal_action = "budget_exhausted"
        self._world.terminal_correct = False
        final = StepReward(value=combined, breakdown=breakdown)
        self.last_reward = final
        return final

    def _was_correct(self, act: _RawAction) -> bool:
        sc = self._world.scenario
        if act.action_type == "identify_attacker":
            return (not sc.is_benign) and act.ip == sc.attacker_ip
        if act.action_type == "close_as_benign":
            return sc.is_benign
        if act.action_type == "isolate_host":
            return (not sc.is_benign) and act.host in (sc.target_hosts or [])
        if act.action_type == "escalate_incident":
            summary = (act.summary or "").lower()
            return (
                any(str(kw).lower() in summary for kw in sc.summary_keywords)
                and not sc.is_benign
            )
        return False

    def _build_obs(
        self, reward: Optional[StepReward], info: Dict[str, Any]
    ) -> CyberObservation:
        """Render WorldState -> CyberObservation Pydantic model."""
        raw = build_observation(self._world)
        evidence_items = [
            EvidenceItem(
                step=int(e["step"]),
                action=str(e["action"]),
                target=str(e["target"]),
                finding=str(e["finding"]),
            )
            for e in raw.get("evidence_collected", [])
        ]
        action_items = [
            ActionHistoryEntry(
                action_type=str(h["action_type"]),
                target=str(h["target"]),
                success=bool(h["success"]),
            )
            for h in raw.get("action_history", [])
        ]
        full_info: Dict[str, Any] = dict(info)
        if reward is not None:
            full_info["reward_breakdown"] = reward.breakdown
        return CyberObservation(
            done=bool(self._world.done),
            reward=float(reward.value) if reward is not None else None,
            alert=AlertView(
                summary=raw["alert"]["summary"],
                severity=raw["alert"]["severity"],
            ),
            step=raw["step"],
            remaining_steps=raw["remaining_steps"],
            step_budget=raw["step_budget"],
            asset_inventory=AssetInventory(
                hosts=raw["asset_inventory"]["hosts"],
                visible_ips=raw["asset_inventory"]["visible_ips"],
            ),
            evidence_collected=evidence_items,
            evidence_count=raw["evidence_count"],
            action_history=action_items,
            noise_sample=list(raw.get("noise_sample", [])),
            available_actions=list(raw.get("available_actions", [])),
            goal=raw.get("goal", ""),
            info=full_info,
        )
