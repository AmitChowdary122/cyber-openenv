"""
CyberSOCEnv — the SOC Arena core environment.

This class subclasses :class:`openenv.core.Environment` (when available) so it
plugs into the OpenEnv ecosystem, but it also exposes the simple
gym-style tuple API ``(obs, reward, done, info) = env.step(action)`` so it
can be driven directly from Python without going through the JSON-RPC
server.

Observation (dict, OpenEnv-friendly):
  - scenario_type: str
  - initial_alert: str
  - alert_severity: str
  - step: int
  - step_budget: int
  - revealed_evidence: List[str]
  - known_entities: {"ips": [...], "hosts": [...]}
  - action_history: List[{action_type, target, ...}]   (last 5)
  - last_action_result: str
  - is_terminal: bool

Action: a cybersoc_arena.actions.Action (or any dict / JSON parseable
into one).

Reward:
  - step penalty: -0.02 every step
  - reveal new attacker evidence:    +0.10 * weight
  - reveal new decoy evidence:       +0.04 * weight  (still useful — rules out a lead)
  - duplicate investigative call:    -0.05
  - terminal action with < 3 confirming evidence pieces:  premature_decision -0.30
  - correct identify_attacker:       +1.00
  - wrong identify_attacker:         -0.50
  - correct block_ip:                +0.60
  - wrong block_ip (not attacker):   -0.50
  - escalate_incident on real attack:+0.40
  - escalate_incident on benign:     -0.30
  - mark_benign on benign:           +1.00
  - mark_benign on real attack:      -0.80
  - exceeded budget without verdict: -0.20
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import random

from .actions import (
    Action, parse_action,
    INVESTIGATIVE_ACTIONS, TERMINAL_ACTIONS, ALL_ACTIONS,
    ACTION_DESCRIPTIONS,
)
from .scenarios import (
    Scenario, Evidence, generate_scenario, SCENARIO_TYPES,
)

# Optional OpenEnv base class. We inherit from it so the environment is a
# first-class OpenEnv environment, but fall back gracefully if openenv is
# not installed (e.g. on minimal CPU rollout machines).
try:  # pragma: no cover
    from openenv.core import Environment as _OpenEnvBase  # type: ignore
    _HAS_OPENENV = True
except Exception:  # pragma: no cover
    _OpenEnvBase = object  # type: ignore
    _HAS_OPENENV = False

try:  # pragma: no cover
    from openenv.core import EnvironmentMetadata as _OpenEnvMetadata  # type: ignore
except Exception:  # pragma: no cover
    try:
        from openenv.core.env_server.interfaces import EnvironmentMetadata as _OpenEnvMetadata  # type: ignore
    except Exception:
        _OpenEnvMetadata = None  # type: ignore


# Reward constants (kept module-level so they can be tweaked / inspected)
STEP_PENALTY            = -0.02
REVEAL_ATTACKER_BONUS   =  0.10   # multiplied by evidence.weight
REVEAL_DECOY_BONUS      =  0.04   # multiplied by evidence.weight (still useful)
DUPLICATE_PENALTY       = -0.05
PREMATURE_DECISION_PEN  = -0.30
MIN_EVIDENCE_FOR_VERDICT = 3       # confirming evidence pieces

REWARD_IDENTIFY_OK      =  1.00
REWARD_IDENTIFY_WRONG   = -0.50
REWARD_BLOCK_OK         =  0.60
REWARD_BLOCK_WRONG      = -0.50
REWARD_ESCALATE_ATTACK  =  0.40
REWARD_ESCALATE_BENIGN  = -0.30
REWARD_BENIGN_OK        =  1.00
REWARD_BENIGN_WRONG     = -0.80
REWARD_BUDGET_EXCEEDED  = -0.20


class CyberSOCEnv(_OpenEnvBase):
    """
    A turn-based SOC investigation environment.

    Inherits from :class:`openenv.core.Environment` when OpenEnv is
    installed (so it satisfies the hackathon judging requirement that
    environments use the OpenEnv base class) and falls back to a plain
    ``object`` base otherwise. The agent acts; the environment reveals
    evidence and reward.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, scenario_type: Optional[str] = None,
                 seed: Optional[int] = None):
        if _HAS_OPENENV:
            super().__init__()
        self._initial_scenario_type = scenario_type
        self._master_seed = seed
        self._rng = random.Random(seed)

        self.scenario: Optional[Scenario] = None
        self._evidence_idx: Dict[Tuple[str, str], List[Evidence]] = {}
        self._revealed_keys: set = set()
        self._revealed_texts: List[str] = []
        self._revealed_evidence_objs: List[Evidence] = []
        self._duplicate_calls: set = set()
        self._action_history: List[Dict[str, Any]] = []
        self._last_action_result: str = ""

        self._step: int = 0
        self._done: bool = False
        self._verdict: Optional[Dict[str, Any]] = None
        self._last_reward: float = 0.0
        self._cum_reward: float = 0.0

        self.reset(scenario_type=scenario_type, seed=seed)

    # -------------------------------------------------------------- API
    def reset(self, scenario_type: Optional[str] = None,
              seed: Optional[int] = None,
              episode_id: Optional[str] = None,
              **kwargs: Any) -> Dict[str, Any]:
        if seed is not None:
            self._master_seed = seed
            self._rng = random.Random(seed)
        self._episode_id = episode_id

        st = scenario_type if scenario_type is not None else self._initial_scenario_type
        ep_seed = self._rng.randint(0, 1 << 30)
        self.scenario = generate_scenario(st, seed=ep_seed)
        self._evidence_idx = self.scenario.evidence_index()
        self._revealed_keys = set()
        self._revealed_texts = []
        self._revealed_evidence_objs = []
        self._duplicate_calls = set()
        self._action_history = []
        self._last_action_result = (
            f"NEW ALERT [{self.scenario.alert_severity.upper()}]: "
            f"{self.scenario.initial_alert}"
        )
        self._step = 0
        self._done = False
        self._verdict = None
        self._last_reward = 0.0
        self._cum_reward = 0.0
        return self._observation()

    def step(self, raw_action: Any, timeout_s: Optional[float] = None,
             **kwargs: Any) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if self._done:
            return self._observation(), 0.0, True, self._info()

        action = parse_action(raw_action)
        self._step += 1
        reward = STEP_PENALTY

        if action.is_investigative:
            reward += self._handle_investigative(action)
        elif action.is_terminal:
            reward += self._handle_terminal(action)
        else:
            reward += -0.10
            self._last_action_result = f"unknown action_type={action.action_type}"

        # log action
        self._action_history.append({
            "step": self._step,
            "action_type": action.action_type,
            "ip": action.ip,
            "host": action.host,
            "target": action.target,
        })

        # budget enforcement
        if not self._done and self._step >= self.scenario.step_budget:
            if self._verdict is None:
                reward += REWARD_BUDGET_EXCEEDED
                self._last_action_result += " | step budget exhausted with no verdict."
            self._done = True

        self._last_reward = reward
        self._cum_reward += reward
        return self._observation(), reward, self._done, self._info()

    @property
    def state(self) -> Dict[str, Any]:
        return {
            "scenario_type": self.scenario.scenario_type,
            "attacker_ip": self.scenario.attacker_ip,  # ground truth (grader only)
            "c2_ip": self.scenario.c2_ip,
            "is_benign": self.scenario.is_benign,
            "step": self._step,
            "step_budget": self.scenario.step_budget,
            "revealed_count": len(self._revealed_evidence_objs),
            "verdict": self._verdict,
            "cum_reward": self._cum_reward,
            "done": self._done,
            "episode_id": getattr(self, "_episode_id", None),
            "step_count": self._step,
        }

    def get_metadata(self):
        info = {
            "name": "CyberSOCEnv",
            "description": "Long-horizon SOC analyst training environment "
                           "with 9 tools, 6 scenarios up to a 20-step APT.",
            "version": "0.2.0",
        }
        if _OpenEnvMetadata is not None:
            try:
                return _OpenEnvMetadata(**info)
            except Exception:
                pass
        return info

    # ---------------------------------------------------------- helpers
    def _handle_investigative(self, action: Action) -> float:
        target = (action.ip or action.host or action.target or "").strip()
        key = (action.action_type, target)

        if action.action_type == "list_entities":
            ips = self._known_ips()
            hosts = list(self.scenario.target_hosts)
            self._last_action_result = (
                f"Known entities — IPs: {ips} | Hosts: {hosts}"
            )
            return 0.0

        if not target:
            self._last_action_result = (
                f"{action.action_type} called without a target — no-op."
            )
            return -0.05

        # Duplicate call?
        if key in self._duplicate_calls:
            self._last_action_result = (
                f"{action.action_type}({target}) — already queried, no new info."
            )
            return DUPLICATE_PENALTY
        self._duplicate_calls.add(key)

        evs = self._evidence_idx.get(key, [])
        if not evs:
            self._last_action_result = (
                f"{action.action_type}({target}) — nothing of interest."
            )
            return 0.0

        gained = 0.0
        new_texts = []
        for e in evs:
            ek = (e.action_type, e.target, e.text)
            if ek in self._revealed_keys:
                continue
            self._revealed_keys.add(ek)
            self._revealed_texts.append(e.text)
            self._revealed_evidence_objs.append(e)
            new_texts.append(e.text)
            if e.confirms_attacker:
                gained += REVEAL_ATTACKER_BONUS * e.weight
            else:
                gained += REVEAL_DECOY_BONUS * e.weight

        if new_texts:
            self._last_action_result = (
                f"{action.action_type}({target}) revealed: " + " | ".join(new_texts)
            )
        else:
            self._last_action_result = (
                f"{action.action_type}({target}) — already-seen finding."
            )
        return gained

    def _handle_terminal(self, action: Action) -> float:
        confirming = sum(1 for e in self._revealed_evidence_objs if e.confirms_attacker)
        premature = confirming < MIN_EVIDENCE_FOR_VERDICT
        pre_pen = PREMATURE_DECISION_PEN if premature else 0.0

        sc = self.scenario
        attacker_ip = sc.attacker_ip
        target_ip = action.ip or action.target

        verdict = {"action_type": action.action_type, "target": target_ip,
                   "premature": premature, "evidence_count": confirming}

        reward = pre_pen

        valid_attribution = {ip for ip in (attacker_ip, sc.c2_ip) if ip}

        if action.action_type == "identify_attacker":
            ok = (target_ip is not None and target_ip in valid_attribution)
            reward += REWARD_IDENTIFY_OK if ok else REWARD_IDENTIFY_WRONG
            # bonus for nailing the human-at-keyboard attacker_ip exactly
            if ok and target_ip == attacker_ip and sc.c2_ip:
                reward += 0.1
            verdict["correct"] = ok
            self._last_action_result = (
                f"identify_attacker({target_ip}): "
                + ("CORRECT" if ok else f"WRONG (truth={attacker_ip})")
                + (" [premature]" if premature else "")
            )

        elif action.action_type == "block_ip":
            ok = (target_ip is not None and target_ip in valid_attribution)
            reward += REWARD_BLOCK_OK if ok else REWARD_BLOCK_WRONG
            verdict["correct"] = ok
            self._last_action_result = (
                f"block_ip({target_ip}): "
                + ("BLOCKED ATTACKER" if ok else "WRONG IP BLOCKED")
                + (" [premature]" if premature else "")
            )

        elif action.action_type == "escalate_incident":
            ok = not sc.is_benign
            reward += REWARD_ESCALATE_ATTACK if ok else REWARD_ESCALATE_BENIGN
            verdict["correct"] = ok
            self._last_action_result = (
                "escalate_incident: "
                + ("escalated real incident — IR team paged" if ok else "false-alarm escalation")
                + (" [premature]" if premature else "")
            )

        elif action.action_type == "mark_benign":
            ok = sc.is_benign
            reward += REWARD_BENIGN_OK if ok else REWARD_BENIGN_WRONG
            verdict["correct"] = ok
            self._last_action_result = (
                "mark_benign: "
                + ("correctly closed false positive" if ok else "MISSED A REAL ATTACK")
                + (" [premature]" if premature else "")
            )

        self._verdict = verdict
        self._done = True
        return reward

    # ---------------------------------------------------------- observation
    @staticmethod
    def _looks_like_ip(s: str) -> bool:
        if not s or "." not in s:
            return False
        parts = s.split(".")
        if len(parts) != 4:
            return False
        for p in parts:
            if not p.isdigit():
                return False
        return True

    def _known_ips(self) -> List[str]:
        ips: List[str] = []
        # 1) every IP that has evidence attached (so the agent can query it)
        for (atype, target), _evs in self._evidence_idx.items():
            if self._looks_like_ip(target):
                ips.append(target)
        # 2) explicit attacker / c2 / decoys (covers any not in evidence)
        if self.scenario.attacker_ip:
            ips.append(self.scenario.attacker_ip)
        if self.scenario.c2_ip:
            ips.append(self.scenario.c2_ip)
        ips.extend(self.scenario.decoy_ips or [])
        # de-dup, preserve order
        seen = set()
        out: List[str] = []
        for ip in ips:
            if ip and ip not in seen:
                seen.add(ip); out.append(ip)
        return out

    def _observation(self) -> Dict[str, Any]:
        sc = self.scenario
        confirming = sum(1 for e in self._revealed_evidence_objs if e.confirms_attacker)
        return {
            "scenario_type": sc.scenario_type,
            "initial_alert": sc.initial_alert,
            "alert_severity": sc.alert_severity,
            "step": self._step,
            "step_budget": sc.step_budget,
            "revealed_evidence": list(self._revealed_texts),
            "evidence_count": len(self._revealed_evidence_objs),
            "min_evidence_for_verdict": MIN_EVIDENCE_FOR_VERDICT,
            "known_entities": {
                "ips": self._known_ips(),
                "hosts": list(sc.target_hosts),
            },
            "action_history": list(self._action_history[-5:]),
            "last_action_result": self._last_action_result,
            "available_actions": list(ALL_ACTIONS),
            "is_terminal": self._done,
        }

    def _info(self) -> Dict[str, Any]:
        confirming = sum(1 for e in self._revealed_evidence_objs if e.confirms_attacker)
        return {
            "step": self._step,
            "scenario_type": self.scenario.scenario_type,
            "verdict": self._verdict,
            "evidence_collected": len(self._revealed_evidence_objs),
            "confirming_evidence": confirming,
            "cum_reward": self._cum_reward,
            "correct": (self._verdict or {}).get("correct", False) if self._done else None,
        }

    # ---------------------------------------------------------- conveniences
    @property
    def last_reward(self) -> float:
        return self._last_reward

    @property
    def cum_reward(self) -> float:
        return self._cum_reward

    @property
    def done(self) -> bool:
        return self._done
