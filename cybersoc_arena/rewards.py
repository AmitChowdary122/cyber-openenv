"""Reward computation for CyberSOC Arena.

Design goals
------------
1. Dense intermediate signal: every investigation step yields evidence-based
   reward, so an LLM gets useful gradient long before the terminal decision.
2. Hard to game: there are decoys with non-zero evidence weight. Spamming
   investigations does not maximise reward — the step penalty + repeated-action
   penalty dominate.
3. Strong terminal signal: terminal actions carry the bulk of reward and clearly
   distinguish correct attribution from false positives.
4. Sanity-bounded: per-step rewards are clipped to [-2, 2] so a single mistake
   cannot poison the whole batch.

Returned by step():
    StepReward(value=float, breakdown=dict)
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, Optional

from cybersoc_arena.actions import (
    INVESTIGATIVE_ACTIONS,
    TERMINAL_ACTIONS,
    Action,
)
from cybersoc_arena.state import WorldState

# ── Reward magnitudes (tuned for stable LLM-scale RL) ─────────────────────────
STEP_PENALTY              = -0.05
REPEAT_PENALTY            = -0.10
NEW_EVIDENCE_REWARD       =  0.20
DECOY_EVIDENCE_REWARD     =  0.05    # still positive — exploration is good
CORRELATION_BONUS         =  0.20
EVIDENCE_QUALITY_BONUS    =  0.30    # awarded with the terminal reward
PREMATURE_DECISION_PENALTY = -0.30   # terminal action with <2 evidence
OVER_BUDGET_PENALTY       = -1.00    # ran out of steps without deciding

# Terminal magnitudes
CORRECT_ATTACKER_ID       =  1.50
WRONG_ATTACKER_ID         = -1.50
CORRECT_BENIGN_CLOSE      =  1.20
WRONG_BENIGN_CLOSE        = -1.50    # the worst — closing a real incident
CORRECT_ISOLATE           =  0.80
WRONG_ISOLATE             = -1.00
ESCALATE_BASE             =  0.30
ESCALATE_KEYWORD_BONUS    =  0.10    # per matched keyword
ESCALATE_MAX_BONUS        =  0.60

# Step-reward clip
STEP_CLIP = 2.0


@dataclasses.dataclass
class StepReward:
    value: float
    breakdown: Dict[str, float]


# ── Per-step (called by env.step before terminal handling) ────────────────────
def investigative_reward(
    action: Action, state: WorldState, new_evidence_idxs: list,
) -> StepReward:
    """Reward for a non-terminal investigative action."""
    parts: Dict[str, float] = {"step_penalty": STEP_PENALTY}
    val = STEP_PENALTY

    target = action.ip or action.host or action.entity or ""
    repeat_key = (action.action_type, (target or "").lower())
    # Count PRIOR occurrences only — the current action has already been
    # appended to action_history by env.step(), so we exclude the last entry.
    prior_history = state.action_history[:-1] if state.action_history else []
    repeats = sum(1 for (at, tg, _) in prior_history
                  if (at, (tg or "").lower()) == repeat_key)
    if repeats > 0:
        parts["repeat_penalty"] = REPEAT_PENALTY
        val += REPEAT_PENALTY

    if new_evidence_idxs:
        ev_part = 0.0
        decoy_part = 0.0
        for idx in new_evidence_idxs:
            ev = state.scenario.evidence[idx]
            if ev.confirms_attacker:
                ev_part += NEW_EVIDENCE_REWARD * ev.weight
            else:
                decoy_part += DECOY_EVIDENCE_REWARD * ev.weight
        if ev_part:
            parts["new_attacker_evidence"] = ev_part
            val += ev_part
        if decoy_part:
            parts["decoy_evidence"] = decoy_part
            val += decoy_part

    # Bonus: correlate_events that confirms a real attacker pair
    if action.action_type == "correlate_events" and new_evidence_idxs:
        confirmed = any(
            state.scenario.evidence[i].confirms_attacker
            for i in new_evidence_idxs
        )
        if confirmed:
            parts["correlation_bonus"] = CORRELATION_BONUS
            val += CORRELATION_BONUS

    val = max(-STEP_CLIP, min(STEP_CLIP, val))
    return StepReward(value=val, breakdown=parts)


# ── Terminal (called when episode ends via terminal action OR budget exhaustion)
def terminal_reward(action: Optional[Action], state: WorldState) -> StepReward:
    parts: Dict[str, float] = {}
    val = 0.0
    sc = state.scenario

    # Budget exhausted without a terminal action
    if action is None:
        parts["over_budget"] = OVER_BUDGET_PENALTY
        val = OVER_BUDGET_PENALTY
        return StepReward(value=max(-STEP_CLIP, min(STEP_CLIP, val)), breakdown=parts)

    # Premature: terminal action with <2 evidence (and not closing a benign correctly)
    if (
        len(state.revealed_evidence) < 2
        and not (action.action_type == "close_as_benign" and sc.is_benign)
    ):
        parts["premature_decision"] = PREMATURE_DECISION_PENALTY
        val += PREMATURE_DECISION_PENALTY

    if action.action_type == "identify_attacker":
        if (not sc.is_benign) and action.ip == sc.attacker_ip:
            parts["correct_attacker_id"] = CORRECT_ATTACKER_ID
            val += CORRECT_ATTACKER_ID
        else:
            parts["wrong_attacker_id"] = WRONG_ATTACKER_ID
            val += WRONG_ATTACKER_ID

    elif action.action_type == "close_as_benign":
        if sc.is_benign:
            parts["correct_benign_close"] = CORRECT_BENIGN_CLOSE
            val += CORRECT_BENIGN_CLOSE
        else:
            parts["wrong_benign_close"] = WRONG_BENIGN_CLOSE
            val += WRONG_BENIGN_CLOSE

    elif action.action_type == "isolate_host":
        if (not sc.is_benign) and action.host in (sc.target_hosts or []):
            parts["correct_isolate"] = CORRECT_ISOLATE
            val += CORRECT_ISOLATE
        else:
            parts["wrong_isolate"] = WRONG_ISOLATE
            val += WRONG_ISOLATE

    elif action.action_type == "escalate_incident":
        # base + keyword overlap with reference summary
        summary = (action.summary or "").lower()
        matched = sum(1 for kw in sc.summary_keywords if str(kw).lower() in summary)
        bonus = min(ESCALATE_MAX_BONUS, matched * ESCALATE_KEYWORD_BONUS)
        # Wrong scenario class still gets some credit if at least 1 keyword matched
        scaling = 1.0 if not sc.is_benign else 0.5
        parts["escalate_base"] = ESCALATE_BASE * scaling
        val += ESCALATE_BASE * scaling
        if bonus:
            parts["escalate_keyword_bonus"] = bonus
            val += bonus

    # Evidence-quality bonus: awarded once if >= 3 attacker evidences gathered
    if state.attacker_evidence_collected >= 3:
        parts["evidence_quality_bonus"] = EVIDENCE_QUALITY_BONUS
        val += EVIDENCE_QUALITY_BONUS

    val = max(-STEP_CLIP, min(STEP_CLIP, val))
    return StepReward(value=val, breakdown=parts)
