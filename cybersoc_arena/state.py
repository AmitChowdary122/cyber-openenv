"""Mutable world state for an active CyberSOC investigation."""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Set, Tuple

from cybersoc_arena.scenarios import Scenario


@dataclasses.dataclass
class WorldState:
    scenario: Scenario
    step: int = 0
    done: bool = False
    terminal_action: Optional[str] = None    # which terminal action ended it
    terminal_correct: Optional[bool] = None  # whether that action was the right call

    # Evidence accumulated so far (indexes into scenario.evidence)
    revealed_evidence: List[int] = dataclasses.field(default_factory=list)

    # History of all action invocations (action_type, target, success_flag)
    action_history: List[Tuple[str, str, bool]] = dataclasses.field(default_factory=list)

    # Things the agent has seen — supports "wasted repeated action" penalty
    queried_ips: Set[str] = dataclasses.field(default_factory=set)
    inspected_hosts: Set[str] = dataclasses.field(default_factory=set)
    threat_intel_ips: Set[str] = dataclasses.field(default_factory=set)
    correlated_entities: Set[str] = dataclasses.field(default_factory=set)
    investigated_ips: Set[str] = dataclasses.field(default_factory=set)

    # Final summary text if escalated/closed
    final_summary: Optional[str] = None

    @property
    def remaining_steps(self) -> int:
        return max(0, self.scenario.step_budget - self.step)

    @property
    def total_evidence_count(self) -> int:
        return len(self.scenario.evidence)

    @property
    def attacker_evidence_collected(self) -> int:
        return sum(
            1 for i in self.revealed_evidence
            if self.scenario.evidence[i].confirms_attacker
        )

    def evidence_for_action(self, action_type: str, target: str) -> List[int]:
        """Return indices of unrevealed evidence that match this action+target."""
        out = []
        target_l = (target or "").lower()
        for i, ev in enumerate(self.scenario.evidence):
            if i in self.revealed_evidence:
                continue
            if ev.source != action_type:
                continue
            # Match by exact target
            if target_l and ev.target.lower() != target_l:
                continue
            out.append(i)
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "done": self.done,
            "terminal_action": self.terminal_action,
            "terminal_correct": self.terminal_correct,
            "revealed_evidence": list(self.revealed_evidence),
            "action_history": [list(t) for t in self.action_history],
            "remaining_steps": self.remaining_steps,
            "scenario_type": self.scenario.scenario_type,
            "is_benign": self.scenario.is_benign,
            "final_summary": self.final_summary,
        }
