"""Observation builder: world state -> structured observation for the agent.

The observation deliberately HIDES ground truth (attacker IP) but EXPOSES:
  - the original alert and severity
  - asset inventory (hosts) and IPs visible in logs
  - evidence collected so far (text descriptions)
  - history of actions taken
  - remaining step budget
  - background noise (a slice of unrelated log lines)

The observation is JSON-serialisable so it can be fed to any LLM via prompt.
"""

from __future__ import annotations

from typing import Any, Dict, List

from cybersoc_arena.actions import ACTION_SCHEMA
from cybersoc_arena.state import WorldState


def build_observation(state: WorldState) -> Dict[str, Any]:
    sc = state.scenario

    # All IPs the analyst can see in logs (decoys + benign + attacker)
    visible_ips: List[str] = sorted(set(sc.all_ips()))

    # Evidence revealed (descriptions only — never weights or confirms_attacker flags)
    evidence_log = [
        {
            "step": h_step,
            "action": h_action,
            "target": h_target,
            "finding": sc.evidence[ev_idx].description,
        }
        for h_step, (h_action, h_target, _, ev_idx) in enumerate(_evidence_history(state))
    ]

    return {
        "alert": {
            "summary": sc.initial_alert,
            "severity": sc.alert_severity,
        },
        "step": state.step,
        "remaining_steps": state.remaining_steps,
        "step_budget": sc.step_budget,
        "asset_inventory": {
            "hosts": sorted(set(sc.target_hosts)),
            "visible_ips": visible_ips,
        },
        "evidence_collected": evidence_log,
        "evidence_count": len(state.revealed_evidence),
        "action_history": [
            {"action_type": at, "target": tgt, "success": ok}
            for (at, tgt, ok) in state.action_history
        ],
        "noise_sample": sc.background_logs[:3],
        "available_actions": sorted(
            ACTION_SCHEMA["properties"]["action_type"]["enum"]
        ),
        "goal": (
            "Triage the alert. Investigate to gather evidence, then take ONE "
            "terminal action (identify_attacker / isolate_host / escalate_incident "
            "/ close_as_benign). Avoid false positives. You have a step budget."
        ),
    }


def _evidence_history(state: WorldState):
    """Reconstruct (action, target, ok, ev_idx) for each revealed evidence."""
    out = []
    seen = set()
    for ev_idx in state.revealed_evidence:
        ev = state.scenario.evidence[ev_idx]
        # Find the action_history entry that produced this evidence
        for (at, tgt, ok) in state.action_history:
            if (at, tgt) in seen:
                continue
            if at == ev.source and (tgt or "").lower() == ev.target.lower():
                out.append((at, tgt, ok, ev_idx))
                seen.add((at, tgt))
                break
    return out


def render_observation_text(obs: Dict[str, Any]) -> str:
    """Plain-text rendering for prompt construction."""
    lines = []
    lines.append(f"=== ALERT [{obs['alert']['severity'].upper()}] ===")
    lines.append(obs["alert"]["summary"])
    lines.append("")
    lines.append(
        f"Step {obs['step']}/{obs['step_budget']}  "
        f"(remaining: {obs['remaining_steps']})"
    )
    lines.append("")
    lines.append("Asset inventory:")
    lines.append(f"  hosts:       {', '.join(obs['asset_inventory']['hosts']) or '(none)'}")
    lines.append(f"  visible IPs: {', '.join(obs['asset_inventory']['visible_ips']) or '(none)'}")
    lines.append("")
    lines.append("Background noise (sample):")
    for ln in obs["noise_sample"]:
        lines.append(f"  {ln}")
    lines.append("")
    if obs["evidence_collected"]:
        lines.append("Evidence collected so far:")
        for e in obs["evidence_collected"]:
            lines.append(f"  [{e['action']}({e['target']})] {e['finding']}")
    else:
        lines.append("Evidence collected so far: NONE — investigate before deciding.")
    lines.append("")
    lines.append(f"Available actions: {', '.join(obs['available_actions'])}")
    lines.append("")
    lines.append(obs["goal"])
    return "\n".join(lines)
