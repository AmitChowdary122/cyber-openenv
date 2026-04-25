"""
Actions available to the SOC analyst agent.

Nine tools, split into:
  - Investigative tools (5): gather evidence, no commitment.
  - Terminal tools (4): commit to a verdict, end the episode.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import json


INVESTIGATIVE_ACTIONS = (
    "query_logs",
    "check_threat_intel",
    "inspect_endpoint",
    "correlate_events",
    "list_entities",
)

TERMINAL_ACTIONS = (
    "identify_attacker",
    "block_ip",
    "escalate_incident",
    "mark_benign",
)

ALL_ACTIONS = INVESTIGATIVE_ACTIONS + TERMINAL_ACTIONS


ACTION_DESCRIPTIONS = {
    "query_logs": "Pull SIEM log lines for an IP, host, or entity. Reveals evidence at that target.",
    "check_threat_intel": "Look up an IP/domain in external threat intel feeds. Reveals reputation evidence.",
    "inspect_endpoint": "Run an EDR scan on a host. Reveals process / file / persistence evidence.",
    "correlate_events": "Cross-reference timestamps & flows across hosts. Reveals causal chain evidence.",
    "list_entities": "List the known IPs and hosts seen so far in the investigation.",
    "identify_attacker": "Submit the attacker IP. Terminal verdict — episode ends.",
    "block_ip": "Block an IP at the firewall. Terminal — only correct if it is the attacker.",
    "escalate_incident": "Escalate to incident response. Terminal — correct for confirmed attacks.",
    "mark_benign": "Close the alert as a false positive. Terminal — correct only for benign scenarios.",
}


@dataclass
class Action:
    action_type: str
    ip: Optional[str] = None
    host: Optional[str] = None
    target: Optional[str] = None  # generic target (ip or host)

    def as_dict(self) -> Dict[str, Any]:
        d = {"action_type": self.action_type}
        if self.ip is not None:
            d["ip"] = self.ip
        if self.host is not None:
            d["host"] = self.host
        if self.target is not None:
            d["target"] = self.target
        return d

    @property
    def is_terminal(self) -> bool:
        return self.action_type in TERMINAL_ACTIONS

    @property
    def is_investigative(self) -> bool:
        return self.action_type in INVESTIGATIVE_ACTIONS


def parse_action(raw: Any) -> Action:
    """Robustly parse an Action from a dict, JSON string, or Action instance."""
    if isinstance(raw, Action):
        return raw
    if isinstance(raw, str):
        s = raw.strip()
        try:
            raw = json.loads(s)
        except Exception:
            # Fallback: treat as bare action_type
            return Action(action_type=s)
    if not isinstance(raw, dict):
        return Action(action_type="list_entities")

    at = str(raw.get("action_type", "list_entities")).strip()
    if at not in ALL_ACTIONS:
        # Unknown action -> default to a no-op investigative action
        at = "list_entities"

    params = raw.get("parameters") or {}
    ip = raw.get("ip") or params.get("ip")
    host = raw.get("host") or params.get("host")
    target = raw.get("target") or params.get("target")

    return Action(action_type=at, ip=ip, host=host, target=target)
