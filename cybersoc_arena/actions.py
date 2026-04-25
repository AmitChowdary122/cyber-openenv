"""Action space for CyberSOC Arena.

Nine SOC analyst tools, split into two groups:

  Investigative (non-terminal, advance the investigation)
    investigate_ip(ip)
    query_logs(ip)
    inspect_endpoint(host)
    check_threat_intel(ip)
    correlate_events(entity)

  Decisive (terminal, end the episode)
    isolate_host(host)
    escalate_incident(summary)
    identify_attacker(ip)
    close_as_benign(reason)
"""

from __future__ import annotations

import dataclasses
import json
import re
from typing import Any, Dict, Optional

INVESTIGATIVE_ACTIONS = {
    "investigate_ip",
    "query_logs",
    "inspect_endpoint",
    "check_threat_intel",
    "correlate_events",
}

TERMINAL_ACTIONS = {
    "isolate_host",
    "escalate_incident",
    "identify_attacker",
    "close_as_benign",
}

ALL_ACTIONS = INVESTIGATIVE_ACTIONS | TERMINAL_ACTIONS


@dataclasses.dataclass
class Action:
    """Structured action.

    For tool calls that operate on an IP/host, the relevant identifier goes in
    the matching field. For escalate_incident, summary holds the analyst note.
    For close_as_benign, summary holds the closure reason.
    """
    action_type: str
    ip: Optional[str] = None
    host: Optional[str] = None
    entity: Optional[str] = None
    summary: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in dataclasses.asdict(self).items() if v is not None}


# JSON-Schema-like description, exposed to LLM agents in the prompt
ACTION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "description": "A SOC analyst action. Pick one action_type per step.",
    "properties": {
        "action_type": {
            "type": "string",
            "enum": sorted(ALL_ACTIONS),
            "description": "Which tool to invoke this step.",
        },
        "ip":      {"type": "string", "description": "IPv4 for ip-targeted actions."},
        "host":    {"type": "string", "description": "Hostname for host-targeted actions."},
        "entity":  {"type": "string", "description": "Entity to correlate (ip or host)."},
        "summary": {
            "type": "string",
            "description": "1-2 sentence analyst note (for escalate_incident / close_as_benign).",
        },
    },
    "required": ["action_type"],
}


# ── Parsing ───────────────────────────────────────────────────────────────────
_IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


def parse_action(raw: Any) -> Action:
    """Parse an action from a dict or a JSON string emitted by an LLM.

    Tolerant of:
      - top-level JSON object
      - JSON wrapped in markdown code fences
      - LLM prose containing one JSON block
      - missing fields (filled with None)
      - common typos in action_type

    Raises ValueError only when no recognisable action_type can be extracted.
    """
    if isinstance(raw, Action):
        return raw

    if isinstance(raw, dict):
        return _action_from_dict(raw)

    if not isinstance(raw, str):
        raise ValueError(f"Cannot parse action from {type(raw).__name__}")

    # Try direct JSON
    text = raw.strip()
    for candidate in _candidate_json_blobs(text):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict) and "action_type" in obj:
                return _action_from_dict(obj)
        except json.JSONDecodeError:
            continue

    # Last-ditch: regex for an action_type keyword
    for at in ALL_ACTIONS:
        if at in text:
            ip_match = _IP_RE.search(text)
            return Action(action_type=at, ip=ip_match.group(0) if ip_match else None)

    raise ValueError(f"No recognisable action in: {text[:200]}")


def _candidate_json_blobs(text: str):
    yield text
    # Markdown fences
    for m in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL):
        yield m.group(1)
    # First {...} block
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        yield m.group(0)


def _action_from_dict(d: Dict[str, Any]) -> Action:
    at = str(d.get("action_type", "")).strip()
    if at not in ALL_ACTIONS:
        # Tolerate 'tool', 'name', 'function' aliases
        for k in ("tool", "name", "function"):
            if d.get(k) in ALL_ACTIONS:
                at = d[k]
                break
    if at not in ALL_ACTIONS:
        raise ValueError(f"Unknown action_type: {d.get('action_type')!r}")

    params = d.get("parameters", d)
    return Action(
        action_type=at,
        ip=params.get("ip"),
        host=params.get("host"),
        entity=params.get("entity") or params.get("target"),
        summary=params.get("summary") or params.get("reason") or params.get("note"),
    )
