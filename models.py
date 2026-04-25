"""
OpenEnv-compatible Pydantic models for CyberSOC Arena.

These wrap the dict-based observations / actions of cybersoc_arena into the
typed pydantic Action / Observation classes that openenv-core expects.
"""
from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class CyberSOCAction(Action):
    """Action for CyberSOC Arena.

    The agent picks one of nine tools (5 investigative + 4 terminal) and
    optionally targets an IP, host, or generic entity name.
    """

    action_type: str = Field(
        ...,
        description=(
            "One of: query_logs, check_threat_intel, inspect_endpoint, "
            "correlate_events, list_entities, identify_attacker, block_ip, "
            "escalate_incident, mark_benign."
        ),
    )
    ip: Optional[str] = Field(default=None, description="Target IP for the action.")
    host: Optional[str] = Field(default=None, description="Target host for the action.")
    target: Optional[str] = Field(
        default=None,
        description="Generic target (IP or host) — used when the agent does not distinguish.",
    )


class CyberSOCObservation(Observation):
    """Observation from CyberSOC Arena.

    Inherits ``done``, ``reward``, ``metadata`` from the OpenEnv base
    Observation and adds SOC-specific fields.
    """

    scenario_type: str = Field(default="", description="Currently active scenario.")
    initial_alert: str = Field(default="", description="The SIEM alert that fired.")
    alert_severity: str = Field(default="", description="low / medium / high.")
    step: int = Field(default=0, description="Current step (1-indexed after first action).")
    step_budget: int = Field(default=0, description="Total steps allowed in this episode.")
    revealed_evidence: List[str] = Field(
        default_factory=list,
        description="Evidence findings the agent has surfaced so far.",
    )
    evidence_count: int = Field(default=0, description="Number of distinct evidence items revealed.")
    min_evidence_for_verdict: int = Field(
        default=3,
        description="Confirming-evidence threshold below which a terminal action is penalised.",
    )
    known_entities: Dict[str, List[str]] = Field(
        default_factory=lambda: {"ips": [], "hosts": []},
        description="IPs and hosts visible in the current investigation.",
    )
    action_history: List[Dict[str, Any]] = Field(
        default_factory=list, description="The last 5 tool calls."
    )
    last_action_result: str = Field(
        default="", description="Natural-language feedback on the most recent action."
    )
    available_actions: List[str] = Field(
        default_factory=list, description="The 9 tools available to the agent."
    )
    is_terminal: bool = Field(default=False, description="Whether the episode has ended.")
    curriculum: Dict[str, Any] = Field(
        default_factory=dict,
        description="Tier / progress info from CurriculumEnv (or empty when wrapping CyberSOCEnv).",
    )
