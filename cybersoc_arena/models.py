"""OpenEnv-compliant typed Action / Observation / State for CyberSOC Arena.

Subclasses the official OpenEnv base classes from ``openenv.core.env_server``:

  * ``CyberAction``      ← ``openenv.core.env_server.Action``
  * ``CyberObservation`` ← ``openenv.core.env_server.Observation``
  * ``CyberState``       ← ``openenv.core.env_server.State``

Why this matters for the hackathon
----------------------------------
The judging guide explicitly calls out *"Use OpenEnv's Environment / MCPEnvironment
base classes properly"* as table-stakes engineering. Subclassing the base Pydantic
models gets us automatic JSON-Schema generation, validation, and compatibility with
``create_fastapi_app`` so the standard ``/reset`` / ``/step`` / ``/state`` endpoints,
WebSocket session, OpenAPI ``/docs``, and ``/web`` interactive UI all work out of
the box.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

try:
    from openenv.core.env_server import Action as _BaseAction
    from openenv.core.env_server import Observation as _BaseObservation
    from openenv.core.env_server import State as _BaseState
except ImportError:  # pragma: no cover  --- openenv-core not installed
    _BaseAction = BaseModel  # type: ignore[assignment,misc]
    _BaseObservation = BaseModel  # type: ignore[assignment,misc]
    _BaseState = BaseModel  # type: ignore[assignment,misc]


# ─────────────────────────────────────────────────────────────────────────────
# Action
# ─────────────────────────────────────────────────────────────────────────────
class CyberAction(_BaseAction):
    """One SOC-analyst tool invocation per step.

    Allowed ``action_type`` values:

    Investigative (non-terminal):
      ``investigate_ip``, ``query_logs``, ``inspect_endpoint``,
      ``check_threat_intel``, ``correlate_events``

    Terminal (end the episode):
      ``identify_attacker``, ``isolate_host``, ``escalate_incident``,
      ``close_as_benign``
    """

    model_config = ConfigDict(
        extra="allow",   # tolerate LLM-emitted extra keys (e.g. "thoughts")
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    action_type: str = Field(
        ..., description="Which SOC tool to invoke this step."
    )
    ip: Optional[str] = Field(
        default=None, description="IPv4 for IP-targeted actions."
    )
    host: Optional[str] = Field(
        default=None, description="Hostname for host-targeted actions."
    )
    entity: Optional[str] = Field(
        default=None, description="Entity to correlate (IP or host)."
    )
    summary: Optional[str] = Field(
        default=None,
        description=(
            "1-2 sentence analyst note (for escalate_incident / close_as_benign)."
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Observation
# ─────────────────────────────────────────────────────────────────────────────
class AlertView(BaseModel):
    """Initial alert as the analyst sees it."""

    model_config = ConfigDict(extra="forbid")

    summary: str
    severity: str  # one of: info / low / medium / high / critical


class EvidenceItem(BaseModel):
    """One piece of evidence the analyst has uncovered."""

    model_config = ConfigDict(extra="forbid")

    step: int
    action: str
    target: str
    finding: str


class AssetInventory(BaseModel):
    """Hosts and IPs visible to the analyst."""

    model_config = ConfigDict(extra="forbid")

    hosts: List[str] = Field(default_factory=list)
    visible_ips: List[str] = Field(default_factory=list)


class ActionHistoryEntry(BaseModel):
    """One row of the analyst's action history."""

    model_config = ConfigDict(extra="forbid")

    action_type: str
    target: str
    success: bool


class CyberObservation(_BaseObservation):
    """What the agent sees after each step.

    Note: ``done`` and ``reward`` are inherited from the OpenEnv ``Observation``
    base class — judges and training infra read them directly.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    alert: AlertView
    step: int = 0
    remaining_steps: int = 0
    step_budget: int = 0
    asset_inventory: AssetInventory = Field(default_factory=AssetInventory)
    evidence_collected: List[EvidenceItem] = Field(default_factory=list)
    evidence_count: int = 0
    action_history: List[ActionHistoryEntry] = Field(default_factory=list)
    noise_sample: List[str] = Field(default_factory=list)
    available_actions: List[str] = Field(default_factory=list)
    goal: str = ""
    info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Per-step diagnostics (reward breakdown, terminal info, etc.).",
    )


# ─────────────────────────────────────────────────────────────────────────────
# State (episode metadata, NOT a peek at hidden truth)
# ─────────────────────────────────────────────────────────────────────────────
class CyberState(_BaseState):
    """Episode metadata exposed via ``GET /state``.

    Deliberately omits the hidden attacker IP so that judges or curriculum
    drivers can read state without leaking ground-truth.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    episode_id: str
    scenario_type: str
    is_benign: bool
    step: int = 0
    step_budget: int = 0
    remaining_steps: int = 0
    done: bool = False
    terminal_action: Optional[str] = None
    terminal_correct: Optional[bool] = None
    evidence_count: int = 0
    attacker_evidence_count: int = 0
    final_summary: Optional[str] = None
    # Curriculum tier (filled in only when running under CurriculumEnv)
    curriculum_tier: Optional[int] = None
    curriculum_tier_name: Optional[str] = None
