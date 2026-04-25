"""CyberSOC Arena - SOC investigation OpenEnv environment for LLM agents.

CyberSOC Arena is a long-horizon, multi-tool environment where an LLM acts
as a Tier-2 SOC analyst. It triages noisy alerts, picks the right tool with
the right target, correlates evidence across multiple hosts, and reaches a
final incident verdict under a step budget. Built on top of the official
openenv.core.env_server.Environment base class so the standard /reset,
/step, /state Gym-style HTTP surface, the /web interactive UI, and the /ws
WebSocket session all work out of the box once deployed to a HF Space.

Public API
==========
- CyberSOCEnv         : the OpenEnv-compliant environment
- CurriculumEnv       : adaptive-difficulty self-improvement wrapper (Theme 4)
- CyberSOCClient      : synchronous HTTP client for a remote Space
- CyberSOCAsyncClient : async EnvClient subclass (TRL-friendly), or None if openenv-core is missing
- CyberAction, CyberObservation, CyberState : Pydantic models subclassing OpenEnv base types
- AlertView, EvidenceItem    : observation sub-models
- ACTION_SCHEMA, parse_action : LLM-tolerant action parser
- SCENARIO_TYPES, generate_scenario : 6 procedural scenario generators
- TIERS               : the 6-tier curriculum ladder
"""

from cybersoc_arena.actions import ACTION_SCHEMA, parse_action
from cybersoc_arena.actions import Action  # legacy dataclass (used by rewards.py)
from cybersoc_arena.client import CyberSOCAsyncClient, CyberSOCClient
from cybersoc_arena.curriculum import TIERS, CurriculumEnv, Tier
from cybersoc_arena.env import CyberSOCEnv
from cybersoc_arena.models import (
    AlertView,
    CyberAction,
    CyberObservation,
    CyberState,
    EvidenceItem,
)
from cybersoc_arena.scenarios import SCENARIO_TYPES, generate_scenario

__version__ = "0.3.0"

__all__ = [
    "CyberSOCEnv",
    "CurriculumEnv",
    "Tier",
    "TIERS",
    "CyberSOCClient",
    "CyberSOCAsyncClient",
    "CyberAction",
    "CyberObservation",
    "CyberState",
    "AlertView",
    "EvidenceItem",
    "Action",
    "ACTION_SCHEMA",
    "parse_action",
    "SCENARIO_TYPES",
    "generate_scenario",
]
