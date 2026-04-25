"""
CyberSOC Arena — an OpenEnv-style environment for training LLMs to act as
SOC analysts: long-horizon, partially-observable, tool-using, and gated
behind an adaptive difficulty curriculum.
"""
from .actions import (
    Action,
    parse_action,
    INVESTIGATIVE_ACTIONS,
    TERMINAL_ACTIONS,
    ALL_ACTIONS,
    ACTION_DESCRIPTIONS,
)
from .scenarios import (
    Evidence,
    Scenario,
    SCENARIO_TYPES,
    generate_scenario,
)
from .env import CyberSOCEnv
from .curriculum import CurriculumEnv, TIERS, Tier

__all__ = [
    "CyberSOCEnv",
    "CurriculumEnv",
    "TIERS",
    "Tier",
    "Action",
    "parse_action",
    "SCENARIO_TYPES",
    "Scenario",
    "Evidence",
    "generate_scenario",
    "INVESTIGATIVE_ACTIONS",
    "TERMINAL_ACTIONS",
    "ALL_ACTIONS",
    "ACTION_DESCRIPTIONS",
]

__version__ = "0.2.0"
