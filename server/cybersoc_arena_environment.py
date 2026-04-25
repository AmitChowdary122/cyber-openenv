"""OpenEnv-template environment file: re-exports the real CyberSOCEnv.

The implementation lives in :mod:`cybersoc_arena.env`. This module exists
so the ``openenv push`` template validator can find a
``<env_name>_environment.py`` next to ``server/app.py``.
"""
from cybersoc_arena.env import CyberSOCEnv  # noqa: F401

__all__ = ["CyberSOCEnv"]
