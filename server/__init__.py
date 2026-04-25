"""OpenEnv server package shim.

The FastAPI app is built in :mod:`cybersoc_arena.server` via
``openenv.core.env_server.create_fastapi_app``. This package just
re-exports it so ``openenv push`` recognises the template layout.
"""
from cybersoc_arena.server import app, main  # noqa: F401

__all__ = ["app", "main"]
