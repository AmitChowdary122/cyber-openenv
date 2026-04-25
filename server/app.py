"""Server entry point at the OpenEnv-template path ``server/app.py``.

Re-exports the FastAPI ``app`` built in :mod:`cybersoc_arena.server`. Run with::

    uvicorn server.app:app --host 0.0.0.0 --port 8000
    # or, equivalently:
    uvicorn cybersoc_arena.server:app --host 0.0.0.0 --port 8000
"""
from cybersoc_arena.server import app, main  # noqa: F401

__all__ = ["app", "main"]
