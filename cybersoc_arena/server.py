"""OpenEnv-compliant FastAPI server for CyberSOC Arena.

Built on top of openenv.core.env_server.create_fastapi_app so the standard
endpoints judges expect are exposed automatically:

  * POST /reset    : start a new episode (optionally with seed)
  * POST /step     : take one action, get an observation back
  * GET  /state    : current episode metadata (no ground-truth leak)
  * GET  /health   : liveness probe
  * GET  /docs     : OpenAPI / Swagger UI
  * GET  /web      : interactive HumanAgent UI (nice for the Space)
  * WS   /ws       : persistent WebSocket session (used by EnvClient)

Run locally:
  uv run server                            # via the [project.scripts] shim
  uvicorn cybersoc_arena.server:app --host 0.0.0.0 --port 8000

On Hugging Face Spaces the Dockerfile invokes the `server` entry point
defined in pyproject.toml.
"""

from __future__ import annotations

from typing import Any

from cybersoc_arena.env import CyberSOCEnv
from cybersoc_arena.models import CyberAction, CyberObservation


def _create_app() -> Any:
    """Build the FastAPI app; imported lazily so test envs do not need uvicorn."""
    from openenv.core.env_server import create_fastapi_app

    # Factory pattern: a fresh CyberSOCEnv per WebSocket session.
    return create_fastapi_app(
        env=CyberSOCEnv,
        action_cls=CyberAction,
        observation_cls=CyberObservation,
    )


# Module-level FastAPI app for `uvicorn cybersoc_arena.server:app`.
try:
    app = _create_app()
    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover  --- openenv-core / fastapi missing
    app = None
    _IMPORT_ERROR = exc


def main() -> None:
    """Entry point for [project.scripts] server."""
    import os
    import uvicorn

    if app is None:  # pragma: no cover
        raise RuntimeError(
            "FastAPI app could not be created. "
            "Did you `pip install openenv-core fastapi uvicorn`? "
            f"Original error: {_IMPORT_ERROR}"
        )
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    workers = int(os.environ.get("WORKERS", "1"))
    uvicorn.run(
        "cybersoc_arena.server:app",
        host=host,
        port=port,
        workers=workers,
    )


if __name__ == "__main__":
    main()
