"""OpenEnv-compliant FastAPI server for CyberSOC Arena.

Tries the Gradio-backed ``create_web_interface_app`` with our custom
SOC-analyst Gradio UI first. Falls back to the generic auto-generated
Gradio form if our custom builder fails to import or build, and finally
falls back to the plain ``create_fastapi_app`` (JSON-only) if gradio
itself is missing. So the JSON ``/reset``, ``/step``, ``/state`` surface
keeps working in every failure mode.

Endpoints exposed:
  * POST /reset    : start a new episode (optionally with seed)
  * POST /step     : take one action, get an observation back
  * GET  /state    : current episode metadata (no ground-truth leak)
  * GET  /health   : liveness probe
  * GET  /docs     : OpenAPI / Swagger UI
  * GET  /web      : custom SOC-analyst Gradio UI (when gradio installed)
  * WS   /ws       : persistent WebSocket session (used by EnvClient)
"""

from __future__ import annotations

from typing import Any

from cybersoc_arena.env import CyberSOCEnv
from cybersoc_arena.models import CyberAction, CyberObservation


def _create_app() -> Any:
    """Build the FastAPI app, preferring the custom Gradio /web UI."""
    import sys
    import traceback

    # Step 1: try gradio + custom CyberSOC builder
    try:
        from openenv.core.env_server.web_interface import (
            create_web_interface_app,
        )
        try:
            from cybersoc_arena.web_ui import build_cybersoc_gradio_ui
            print("[server] building /web with custom CyberSOC Gradio UI",
                  flush=True)
            app = create_web_interface_app(
                env=CyberSOCEnv,
                action_cls=CyberAction,
                observation_cls=CyberObservation,
                env_name="cybersoc-arena",
                gradio_builder=build_cybersoc_gradio_ui,
            )
        except Exception as exc:
            print(
                f"[server] custom Gradio builder failed "
                f"({exc.__class__.__name__}: {exc}); "
                "falling back to default Gradio UI.",
                flush=True,
            )
            traceback.print_exc()
            app = create_web_interface_app(
                env=CyberSOCEnv,
                action_cls=CyberAction,
                observation_cls=CyberObservation,
                env_name="cybersoc-arena",
            )
        web_present = "/web" in [r.path for r in app.routes]
        print(f"[server] /web route registered: {web_present}", flush=True)
        return app
    except Exception as exc:
        print(
            f"[server] /web Gradio UI unavailable "
            f"({exc.__class__.__name__}: {exc}); "
            "falling back to JSON-only API. /reset /step /state /docs still work.",
            flush=True,
        )
        traceback.print_exc()
        from openenv.core.env_server import create_fastapi_app
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
