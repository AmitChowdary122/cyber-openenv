"""
FastAPI app for CyberSOC Arena (OpenEnv-compatible).

Endpoints exposed by openenv-core's create_app:
  - POST /reset       reset the environment
  - POST /step        execute an action
  - GET  /state       get current environment state
  - GET  /schema      get action / observation schemas
  - WS   /ws          persistent WebSocket session

Run locally:
    uvicorn server.app:app --host 0.0.0.0 --port 8000
or:
    python -m server.app
"""
from openenv.core.env_server.http_server import create_app

try:
    from ..models import CyberSOCAction, CyberSOCObservation
    from .cybersoc_environment import CyberSOCEnvironment
except ImportError:
    from models import CyberSOCAction, CyberSOCObservation  # type: ignore
    from server.cybersoc_environment import CyberSOCEnvironment  # type: ignore


app = create_app(
    CyberSOCEnvironment,
    CyberSOCAction,
    CyberSOCObservation,
    env_name="cybersoc_arena",
    max_concurrent_envs=4,
)


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
