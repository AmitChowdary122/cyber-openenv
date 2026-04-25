"""CyberSOC Arena Environment Client.

A thin EnvClient subclass that talks to a running CyberSOC Arena FastAPI
server (locally via uvicorn or remotely via the HF Space). Used by the
HF Spaces web interface to connect to the deployed environment.
"""
from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import CyberSOCAction, CyberSOCObservation
except ImportError:
    from models import CyberSOCAction, CyberSOCObservation  # type: ignore


class CyberSOCEnvClient(
    EnvClient[CyberSOCAction, CyberSOCObservation, State]
):
    """
    Client for the CyberSOC Arena Environment.

    Maintains a WebSocket session with a running server.

    Example:
        >>> with CyberSOCEnvClient(base_url="http://localhost:8000") as c:
        ...     r = c.reset()
        ...     print(r.observation.initial_alert)
        ...     r = c.step(CyberSOCAction(action_type="query_logs", ip="10.0.0.1"))
        ...     print(r.observation.last_action_result)
    """

    def _step_payload(self, action: CyberSOCAction) -> Dict[str, Any]:
        return {
            "action_type": action.action_type,
            "ip": action.ip,
            "host": action.host,
            "target": action.target,
        }

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[CyberSOCObservation]:
        obs_data = payload.get("observation") or payload
        observation = CyberSOCObservation(
            done=payload.get("done", obs_data.get("done", False)),
            reward=payload.get("reward", obs_data.get("reward")),
            metadata=obs_data.get("metadata", {}) or {},
            scenario_type=obs_data.get("scenario_type", ""),
            initial_alert=obs_data.get("initial_alert", ""),
            alert_severity=obs_data.get("alert_severity", ""),
            step=int(obs_data.get("step", 0)),
            step_budget=int(obs_data.get("step_budget", 0)),
            revealed_evidence=list(obs_data.get("revealed_evidence", []) or []),
            evidence_count=int(obs_data.get("evidence_count", 0)),
            min_evidence_for_verdict=int(
                obs_data.get("min_evidence_for_verdict", 3)
            ),
            known_entities=dict(
                obs_data.get("known_entities", {"ips": [], "hosts": []})
            ),
            action_history=list(obs_data.get("action_history", []) or []),
            last_action_result=obs_data.get("last_action_result", ""),
            available_actions=list(obs_data.get("available_actions", []) or []),
            is_terminal=bool(obs_data.get("is_terminal", False)),
            curriculum=dict(obs_data.get("curriculum", {}) or {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward", obs_data.get("reward")),
            done=payload.get("done", obs_data.get("done", False)),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=int(payload.get("step_count", 0)),
        )
