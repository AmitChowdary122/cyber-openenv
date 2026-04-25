"""HTTP / WebSocket client for a remote CyberSOC Arena server.

This is the public client surface. It does NOT import any server-side
internals (per OpenEnv's "respect the client / server separation" rule).

Two flavours are provided:

  * CyberSOCClient       - synchronous requests-based client; great for
                           notebooks, demos, and tests. Hits /reset, /step,
                           /state on a CyberSOC Arena server.
  * CyberSOCAsyncClient  - subclass of openenv.core.EnvClient. Use this in
                           TRL / Unsloth / torchforge / SkyRL / veRL training
                           loops over WebSocket sessions. Falls back to None
                           if openenv-core is not installed.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from cybersoc_arena.models import CyberAction, CyberObservation, CyberState


class CyberSOCClient:
    """Synchronous HTTP client for a CyberSOC Arena server.

    Mirrors the in-process CyberSOCEnv API so an LLM agent can target either
    the local class or a remote Space without code changes.

    Example:
        client = CyberSOCClient("https://amit51-cybersoc-arena.hf.space")
        obs = client.reset(seed=42)
        obs = client.step(CyberAction(action_type="investigate_ip", ip="10.0.0.5"))
        snap = client.state()
    """

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 30.0):
        import requests  # imported lazily so unit tests do not need it
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()
        self._timeout = timeout

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        scenario_type: Optional[str] = None,
    ) -> CyberObservation:
        payload: Dict[str, Any] = {}
        if seed is not None:
            payload["seed"] = seed
        if episode_id is not None:
            payload["episode_id"] = episode_id
        if scenario_type is not None:
            payload["scenario_type"] = scenario_type
        r = self._session.post(
            f"{self.base_url}/reset", json=payload, timeout=self._timeout
        )
        r.raise_for_status()
        body = r.json()
        return CyberObservation(**body.get("observation", body))

    def step(self, action: CyberAction) -> CyberObservation:
        if isinstance(action, CyberAction):
            action_payload = action.model_dump(exclude_none=True)
        else:
            action_payload = dict(action)
        r = self._session.post(
            f"{self.base_url}/step",
            json={"action": action_payload},
            timeout=self._timeout,
        )
        r.raise_for_status()
        body = r.json()
        return CyberObservation(**body.get("observation", body))

    def state(self) -> CyberState:
        r = self._session.get(f"{self.base_url}/state", timeout=self._timeout)
        r.raise_for_status()
        body = r.json()
        return CyberState(**body.get("state", body))

    def health(self) -> Dict[str, Any]:
        r = self._session.get(f"{self.base_url}/health", timeout=self._timeout)
        r.raise_for_status()
        return r.json()


# Async client subclasses the official openenv.core.EnvClient when available.
try:
    from openenv.core import EnvClient as _BaseEnvClient

    class CyberSOCAsyncClient(_BaseEnvClient):  # type: ignore[type-arg,misc]
        """Async WebSocket client that drives a CyberSOC Arena Space.

        Use this from TRL / Unsloth GRPO training loops:
            async with CyberSOCAsyncClient(
                base_url="https://amit51-cybersoc-arena.hf.space"
            ) as env:
                obs = await env.reset(seed=42)
                obs = await env.step(
                    CyberAction(action_type="investigate_ip", ip="10.0.0.5")
                )
        """
        action_cls = CyberAction
        observation_cls = CyberObservation

except ImportError:  # pragma: no cover  --- openenv-core not installed
    CyberSOCAsyncClient = None  # type: ignore[assignment]


__all__ = ["CyberSOCClient", "CyberSOCAsyncClient"]
