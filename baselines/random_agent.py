"""Random agent — uniform over actions, random target selection."""

from __future__ import annotations

import random
from typing import Any, Dict

from cybersoc_arena.actions import ALL_ACTIONS


class RandomAgent:
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        action_type = self.rng.choice(sorted(ALL_ACTIONS))
        ips = obs.get("asset_inventory", {}).get("visible_ips", [])
        hosts = obs.get("asset_inventory", {}).get("hosts", [])

        out: Dict[str, Any] = {"action_type": action_type}
        if action_type in ("investigate_ip", "query_logs", "check_threat_intel", "identify_attacker"):
            if ips:
                out["ip"] = self.rng.choice(ips)
        elif action_type in ("inspect_endpoint", "isolate_host"):
            if hosts:
                out["host"] = self.rng.choice(hosts)
        elif action_type == "correlate_events":
            entities = ips + hosts
            if entities:
                out["entity"] = self.rng.choice(entities)
        elif action_type == "escalate_incident":
            out["summary"] = "Random escalation, please review."
        elif action_type == "close_as_benign":
            out["summary"] = "Random closure - looks benign."
        return out
