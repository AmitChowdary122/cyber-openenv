"""Untrained-prior agent — minimal LLM-style policy with no training.

Mimics what an out-of-the-box base model does: invokes the most generic
investigation action repeatedly, then commits to an under-informed terminal
decision. Used as a 'before training' lower bound in benchmarks.
"""

from __future__ import annotations

import random
from typing import Any, Dict


class UntrainedPriorAgent:
    name = "untrained_prior"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self._step = 0

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        self._step += 1
        ips = obs.get("asset_inventory", {}).get("visible_ips", [])
        hosts = obs.get("asset_inventory", {}).get("hosts", [])
        evidence = obs.get("evidence_collected", [])

        # Untrained behaviour: always check threat intel on a random IP for the
        # first few steps, then jump to a guess.
        if self._step <= 2 and ips:
            return {"action_type": "check_threat_intel", "ip": self.rng.choice(ips)}
        if self._step <= 4 and ips:
            return {"action_type": "investigate_ip", "ip": self.rng.choice(ips)}

        # Premature guess: pick a random IP and call it the attacker
        if ips and len(evidence) >= 1 and self.rng.random() < 0.6:
            return {"action_type": "identify_attacker", "ip": self.rng.choice(ips)}
        if hosts and self.rng.random() < 0.4:
            return {"action_type": "isolate_host", "host": self.rng.choice(hosts)}
        return {"action_type": "escalate_incident",
                "summary": "Possibly malicious. Forwarding."}
