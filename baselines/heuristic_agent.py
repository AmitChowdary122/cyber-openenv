"""Heuristic agent — rule-based SOC playbook.

Strategy:
  Phase 1 (host-first): inspect every host. Hosts are scarce (1-3) and
                        typically encode high-signal evidence.
  Phase 2 (intel-then-logs interleaved): check_threat_intel + query_logs on
                        every visible IP, alternating per IP.
  Phase 3 (host logs): query_logs on hosts (some scenarios index logs by host).
  Phase 4 (correlate): correlate_events on the IP with the most evidence hits.
  Decide when:
      evidence >= 3 OR remaining_steps <= 1.

The decision logic (_decide) classifies the evidence text as benign, malicious,
or ambiguous and emits the appropriate terminal action with a keyword-rich
summary so the escalate keyword bonus can fire.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, Optional

_BENIGN_KEYWORDS = ("benign", "no malicious", "no harm", "scanner", "load test", "unrelated")


class HeuristicAgent:
    name = "heuristic"

    def __init__(self):
        self._asked: set = set()

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        evidence = obs.get("evidence_collected", [])
        remaining = obs.get("remaining_steps", 99)
        ips = obs.get("asset_inventory", {}).get("visible_ips", [])
        hosts = obs.get("asset_inventory", {}).get("hosts", [])

        # Time-pressure gate: decide if running out of budget
        if len(evidence) >= 3 or remaining <= 1:
            return self._decide(evidence, ips, hosts)

        return self._gather_more(ips, hosts, evidence)

    # ── helpers ───────────────────────────────────────────────────────────────
    def _pick_unasked(self, candidates: list, action: str) -> str:
        for c in candidates:
            if (action, c) not in self._asked:
                self._asked.add((action, c))
                return c
        return candidates[0]

    def _gather_more(self, ips: list, hosts: list, evidence: list) -> Dict[str, Any]:
        """Plan layout: hosts first, then IPs interleaved (intel + logs)."""
        plan: list = []

        # Phase 1 — inspect every host (scarce, high signal)
        for h in hosts:
            plan.append(("inspect_endpoint", h, "host"))

        # Phase 2 — for each IP, alternate threat_intel and query_logs
        for ip in ips:
            plan.append(("check_threat_intel", ip, "ip"))
            plan.append(("query_logs", ip, "ip"))

        # Phase 3 — log queries on hosts (some scenarios key logs on hosts)
        for h in hosts:
            plan.append(("query_logs", h, "host"))

        # Phase 4 — correlate on entities mentioned in evidence (or top IP/host)
        seen_entities = self._entities_in_evidence(evidence)
        for e in (seen_entities + ips[:1] + hosts[:1]):
            plan.append(("correlate_events", e, "entity"))

        for action, target, key in plan:
            if (action, target) not in self._asked:
                self._asked.add((action, target))
                return {"action_type": action, key: target}

        # Plan exhausted → fall back
        if ips:
            return {"action_type": "investigate_ip", "ip": ips[0]}
        return {"action_type": "escalate_incident",
                "summary": "Insufficient signal, escalating for review."}

    def _entities_in_evidence(self, evidence: list) -> list:
        """Extract distinct IPs/hosts mentioned in any evidence finding."""
        out: list = []
        seen: set = set()
        text = " ".join(e.get("finding", "") for e in evidence)
        # Pull IPs
        for m in re.finditer(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", text):
            ip = m.group(0)
            if ip not in seen:
                seen.add(ip)
                out.append(ip)
        # Pull hostnames matching common roles
        for m in re.finditer(r"\b(?:ws|srv|db|edge|auth)-\d{3}\b", text):
            h = m.group(0)
            if h not in seen:
                seen.add(h)
                out.append(h)
        return out

    def _decide(self, evidence: list, ips: list, hosts: list) -> Dict[str, Any]:
        text_blob = " ".join(e.get("finding", "") for e in evidence).lower()

        has_malicious = _has_malicious_signal(text_blob)
        has_benign    = any(kw in text_blob for kw in _BENIGN_KEYWORDS)

        # Strong benign signal + no malicious noise -> close
        if has_benign and not has_malicious:
            return {
                "action_type": "close_as_benign",
                "summary": "Evidence indicates benign scanner activity, no compromise.",
            }

        # Tally IP mentions across the evidence text
        ip_counts: Counter = Counter()
        for ip in ips:
            c = text_blob.count(ip.lower())
            if c:
                ip_counts[ip] = c

        # Malicious signal + at least one IP mentioned -> identify the top one
        if has_malicious and ip_counts:
            best_ip = ip_counts.most_common(1)[0][0]
            return {"action_type": "identify_attacker", "ip": best_ip}

        # Malicious signal but ambiguous attribution -> escalate with a summary
        # rich in keywords so the keyword-overlap reward fires
        if has_malicious:
            kws = [
                kw for kw in (
                    "phishing", "lateral", "credential", "stuffing", "successful",
                    "exfiltration", "egress", "kill chain", "persistence",
                    "pivot", "c2",
                )
                if kw in text_blob
            ]
            ip_hint = ips[0] if ips else ""
            return {
                "action_type": "escalate_incident",
                "summary": f"Malicious activity: {', '.join(kws)} indicators. Source IP {ip_hint}",
            }

        # No clear signal — pick the most-mentioned host and isolate as a hedge
        host_counts: Counter = Counter()
        for h in hosts:
            host_counts[h] = text_blob.count(h.lower())
        if host_counts and max(host_counts.values()) >= 1:
            best_host = host_counts.most_common(1)[0][0]
            return {"action_type": "isolate_host", "host": best_host}

        return {
            "action_type": "escalate_incident",
            "summary": "Insufficient evidence; escalating for human review.",
        }


def _has_malicious_signal(text: str) -> bool:
    bad = ("mimikatz", "exfil", "c2", "webshell", "exploit", "ransomware", "credential",
           "successful auth", "lateral", "kill chain")
    return any(b in text for b in bad)
