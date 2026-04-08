# app/state.py
"""
CyberEnv State – simulates a multi-stage cyber attack with realistic SOC logs.
"""

import random
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta


class SimpleState:
    def __init__(self):
        self.attackers: List[str] = []
        self.decoy_ips: List[str] = []
        self.benign_suspicious_ips: List[str] = []
        self.attack_types: Dict[str, str] = {}
        self.attack_stages: Dict[str, str] = {}
        self.suspicion_scores: Dict[str, float] = {}

        self.attacker_ip: Optional[str] = None
        self.attack_scenario: Optional[str] = None

        self.logs: List[str] = []
        self.alerts: List[str] = []
        self.system_state: Dict[str, Any] = {
            "status": "operational",
            "threat_level": 0.0,
            "compromised": False,
            "blocked_ips": [],
            "identified_attacker": None,
        }

        self.steps: int = 0
        self.investigation_history: Dict[str, int] = {}

    # =========================
    # RESET
    # =========================
    def reset(self, difficulty: str = "easy") -> None:
        self.logs = []
        self.alerts = []
        self.steps = 0
        self.investigation_history = {}

        self.system_state = {
            "status": "operational",
            "threat_level": 0.0,
            "compromised": False,
            "blocked_ips": [],
            "identified_attacker": None,
        }

        if difficulty == "hard":
            attacker = f"10.0.0.{random.randint(1,50)}"
            self.attackers = [attacker]

            self.decoy_ips = [f"10.0.0.{i}" for i in random.sample(range(51,100),2)]
            self.benign_suspicious_ips = [f"10.0.0.{i}" for i in random.sample(range(101,150),3)]

            all_ips = self.attackers + self.decoy_ips + self.benign_suspicious_ips
            self.suspicion_scores = {ip: 0.0 for ip in all_ips}

        else:
            self.attackers = [f"10.0.0.{random.randint(1,50)}"]
            self.decoy_ips = []
            self.benign_suspicious_ips = []
            self.suspicion_scores = {self.attackers[0]: 0.0}

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.logs.append(f"[{now}] [INFO] System initialized")
        self.alerts.append(f"[{now}] Monitoring started")

    # =========================
    # LOG GENERATION
    # =========================
    def generate_step_logs(self, difficulty: str = "easy") -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # attacker behavior
        for ip in self.attackers:
            self.logs.append(f"[{ts}] [ATTACK] Suspicious activity from {ip}")
            self.suspicion_scores[ip] = min(1.0, self.suspicion_scores[ip] + 0.2)

        # decoys
        for ip in self.decoy_ips:
            self.logs.append(f"[{ts}] [DECOY] High traffic from {ip}")
            self.suspicion_scores[ip] = min(1.0, self.suspicion_scores[ip] + 0.15)

        # benign
        for ip in self.benign_suspicious_ips:
            self.logs.append(f"[{ts}] [NOISE] Random event from {ip}")
            self.suspicion_scores[ip] = min(1.0, self.suspicion_scores[ip] + 0.05)

        # update threat level
        all_ips = self.attackers + self.decoy_ips + self.benign_suspicious_ips
        if all_ips:
            avg = sum(self.suspicion_scores[ip] for ip in all_ips) / len(all_ips)
            self.system_state["threat_level"] = min(1.0, avg)

    # =========================
    # ACTION HANDLING (FIXED)
    # =========================
    def apply_action(self, action_type: str, parameters: Dict[str, Any] = None) -> None:
        if parameters is None:
            parameters = {}

        self.steps += 1
        ip = parameters.get("ip", None)

        if action_type == "analyze_log":
            self.system_state["threat_level"] = max(0.0, self.system_state["threat_level"] - 0.05)

        elif action_type == "identify_attacker":
            if ip is not None and ip in self.attackers:
                self.system_state["identified_attacker"] = ip
                self.system_state["threat_level"] = max(0.0, self.system_state["threat_level"] - 0.2)
            else:
                self.system_state["threat_level"] = min(1.0, self.system_state["threat_level"] + 0.1)

        elif action_type == "block_ip":
            if ip is not None:
                if ip not in self.system_state["blocked_ips"]:
                    self.system_state["blocked_ips"].append(ip)
                    if ip in self.attackers:
                        self.system_state["threat_level"] = max(0.0, self.system_state["threat_level"] - 0.3)

        elif action_type == "investigate_ip":
            if ip is not None:
                self.investigation_history[ip] = self.investigation_history.get(ip, 0) + 1

        elif action_type == "quarantine_system":
            self.system_state["status"] = "quarantined"
            self.system_state["threat_level"] = max(0.0, self.system_state["threat_level"] - 0.5)

    # =========================
    # TERMINATION
    # =========================
    def is_terminal(self, max_steps: int, task_id: str) -> bool:
        if self.steps >= max_steps:
            return True

        identified = self.system_state.get("identified_attacker")
        blocked = self.system_state.get("blocked_ips", [])

        return identified in blocked

    # =========================
    # SERIALIZATION
    # =========================
    def to_dict(self) -> Dict[str, Any]:
        return {
            "logs": self.logs,
            "alerts": self.alerts,
            "system_state": self.system_state,
            "steps": self.steps,
            "investigation_history": self.investigation_history,
        }