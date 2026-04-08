# app/grader.py
from typing import Dict, Any


def grade_state(state: Dict[str, Any]) -> float:
    sys_state = state.get("system_state", {})
    task = state.get("current_task", {})
    task_id = task.get("id")

    identified = sys_state.get("identified_attacker")
    blocked_ips = sys_state.get("blocked_ips", [])
    threat = sys_state.get("threat_level", 1.0)

    score = 0.0

    # 🟢 EASY — identify attacker (unchanged)
    if task_id == "easy":
        return 1.0 if identified else 0.0

    # 🟡 MEDIUM — identify + block (unchanged)
    elif task_id == "medium":
        if identified:
            score += 0.5
        if identified and identified in blocked_ips:
            score += 0.5
        return min(score, 1.0)

    # 🔴 HARD — full mitigation with penalty for false positives
    elif task_id == "hard":
        # Retrieve list of true attackers from state (added in env.state())
        attackers = state.get("attackers", [])

        # Enforce strict correctness: any false positive results in failure
        false_positives = [ip for ip in blocked_ips if ip not in attackers]
        if false_positives:
            return 0.0

        # Correct identification
        if identified:
            score += 0.4

        # Correct blocking: identified attacker is in blocked_ips
        if identified and identified in blocked_ips:
            score += 0.4

        # Low threat bonus (<0.2)
        if threat < 0.2:
            score += 0.2

        # Clamp to [0, 1]
        return max(0.0, min(score, 1.0))

    # Fallback for unknown task
    return 0.0