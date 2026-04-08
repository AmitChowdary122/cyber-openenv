from typing import Dict, Any

def grade_state(state: Dict[str, Any]) -> float:
    #SAFETY: handle empty or invalid state
    if not state or not state.get("system_state"):
        return 0.05

    sys_state = state.get("system_state", {})
    task = state.get("current_task", {})
    task_id = task.get("id")

    identified = sys_state.get("identified_attacker")
    blocked_ips = sys_state.get("blocked_ips", [])
    threat = sys_state.get("threat_level", 1.0)

    attackers = state.get("attackers", [])

    score = 0.0

    # EASY
    if task_id == "easy":
        if identified in attackers:
            score = 0.95
        else:
            score = 0.05

    # MEDIUM
    elif task_id == "medium":
        if identified in attackers:
            score += 0.4
        if identified in attackers and identified in blocked_ips:
            score += 0.5

        if score == 0.0:
            score = 0.05

    # HARD
    elif task_id == "hard":
        false_positives = [ip for ip in blocked_ips if ip not in attackers]

        if false_positives:
            score = 0.05
        else:
            if identified in attackers:
                score += 0.3

            if identified in attackers and identified in blocked_ips:
                score += 0.4

            if threat < 0.2:
                score += 0.2

        if score == 0.0:
            score = 0.05

    else:
        score = 0.05

    # FINAL CLAMP (ABSOLUTE SAFETY)
    if score <= 0.0:
        score = 0.05
    elif score >= 1.0:
        score = 0.95

    return float(score)
