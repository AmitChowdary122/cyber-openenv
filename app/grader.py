from typing import Dict, Any

def _clamp(score: float) -> float:
    """
    Ensure score is strictly between (0, 1)
    """
    eps = 1e-3
    if score <= 0.0:
        return eps
    if score >= 1.0:
        return 1.0 - eps
    return score


def grade_state(state: Dict[str, Any]) -> float:
    sys_state = state.get("system_state", {})
    task = state.get("current_task", {})
    task_id = task.get("id")

    identified = sys_state.get("identified_attacker")
    blocked_ips = sys_state.get("blocked_ips", [])
    threat = sys_state.get("threat_level", 1.0)

    attackers = state.get("attackers", [])

    score = 0.0

    # 🟢 EASY
    if task_id == "easy":
        score = 1.0 if identified in attackers else 0.0
        return _clamp(score)

    # 🟡 MEDIUM
    elif task_id == "medium":
        if identified in attackers:
            score += 0.5
        if identified in attackers and identified in blocked_ips:
            score += 0.5
        return _clamp(score)

    # 🔴 HARD
    elif task_id == "hard":
        false_positives = [ip for ip in blocked_ips if ip not in attackers]

        if false_positives:
            score = 0.0
        else:
            if identified in attackers:
                score += 0.4

            if identified in attackers and identified in blocked_ips:
                score += 0.4

            if threat < 0.2:
                score += 0.2

        return _clamp(score)

    return _clamp(0.0)
