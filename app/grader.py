from typing import Dict, Any
def grade_state(state: Dict[str, Any]) -> float:
    sys_state = state.get("system_state", {})
    task = state.get("current_task", {})
    task_id = task.get("id")

    identified = sys_state.get("identified_attacker")
    blocked_ips = sys_state.get("blocked_ips", [])
    threat = sys_state.get("threat_level", 1.0)

    attackers = state.get("attackers", [])

    score = 0.0

    # 🟢 EASY — must correctly identify attacker
    if task_id == "easy":
        return 1.0 if identified in attackers else 0.0

    # 🟡 MEDIUM — must correctly identify AND block attacker
    elif task_id == "medium":
        if identified in attackers:
            score += 0.5
        if identified in attackers and identified in blocked_ips:
            score += 0.5
        return min(score, 1.0)

    # 🔴 HARD — already correct (keep your version)
    elif task_id == "hard":
        false_positives = [ip for ip in blocked_ips if ip not in attackers]
        if false_positives:
            return 0.0

        if identified in attackers:
            score += 0.4

        if identified in attackers and identified in blocked_ips:
            score += 0.4

        if threat < 0.2:
            score += 0.2

        return max(0.0, min(score, 1.0))

    return 0.0