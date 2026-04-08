# app/tasks.py
TASKS = [
    {
        "id": "easy",
        "goal": "Identify the attacker IP from logs",
        "success_condition": "identified_attacker is not None"
    },
    {
        "id": "medium",
        "goal": "Identify and block the attacker",
        "success_condition": "identified_attacker in blocked_ips"
    },
    {
        "id": "hard",
        "goal": (
            "You are facing 6–8 suspicious IPs. Only ONE is the real attacker. "
            "Two are strong decoys (high suspicion but benign). The rest are benign. "
            "Logs overlap – no single log reveals the attacker. "
            "You must investigate IPs (investigate_ip), correlate behavior across steps, "
            "then block the correct attacker. "
            "You have only 5 steps. Wrong block ends the episode with a large penalty. "
            "Random guessing (blocking without investigation) incurs an extra penalty."
        ),
        "success_condition": "identified_attacker in blocked_ips and threat_level < 0.1 and no false positives blocked"
    }
]