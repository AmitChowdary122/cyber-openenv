# app/reward.py
"""
Reward function that encourages investigation, correlation, and careful blocking.
Penalises blind guessing and false positives. Designed to make the hard task
require genuine multi‑step reasoning.
"""
from app.models import Reward
from typing import Dict, Any

def compute_reward(action_type: str, parameters: Dict[str, Any], state: Any) -> Reward:
    """
    Compute reward for a single step.
    - Step penalty: -0.05 (discourages brute‑force)
    - Large positive for correct block (+0.5 + bonus)
    - Severe penalty for wrong block (-0.6)
    - Small positive for investigation (+0.05 if real attacker, +0.01 otherwise)
    - Bonus for correlation (+0.02 if identify after investigation)
    - Extra penalty for blocking without investigation (-0.05)
    - False positive escalation penalty (-0.1)
    """
    reward_val = 0.0

    # Extract state
    if hasattr(state, 'system_state'):
        system_state = state.system_state
        suspicion_scores = getattr(state, 'suspicion_scores', {})
        attackers = getattr(state, 'attackers', [])
        investigation_history = getattr(state, 'investigation_history', {})
    else:
        system_state = state.get('system_state', {})
        suspicion_scores = state.get('suspicion_scores', {})
        attackers = state.get('attackers', [])
        investigation_history = state.get('investigation_history', {})

    # Step penalty
    reward_val -= 0.05

    # Action‑specific rewards
    if action_type == "analyze_log":
        reward_val += 0.1
        if system_state.get("threat_level", 0.0) > 0.5:
            reward_val += 0.02

    elif action_type == "identify_attacker":
        ip = parameters.get("ip")
        if ip and ip in attackers:
            reward_val += 0.4 + 0.1   # identification reward
        else:
            reward_val -= 0.5

    elif action_type == "block_ip":
        ip = parameters.get("ip")
        identified = system_state.get("identified_attacker")
        if ip and ip in attackers:
            reward_val += 0.5   # correct block
            if identified == ip:
                reward_val += 0.2   # bonus if identified first
        else:
            reward_val -= 0.6   # wrong block – heavy penalty

    elif action_type == "quarantine_system":
        reward_val += 0.2
        if system_state.get("threat_level", 0.0) > 0.3:
            reward_val += 0.3

    elif action_type == "investigate_ip":
        ip = parameters.get("ip")
        if ip:
            inv_count = investigation_history.get(ip, 0)
            if ip in attackers:
                reward_val += 0.05   # investigating real attacker is valuable
            else:
                reward_val += 0.01   # still some benefit (elimination)
            if inv_count > 1:
                reward_val -= 0.02   # don't investigate same IP repeatedly
        else:
            reward_val -= 0.1

    else:
        reward_val -= 0.2

    # Reasoning bonuses
    # 1. Consistent investigation when threat is high
    if action_type in ("analyze_log", "investigate_ip") and system_state.get("threat_level", 0.0) > 0.3:
        reward_val += 0.03

    # 2. Correlation reward: identify attacker after having investigated at least one IP
    if action_type == "identify_attacker":
        ip = parameters.get("ip")
        if ip and ip in attackers and investigation_history:
            reward_val += 0.02

    # Penalty for blocking without any prior investigation
    if action_type == "block_ip":
        ip = parameters.get("ip")
        if ip and ip not in attackers and investigation_history.get(ip, 0) == 0:
            reward_val -= 0.05

    # Extra false positive escalation penalty
    if action_type == "block_ip" and ip and ip not in attackers:
        reward_val -= 0.1

    # Legacy shaping (kept for easy/medium)
    if system_state.get("identified_attacker"):
        ip = system_state["identified_attacker"]
        if suspicion_scores.get(ip, 0.0) > 0.6:
            reward_val += 0.05

    benign_ips = [ip for ip in suspicion_scores if ip not in attackers]
    for ip in benign_ips:
        if suspicion_scores.get(ip, 0.0) > 0.5:
            reward_val -= 0.02

    if len(attackers) > 2 and action_type == "analyze_log":
        reward_val += 0.02

    return Reward(value=reward_val)