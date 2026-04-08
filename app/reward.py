# app/reward.py
"""
Reward function that encourages investigation, correlation, and careful blocking.
Penalises blind guessing and false positives. Designed to make the hard task
require genuine multi-step reasoning.
"""
from app.models import Reward
from typing import Dict, Any


def compute_reward(action_type: str, parameters: Dict[str, Any], state: Any) -> Reward:
    reward_val = 0.0

    # Extract state safely
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

    # =========================
    # ACTION REWARDS
    # =========================

    if action_type == "analyze_log":
        reward_val += 0.1
        if system_state.get("threat_level", 0.0) > 0.5:
            reward_val += 0.02

    elif action_type == "identify_attacker":
        ip = parameters.get("ip")
        if ip and ip in attackers:
            reward_val += 0.5
            if investigation_history:
                reward_val += 0.02  # correlation bonus
        else:
            reward_val -= 0.5

    elif action_type == "block_ip":
        ip = parameters.get("ip")
        identified = system_state.get("identified_attacker")

        # 🔥 PATCH 3: FORCE INVESTIGATION
        if ip and investigation_history.get(ip, 0) == 0:
            reward_val -= 0.2  # heavy penalty for blind blocking

        if ip and ip in attackers:
            reward_val += 0.5
            if identified == ip:
                reward_val += 0.2
        else:
            reward_val -= 0.6  # wrong block

        # extra false positive penalty
        if ip and ip not in attackers:
            reward_val -= 0.1

    elif action_type == "quarantine_system":
        reward_val += 0.2
        if system_state.get("threat_level", 0.0) > 0.3:
            reward_val += 0.3

    elif action_type == "investigate_ip":
        ip = parameters.get("ip")
        if ip:
            inv_count = investigation_history.get(ip, 0)

            # 🔥 BOOST INVESTIGATION VALUE
            if ip in attackers:
                reward_val += 0.08
            else:
                reward_val += 0.02

            # discourage repeated spam investigation
            if inv_count > 1:
                reward_val -= 0.02
        else:
            reward_val -= 0.1

    else:
        reward_val -= 0.2

    # =========================
    # REASONING BONUSES
    # =========================

    if action_type in ("analyze_log", "investigate_ip") and system_state.get("threat_level", 0.0) > 0.3:
        reward_val += 0.03

    # bonus if identified attacker has strong suspicion
    if system_state.get("identified_attacker"):
        ip = system_state["identified_attacker"]
        if suspicion_scores.get(ip, 0.0) > 0.6:
            reward_val += 0.05

    # penalize high suspicion on benign IPs
    benign_ips = [ip for ip in suspicion_scores if ip not in attackers]
    for ip in benign_ips:
        if suspicion_scores.get(ip, 0.0) > 0.5:
            reward_val -= 0.02

    # slight bonus for complex scenarios
    if len(attackers) > 2 and action_type == "analyze_log":
        reward_val += 0.02

    return Reward(value=reward_val)
