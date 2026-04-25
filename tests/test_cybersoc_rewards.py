"""Reward-shaping correctness tests."""

import pytest

from cybersoc_arena import CyberSOCEnv
from cybersoc_arena.rewards import STEP_CLIP


def test_step_reward_within_clip():
    env = CyberSOCEnv()
    env.reset(scenario_type="phishing_lateral", seed=7)
    for _ in range(3):
        _, reward, done, _ = env.step({"action_type": "investigate_ip"})
        assert -STEP_CLIP <= reward <= STEP_CLIP
        if done:
            break


def test_correct_attacker_id_yields_positive_reward():
    env = CyberSOCEnv()
    env.reset(scenario_type="phishing_lateral", seed=11)
    # Cheat: read the truth then commit (smoke test only — real agents don't see this)
    truth = env._state.scenario.attacker_ip
    # Build evidence first to avoid premature-decision penalty
    env.step({"action_type": "query_logs", "ip": truth})
    env.step({"action_type": "check_threat_intel", "ip": truth})
    _, reward, done, info = env.step({"action_type": "identify_attacker", "ip": truth})
    assert done
    assert reward > 0
    assert info["correct"] is True


def test_wrong_attacker_id_yields_negative_reward():
    env = CyberSOCEnv()
    env.reset(scenario_type="phishing_lateral", seed=11)
    env.step({"action_type": "query_logs"})
    env.step({"action_type": "check_threat_intel"})
    _, reward, done, info = env.step({"action_type": "identify_attacker", "ip": "1.2.3.4"})
    assert done
    assert reward < 0
    assert info["correct"] is False


def test_close_benign_on_benign_scenario_is_rewarded():
    env = CyberSOCEnv()
    env.reset(scenario_type="benign_scan", seed=22)
    # Gather one piece of evidence first
    obs = env.reset(scenario_type="benign_scan", seed=22)
    ips = obs["asset_inventory"]["visible_ips"]
    if ips:
        env.step({"action_type": "check_threat_intel", "ip": ips[0]})
    _, reward, done, info = env.step({
        "action_type": "close_as_benign",
        "summary": "scanner only, no harm",
    })
    assert done
    # Either correct close (+1.2) or premature close (penalty) — make sure we
    # don't get the worst case (closing a malicious as benign).
    assert reward > -1.0


def test_premature_terminal_action_penalty():
    env = CyberSOCEnv()
    env.reset(scenario_type="multi_stage_chain", seed=99)
    _, reward, done, info = env.step({"action_type": "identify_attacker", "ip": "10.0.0.1"})
    # Premature decision penalty is -0.30 on top of wrong-id -1.5 (clipped to STEP_CLIP)
    assert done
    assert reward < 0


def test_repeat_action_penalised():
    env = CyberSOCEnv()
    obs = env.reset(scenario_type="data_exfiltration", seed=5)
    ip = obs["asset_inventory"]["visible_ips"][0]
    _, r1, _, _ = env.step({"action_type": "query_logs", "ip": ip})
    _, r2, _, _ = env.step({"action_type": "query_logs", "ip": ip})
    # The second invocation should be strictly worse because of repeat penalty
    # AND because no new evidence is revealed.
    assert r2 < r1
