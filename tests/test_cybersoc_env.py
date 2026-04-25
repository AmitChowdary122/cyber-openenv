"""Smoke tests for the CyberSOC environment."""

import pytest

from cybersoc_arena import CyberSOCEnv, SCENARIO_TYPES, generate_scenario, parse_action
from cybersoc_arena.actions import ALL_ACTIONS, INVESTIGATIVE_ACTIONS, TERMINAL_ACTIONS


def test_reset_returns_observation_with_required_keys():
    env = CyberSOCEnv()
    obs = env.reset(seed=0)
    for key in ("alert", "step", "remaining_steps", "step_budget",
                "asset_inventory", "evidence_collected", "available_actions",
                "goal"):
        assert key in obs


def test_step_returns_four_tuple():
    env = CyberSOCEnv()
    env.reset(seed=0)
    out = env.step({"action_type": "investigate_ip"})
    assert isinstance(out, tuple) and len(out) == 4
    obs, reward, done, info = out
    assert isinstance(reward, float)
    assert isinstance(done, bool)


@pytest.mark.parametrize("scenario", SCENARIO_TYPES)
def test_each_scenario_can_run_to_termination(scenario):
    env = CyberSOCEnv()
    env.reset(scenario_type=scenario, seed=42)
    # Investigative loop until budget, then a terminal action
    done = False
    steps = 0
    while not done and steps < 30:
        steps += 1
        obs, r, done, info = env.step({"action_type": "investigate_ip"})
    # If still not done, force terminal close
    if not done:
        obs, r, done, info = env.step({"action_type": "close_as_benign", "summary": "x"})
    assert done


def test_terminal_action_ends_episode():
    env = CyberSOCEnv()
    env.reset(scenario_type="benign_scan", seed=1)
    obs, r, done, info = env.step({"action_type": "close_as_benign", "summary": "test"})
    assert done is True
    assert info.get("terminal") == "close_as_benign"


def test_double_step_after_done_raises():
    env = CyberSOCEnv()
    env.reset(seed=0)
    env.step({"action_type": "close_as_benign", "summary": "x"})
    with pytest.raises(RuntimeError):
        env.step({"action_type": "investigate_ip"})


def test_action_set_partition():
    """ALL_ACTIONS = INVESTIGATIVE union TERMINAL, no overlap."""
    assert INVESTIGATIVE_ACTIONS.isdisjoint(TERMINAL_ACTIONS)
    assert INVESTIGATIVE_ACTIONS | TERMINAL_ACTIONS == ALL_ACTIONS


def test_parse_action_from_dict():
    a = parse_action({"action_type": "investigate_ip", "ip": "10.0.0.1"})
    assert a.action_type == "investigate_ip"
    assert a.ip == "10.0.0.1"


def test_parse_action_from_json_string():
    a = parse_action('{"action_type": "identify_attacker", "ip": "10.0.0.5"}')
    assert a.action_type == "identify_attacker"
    assert a.ip == "10.0.0.5"


def test_parse_action_extracts_from_prose():
    a = parse_action(
        "I think we should investigate_ip 10.0.0.7 because of the alerts."
    )
    assert a.action_type == "investigate_ip"
    assert a.ip == "10.0.0.7"


def test_state_snapshot_includes_terminal_flags():
    env = CyberSOCEnv()
    env.reset(scenario_type="phishing_lateral", seed=123)
    env.step({"action_type": "escalate_incident", "summary": "phishing lateral c2"})
    s = env.state()
    assert s["done"] is True
    assert s["terminal_action"] == "escalate_incident"
