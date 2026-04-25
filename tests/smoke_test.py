"""Quick end-to-end smoke test."""
from cybersoc_arena import CurriculumEnv, CyberSOCEnv
from cybersoc_arena.scenarios import generate_scenario


def main():
    env = CurriculumEnv(seed=42)
    obs = env.reset()
    assert "curriculum" in obs
    assert env.tier == 0
    obs, r, done, info = env.step({"action_type": "query_logs", "ip": "10.0.0.1"})
    assert "curriculum_tier" in info
    print("CurriculumEnv OK | Tier:", env.tier_name)

    sc = generate_scenario("long_horizon_apt", seed=1)
    assert sc.step_budget == 20
    assert len(sc.target_hosts) == 5
    assert len(sc.evidence) >= 10
    print("long_horizon_apt OK | Evidence items:", len(sc.evidence))

    env2 = CurriculumEnv(seed=0, start_tier=5)
    obs = env2.reset()
    print("Elite tier scenario:", obs.get("initial_alert", "")[:60])
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
