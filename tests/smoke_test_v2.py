"""Final 8-scenario smoke test (post-upgrade)."""
from cybersoc_arena.scenarios import generate_scenario, SCENARIO_TYPES
from cybersoc_arena import CurriculumEnv
from cybersoc_arena.curriculum import TIERS


def main():
    # 1) All 8 scenarios generate cleanly with sane budgets and evidence sizes.
    assert len(SCENARIO_TYPES) == 8, f"expected 8 scenarios, got {len(SCENARIO_TYPES)}"
    for st in SCENARIO_TYPES:
        sc = generate_scenario(st, seed=42)
        assert sc.step_budget >= 6, f"{st} budget too low: {sc.step_budget}"
        assert len(sc.evidence) >= 5, f"{st} needs more evidence: {len(sc.evidence)}"
        print(f"OK  {st:24s}  evidence={len(sc.evidence):>2}  budget={sc.step_budget:>2}")

    # 2) Curriculum: 6 tiers, Elite has all 8.
    assert len(TIERS) == 6, f"expected 6 tiers, got {len(TIERS)}"
    assert len(TIERS[5].scenarios) == 8, \
        f"Elite tier has {len(TIERS[5].scenarios)} scenarios, expected 8"
    print(f"OK  Elite Hunter scenarios: {len(TIERS[5].scenarios)}")

    # 3) Adversarial Elite tier injects decoys.
    env = CurriculumEnv(start_tier=5, adversarial=True, seed=42)
    obs = env.reset()
    assert obs.get("adversarial_mode") is True
    assert len(obs.get("adversarial_decoys", [])) >= 2
    print(f"OK  Adversarial Elite tier: {env.tier_name} | decoys: "
          f"{obs['adversarial_decoys']}")

    # 4) Curriculum metrics expose adversarial flag.
    m = env.curriculum_metrics()
    assert m["adversarial_mode"] is True
    assert len(m["available_scenarios"]) == 8
    print("OK  curriculum_metrics adversarial_mode=True, 8 scenarios available")

    print("\nALL 8-SCENARIO CHECKS PASSED")


if __name__ == "__main__":
    main()
