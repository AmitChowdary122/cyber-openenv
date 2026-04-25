"""
demo_long_horizon.py — guided 20-step walk through the long_horizon_apt
APT-41 kill chain, narrating each phase.
"""
from cybersoc_arena import CyberSOCEnv
import json


PHASES = [
    "Phase 1: Recon (threat-intel on attacker IP)",
    "Phase 1: Recon (mail logs from attacker IP)",
    "Phase 2: Foothold (EDR scan on dmz_host)",
    "Phase 2: Foothold (web logs on dmz_host)",
    "Phase 3: Discovery (EDR scan on jump_host)",
    "Phase 3: Discovery (lateral logs on jump_host)",
    "Phase 4: DC Compromise (EDR scan on dc_host)",
    "Phase 4: Correlation (kerberos chain on dc_host)",
    "Phase 5: Exfiltration (db_host DNS tunnel)",
    "Phase 5: C2 intel (DGA domain on c2_ip)",
    "Phase 5: deepen — db endpoint scan",
    "Phase 4: deepen — golden ticket on jump",
    "Phase 2: deepen — webshell timeline on dmz",
]


def main():
    print("=" * 72)
    print("LONG-HORIZON APT DEMO — 20 steps, 5 phases, 5 hosts")
    print("=" * 72)

    env = CyberSOCEnv()
    obs = env.reset(scenario_type="long_horizon_apt", seed=7)

    ips = list(obs["known_entities"]["ips"])
    hosts = list(obs["known_entities"]["hosts"])
    print(f"\nALERT:    {obs['initial_alert']}")
    print(f"SEVERITY: {obs['alert_severity']}")
    print(f"BUDGET:   {obs['step_budget']} steps")
    print(f"IPs:      {ips}")
    print(f"HOSTS:    {hosts}")
    print(f"Ground truth attacker (grader-only): {env.scenario.attacker_ip}")
    print(f"Ground truth C2     (grader-only):   {env.scenario.c2_ip}")

    # Map "expected target" per planned step using the actual scenario hosts/ips.
    # APT scenario hosts: [dmz_host, jump_host, dc_host, db_host, hr_host]
    dmz, jump, dc, db, hr = hosts
    attacker_ip = env.scenario.attacker_ip
    c2_ip = env.scenario.c2_ip

    plan = [
        ("check_threat_intel", "ip",   attacker_ip),
        ("query_logs",         "ip",   attacker_ip),
        ("inspect_endpoint",   "host", dmz),
        ("query_logs",         "host", dmz),
        ("inspect_endpoint",   "host", jump),
        ("query_logs",         "host", jump),
        ("inspect_endpoint",   "host", dc),
        ("correlate_events",   "host", dc),
        ("query_logs",         "host", db),
        ("check_threat_intel", "ip",   c2_ip),
        ("inspect_endpoint",   "host", db),
        ("correlate_events",   "host", jump),
        ("query_logs",         "host", dmz),
    ]

    total = 0.0
    info = {}
    done = False

    for i, (tool, kind, target) in enumerate(plan):
        if done:
            break
        action = {"action_type": tool, kind: target}
        obs, r, done, info = env.step(action)
        total += r
        phase = PHASES[i] if i < len(PHASES) else "(extra)"
        print(f"\nStep {i+1:2d} [{phase}]")
        print(f"  Action:  {json.dumps(action)}")
        print(f"  Reward:  {r:+.3f}   cum={total:+.3f}")
        print(f"  Result:  {obs['last_action_result'][:160]}")
        print(f"  Evidence so far: total={info.get('evidence_collected')}, "
              f"confirming={info.get('confirming_evidence')}")
        if done:
            print(f"  >> Episode terminated early at step {i+1}.")

    if not done:
        # Final action: identify_attacker with the spearphish/webshell author
        action = {"action_type": "identify_attacker", "ip": attacker_ip}
        obs, r, done, info = env.step(action)
        total += r
        print(f"\nStep {env._step:2d} [Decision]")
        print(f"  Action:  {json.dumps(action)}")
        print(f"  Reward:  {r:+.3f}   cum={total:+.3f}")
        print(f"  Result:  {obs['last_action_result'][:200]}")

    print(f"\n{'-'*72}")
    verdict = info.get("verdict") or {}
    print(f"FINAL VERDICT: action_type={verdict.get('action_type')}  "
          f"target={verdict.get('target')}  "
          f"premature={verdict.get('premature')}  "
          f"correct={info.get('correct')}")
    print(f"TOTAL REWARD:  {total:+.3f}")
    print(f"STEPS USED:    {env._step} / {obs['step_budget']}")
    print(f"EVIDENCE:      collected={info.get('evidence_collected')}, "
          f"confirming={info.get('confirming_evidence')}")
    print(f"{'-'*72}")

    print("\nFull revealed evidence list:")
    for j, line in enumerate(obs["revealed_evidence"], start=1):
        print(f"  {j:>2}. {line}")


if __name__ == "__main__":
    main()
