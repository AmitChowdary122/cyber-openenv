"""
demo_run.py — one episode per scenario, fully verbose.
Adapted to the real CyberSOC Arena observation/info schema:
  - obs['known_entities'] = {'ips':[...], 'hosts':[...]}
  - obs['revealed_evidence'] is a List[str]
  - terminal action for false-positive is "mark_benign"
  - info contains verdict, evidence_collected, confirming_evidence, correct
"""
from cybersoc_arena import CyberSOCEnv
from cybersoc_arena.scenarios import SCENARIO_TYPES
import json


def run_episode(scenario_type, seed=42):
    print(f"\n{'='*72}")
    print(f"SCENARIO: {scenario_type.upper()}")
    print(f"{'='*72}")

    env = CyberSOCEnv()
    obs = env.reset(scenario_type=scenario_type, seed=seed)

    ips = obs["known_entities"]["ips"]
    hosts = obs["known_entities"]["hosts"]
    print(f"\nINITIAL ALERT: {obs['initial_alert']}")
    print(f"SEVERITY:      {obs['alert_severity']}")
    print(f"BUDGET:        {obs['step_budget']} steps")
    print(f"KNOWN IPs:     {ips}")
    print(f"KNOWN HOSTS:   {hosts}")

    total_reward = 0.0
    step = 0
    done = False
    info = {}

    # Investigative round-robin (cheap intel/log first), then verdict in last 2 steps.
    queue = []
    for ip in ips:
        queue.append(("check_threat_intel", ip, "ip"))
    for ip in ips:
        queue.append(("query_logs", ip, "ip"))
    for h in hosts:
        queue.append(("inspect_endpoint", h, "host"))
    for h in hosts:
        queue.append(("query_logs", h, "host"))
    for h in hosts:
        queue.append(("correlate_events", h, "host"))
    for ip in ips:
        queue.append(("correlate_events", ip, "ip"))

    while not done:
        budget = obs["step_budget"]
        revealed = obs.get("revealed_evidence", [])

        if step < budget - 2 and queue:
            tool, tgt, kind = queue.pop(0)
            action = {"action_type": tool, kind: tgt}
        else:
            # verdict heuristic: most-attack-keyword IP, else mark_benign
            attack_kw = ("phishing", "webshell", "lateral", "mimikatz",
                         "stuffing", "exfiltration", "beacon", "golden",
                         "brute", "successful login", "implant", "spearphish",
                         "malicious", "exploitation", "compromised", "stolen")
            scores = {ip: 0 for ip in ips}
            attack_total = 0
            for line in revealed:
                ll = line.lower()
                a = sum(1 for k in attack_kw if k in ll)
                attack_total += a
                if a:
                    for ip in ips:
                        if ip in line:
                            scores[ip] = scores.get(ip, 0) + a
            if attack_total == 0 or not ips:
                action = {"action_type": "mark_benign"}
            elif scores and max(scores.values()) > 0:
                best = max(scores, key=lambda k: scores[k])
                action = {"action_type": "identify_attacker", "ip": best}
            else:
                action = {"action_type": "escalate_incident"}

        obs, reward, done, info = env.step(action)
        total_reward += reward
        step += 1

        print(f"\n  STEP {step}")
        print(f"  ACTION:  {json.dumps(action)}")
        print(f"  REWARD:  {reward:+.3f}   cum={total_reward:+.3f}")
        new_ev = obs.get("revealed_evidence", [])
        if new_ev:
            print(f"  RESULT:  {obs['last_action_result'][:140]}")
            print(f"           evidence_collected={info.get('evidence_collected')}, "
                  f"confirming={info.get('confirming_evidence')}")

        if done:
            verdict = info.get("verdict") or {}
            print(f"\n  TERMINAL ACTION: {verdict.get('action_type')}  "
                  f"target={verdict.get('target')}  "
                  f"premature={verdict.get('premature')}")
            print(f"  CORRECT:        {info.get('correct')}")

    print(f"\n{'-'*72}")
    print(f"EPISODE SUMMARY")
    print(f"  Total reward:   {total_reward:+.3f}")
    print(f"  Steps taken:    {step} / {obs['step_budget']}")
    print(f"  Evidence revealed: {len(obs.get('revealed_evidence', []))}")
    print(f"  Outcome:        {'CORRECT' if info.get('correct') else 'WRONG'}")
    print(f"{'-'*72}")
    return total_reward


if __name__ == "__main__":
    print("CyberSOC Arena — running one full episode of each scenario type.")
    rewards = {}
    for scenario in SCENARIO_TYPES:
        rewards[scenario] = run_episode(scenario, seed=42)

    print("\n" + "=" * 72)
    print("ALL-SCENARIOS REWARD TABLE")
    print("=" * 72)
    for sc, r in rewards.items():
        print(f"  {sc:24s} -> total reward {r:+.3f}")
