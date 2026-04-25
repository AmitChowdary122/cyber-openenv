"""
demo_curriculum.py — watch the agent unlock all 6 curriculum tiers.
Uses the real CurriculumEnv API and a simple investigate-then-decide
heuristic per episode.
"""
from cybersoc_arena import CurriculumEnv

ATTACK_KW = (
    "phishing", "webshell", "lateral", "mimikatz", "ransom", "beacon",
    "exfiltration", "stuffing", "golden", "kill chain", "kill-chain",
    "brute-force", "brute force", "successful login", "failed login",
    "implant", "payload", "apt", "ddos", "backdoor", "dns tunnel",
    "forged", "dcsync", "botnet", "spearphish", "macro", "malicious",
    "exploitation", "compromised", "stolen", "rogue", "anomalous",
    "exported", "outbound https bursts", "powershell", "psexec",
    "team server", "cobalt", "tunnelling", "tunneling", "spoofed",
)
BENIGN_KW = (
    "scanner", "fat-fingered", "no abuse", "legitimate", "normal",
    "test file", "benign", "false positive", "stock", "vendor scanner",
    "vuln-mgmt", "ms 365", "microsoft 365", "cdn edge", "no successful",
)


def episode(env: CurriculumEnv) -> tuple[float, bool, str]:
    obs = env.reset()
    scenario_type = obs["scenario_type"]
    ips = list(obs["known_entities"]["ips"])
    hosts = list(obs["known_entities"]["hosts"])
    queue = []
    for ip in ips: queue.append(("check_threat_intel", ip, "ip"))
    for ip in ips: queue.append(("query_logs", ip, "ip"))
    for h in hosts: queue.append(("inspect_endpoint", h, "host"))
    for h in hosts: queue.append(("query_logs", h, "host"))
    for h in hosts: queue.append(("correlate_events", h, "host"))
    for ip in ips: queue.append(("correlate_events", ip, "ip"))
    tried = set()
    total = 0.0
    info = {}
    done = False
    step = 0
    while not done and step < 40:
        budget = obs["step_budget"]
        revealed = obs.get("revealed_evidence", [])
        if step < budget - 2 and queue:
            while queue:
                tool, tgt, kind = queue.pop(0)
                if (tool, tgt) in tried:
                    continue
                tried.add((tool, tgt))
                a = {"action_type": tool, kind: tgt}
                break
            else:
                a = {"action_type": "list_entities"}
        else:
            scores = {ip: 0 for ip in ips}
            total_a = total_b = 0
            for line in revealed:
                ll = line.lower()
                aa = sum(1 for k in ATTACK_KW if k in ll)
                bb = sum(1 for k in BENIGN_KW if k in ll)
                total_a += aa; total_b += bb
                if aa:
                    for ip in ips:
                        if ip in line:
                            scores[ip] = scores.get(ip, 0) + aa
            if total_a < max(1, total_b):
                a = {"action_type": "mark_benign"}
            elif scores and max(scores.values()) > 0:
                a = {"action_type": "identify_attacker",
                     "ip": max(scores, key=lambda k: scores[k])}
            else:
                a = {"action_type": "escalate_incident"}
        obs, r, done, info = env.step(a)
        total += r
        step += 1
    return total, bool(info.get("correct")), scenario_type


def main():
    print("=" * 72)
    print("CURRICULUM DEMO — agent unlocks tiers as rolling reward crosses thresholds")
    print("=" * 72)
    env = CurriculumEnv(seed=42, start_tier=0)
    last_tier = -1
    for ep in range(1, 121):
        total, ok, scenario_type = episode(env)
        m = env.curriculum_metrics()

        if m["tier_level"] != last_tier:
            print(f"\n>> Now at Tier {m['tier_level']}: {m['tier_name']}")
            last_tier = m["tier_level"]

        if ep <= 5 or ep % 10 == 0 or m["transitions"] and m["transitions"][-1]["at_episode"] == ep:
            print(f"  Ep {ep:3d}  tier={m['tier_level']} ({m['tier_name']:<16s})  "
                  f"scenario={scenario_type:<22s}  reward={total:+.3f}  "
                  f"rolling_mean={m['rolling_mean_reward']:+.3f}  "
                  f"progress={m['progress_to_next']*100:5.1f}%  "
                  f"correct={'Y' if ok else 'N'}")

        if m["transitions"] and m["transitions"][-1]["at_episode"] == ep:
            t = m["transitions"][-1]
            print(f"\n   *** TIER UNLOCK at ep {ep}: "
                  f"{t['from_name']} -> {t['to_name']} "
                  f"(rolling_mean {t['rolling_mean_at_unlock']:+.3f} "
                  f">= threshold {t['unlock_threshold']:.2f}) ***")

    m = env.curriculum_metrics()
    print("\n" + "=" * 72)
    print("FINAL CURRICULUM STATE")
    print("=" * 72)
    print(f"  Final tier:       {m['tier_level']} ({m['tier_name']})")
    print(f"  Episode count:    {m['episode_count']}")
    print(f"  Tier transitions: {len(m['transitions'])}")
    for t in m["transitions"]:
        print(f"    Ep {t['at_episode']:3d}: {t['from_name']:<16s} -> "
              f"{t['to_name']:<16s} rolling={t['rolling_mean_at_unlock']:+.3f} "
              f"thr={t['unlock_threshold']:.2f}")


if __name__ == "__main__":
    main()
