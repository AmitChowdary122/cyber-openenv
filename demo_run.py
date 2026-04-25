"""Demo: run one heuristic-driven episode of every CyberSOC Arena scenario.

Usage:
    python demo_run.py [--seed 42] [--out training_runs/demo_all.log]

Prints (and tees to a log file) the full step-by-step trajectory for each of
the 6 scenario archetypes, so judges can read the dialogue between agent and
environment without spinning anything up.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cybersoc_arena import (  # noqa: E402
    CyberAction,
    CyberSOCEnv,
    SCENARIO_TYPES,
)


def short(s: str, n: int = 90) -> str:
    s = s.replace("\n", " ")
    return (s[:n - 1] + "...") if len(s) > n else s


def heuristic_action(obs) -> CyberAction:
    """Tiny SOC-analyst heuristic. Investigate first IP, host, then attribute."""
    ips = list(obs.asset_inventory.visible_ips) or [""]
    hosts = list(obs.asset_inventory.hosts) or [""]
    used = {h.target.lower() for h in obs.action_history}

    if obs.evidence_count < 4:
        # Cycle: query_logs(ip0) -> inspect(host0) -> query(ip1) -> threat_intel(ip2) -> ...
        cycle = [
            ("query_logs", "ip", ips[0 % len(ips)]),
            ("inspect_endpoint", "host", hosts[0 % len(hosts)]),
            ("query_logs", "ip", ips[1 % len(ips)]),
            ("check_threat_intel", "ip", ips[2 % len(ips)] if len(ips) > 2 else ips[0]),
            ("inspect_endpoint", "host", hosts[1 % len(hosts)] if len(hosts) > 1 else hosts[0]),
            ("correlate_events", "entity", ips[0]),
        ]
        a_type, kw, target = cycle[obs.step % len(cycle)]
        return CyberAction(action_type=a_type, **{kw: target})

    # Decide based on alert text + evidence
    if "scan" in obs.alert.summary.lower() and obs.evidence_count >= 3:
        return CyberAction(
            action_type="close_as_benign",
            summary="Internet scanner / red-team activity, no impact.",
        )

    # Default: identify the IP that appears most across attacker-style findings.
    import re, collections
    ip_re = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
    counts: collections.Counter = collections.Counter()
    for e in obs.evidence_collected:
        text = e.finding.lower()
        if any(k in text for k in ("exploit", "exfil", "phish", "lateral",
                                    "ransom", "stager", "credential",
                                    "kerberos", "persist", "ja3")):
            for ip in ip_re.findall(e.finding):
                counts[ip] += 1
    if counts:
        top_ip = counts.most_common(1)[0][0]
    else:
        top_ip = ips[0]
    return CyberAction(action_type="identify_attacker", ip=top_ip)


def run_one(scenario: str, seed: int, lines: List[str]) -> None:
    env = CyberSOCEnv()
    obs = env.reset(seed=seed, scenario_type=scenario)
    lines.append("")
    lines.append("=" * 90)
    lines.append(f"SCENARIO: {scenario}   (seed={seed})")
    lines.append("=" * 90)
    lines.append(f"ALERT [{obs.alert.severity.upper()}]: {short(obs.alert.summary, 200)}")
    lines.append(f"  hosts:       {', '.join(obs.asset_inventory.hosts)}")
    lines.append(f"  visible IPs: {', '.join(obs.asset_inventory.visible_ips[:8])}"
                 + ("..." if len(obs.asset_inventory.visible_ips) > 8 else ""))
    lines.append(f"  step budget: {obs.step_budget}")
    lines.append("")
    total = 0.0
    while not obs.done:
        a = heuristic_action(obs)
        target = a.ip or a.host or a.entity or short(a.summary or "", 30)
        obs = env.step(a)
        total += obs.reward
        bd = obs.info.get("reward_breakdown", {})
        bd_s = "  ".join(f"{k}={v:+.2f}" for k, v in bd.items())
        lines.append(
            f"  step {obs.step:2d}/{obs.step_budget:2d}  "
            f"{a.action_type:22s} {short(target, 22):22s}  "
            f"reward={obs.reward:+.3f}  evidence={obs.evidence_count}  [{bd_s}]"
        )
        if obs.evidence_collected and obs.step <= 6:
            for ev in obs.evidence_collected[-1:]:
                lines.append(f"      EVIDENCE: {short(ev.finding, 120)}")
    st = env.state
    lines.append("")
    lines.append(f"  RESULT: terminal={st.terminal_action} correct={st.terminal_correct} "
                 f"  cum_reward={total:+.3f}  evidence_total={st.evidence_count}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="training_runs/demo_all_scenarios.log")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    lines: List[str] = []
    lines.append("CyberSOC Arena -- one heuristic episode per scenario")
    lines.append(f"seed={args.seed}  scenarios={len(SCENARIO_TYPES)}")
    for scen in SCENARIO_TYPES:
        run_one(scen, args.seed, lines)
    text = "\n".join(lines)
    print(text)
    with open(args.out, "w") as f:
        f.write(text + "\n")
    print(f"\n[written] {args.out}")


if __name__ == "__main__":
    main()
