"""Demo: walk through a single long_horizon_apt episode step by step.

This is the marquee demo for Theme 2 (Super Long-Horizon Planning): a 20-step
APT investigation across 5 hosts (edge -> workstation -> file server -> DB ->
egress proxy) with three high-quality decoys. The scripted heuristic deliberately
mirrors the SOC-analyst playbook (recon -> initial access -> persistence ->
lateral -> exfil) so judges can read the trajectory like a movie.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cybersoc_arena import CyberAction, CyberSOCEnv  # noqa: E402


def short(s: str, n: int = 110) -> str:
    s = s.replace("\n", " ")
    return (s[:n - 1] + "...") if len(s) > n else s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=314)
    ap.add_argument("--out", default="training_runs/demo_long_horizon.log")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    env = CyberSOCEnv()
    obs = env.reset(seed=args.seed, scenario_type="long_horizon_apt")
    ips = list(obs.asset_inventory.visible_ips)
    hosts = list(obs.asset_inventory.hosts)
    # The first IP in visible_ips is typically the attacker's (203.0.113.x)
    attacker_guess = next((ip for ip in ips if ip.startswith("203.0.113")), ips[0])

    lines: List[str] = []
    lines.append("CyberSOC Arena -- long_horizon_apt walkthrough")
    lines.append(f"seed={args.seed}  step_budget={obs.step_budget}  hosts={len(hosts)} "
                 f"visible_ips={len(ips)}")
    lines.append("=" * 100)
    lines.append(f"ALERT [{obs.alert.severity.upper()}]: {short(obs.alert.summary, 200)}")
    lines.append(f"  Hosts:       {', '.join(hosts)}")
    lines.append(f"  Visible IPs: {', '.join(ips)}")
    lines.append("")

    # Scripted SOC-analyst playbook. Ordered to mirror the kill chain.
    plan: List[CyberAction] = [
        # Phase 1: Recon - look at the suspect IP across edge logs + threat intel
        CyberAction(action_type="query_logs",         ip=attacker_guess),
        CyberAction(action_type="check_threat_intel", ip=attacker_guess),
        # Phase 2: Initial access - inspect the workstation
        CyberAction(action_type="inspect_endpoint",   host=hosts[1]),
        CyberAction(action_type="query_logs",         ip=hosts[1]),  # may be wrong; see env response
        CyberAction(action_type="query_logs",         ip=ips[3] if len(ips) > 3 else ips[0]),
        # Phase 3: Persistence
        CyberAction(action_type="inspect_endpoint",   host=hosts[1]),
        # Phase 4: Lateral movement - file server, DB
        CyberAction(action_type="query_logs",         ip=hosts[2]),
        CyberAction(action_type="inspect_endpoint",   host=hosts[3]),
        # Correlate the chain
        CyberAction(action_type="correlate_events",   entity=hosts[1]),
        # Phase 5: Exfil - egress proxy
        CyberAction(action_type="query_logs",         ip=hosts[4]),
        CyberAction(action_type="correlate_events",   entity=attacker_guess),
        # Decoys to test discipline (intentional!)
        CyberAction(action_type="check_threat_intel", ip=ips[-1]),
        # Final attribution
        CyberAction(action_type="identify_attacker",  ip=attacker_guess),
    ]

    total = 0.0
    for i, act in enumerate(plan):
        if obs.done:
            break
        target = act.ip or act.host or act.entity or short(act.summary or "", 28)
        obs = env.step(act)
        total += obs.reward
        bd = obs.info.get("reward_breakdown", {})
        bd_s = "  ".join(f"{k}={v:+.2f}" for k, v in bd.items())
        lines.append(
            f"  step {obs.step:2d}/{obs.step_budget:2d}  "
            f"{act.action_type:22s} {short(target, 24):24s}  "
            f"reward={obs.reward:+.3f}  evidence={obs.evidence_count}  [{bd_s}]"
        )
        if obs.evidence_collected and obs.evidence_count > i:
            for ev in obs.evidence_collected[-1:]:
                lines.append(f"      EVIDENCE: {short(ev.finding, 130)}")

    st = env.state
    lines.append("")
    lines.append("=" * 100)
    lines.append(
        f"FINAL: terminal={st.terminal_action} correct={st.terminal_correct}  "
        f"cum_reward={total:+.3f}  evidence_revealed={st.evidence_count}/"
        f"attacker_evidence={st.attacker_evidence_count}"
    )
    text = "\n".join(lines)
    print(text)
    with open(args.out, "w") as f:
        f.write(text + "\n")
    print(f"\n[written] {args.out}")


if __name__ == "__main__":
    main()
