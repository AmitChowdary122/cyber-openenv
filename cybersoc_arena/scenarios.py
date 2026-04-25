"""Incident scenario generators for CyberSOC Arena.

Each scenario carries:
  - ground truth (attacker IP, attack type, kill-chain stage)
  - an evidence trail (what each investigation action should reveal)
  - decoys (benign entities that look suspicious — false-positive bait)
  - background noise (volume of unrelated alerts)
  - a reference summary keyword set for grading escalation reports

Scenarios are stochastic and parameterised; the same scenario_type produces
different concrete incidents on each call.
"""

from __future__ import annotations

import dataclasses
import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

SCENARIO_TYPES = [
    "benign_scan",
    "phishing_lateral",
    "credential_stuffing",
    "data_exfiltration",
    "multi_stage_chain",
    "long_horizon_apt",
]


# ── Data structures ──────────────────────────────────────────────────────────
@dataclasses.dataclass
class Evidence:
    """A piece of evidence the agent can reveal through investigation.

    source: which action_type unlocks this evidence (e.g. 'query_logs').
    target: the entity (IP or host) the action must be targeted at.
    confirms_attacker: True if this evidence points at the real attacker.
    weight: 0..1 — how strong this evidence is for attribution.
    """
    source: str
    target: str
    description: str
    weight: float
    confirms_attacker: bool


@dataclasses.dataclass
class Scenario:
    scenario_type: str
    is_benign: bool
    attacker_ip: Optional[str]
    attacker_host: Optional[str]
    decoy_ips: List[str]
    benign_ips: List[str]
    target_hosts: List[str]
    evidence: List[Evidence]
    initial_alert: str
    alert_severity: str
    correlation_pairs: List[tuple]   # (entity_a, entity_b) — true correlations
    summary_keywords: List[str]      # required terms in a good escalation summary
    step_budget: int
    background_logs: List[str]       # noise lines

    def all_ips(self) -> List[str]:
        ips = list(self.decoy_ips) + list(self.benign_ips)
        if self.attacker_ip:
            ips.append(self.attacker_ip)
        return ips


# ── Helpers ───────────────────────────────────────────────────────────────────
def _ip(rng: random.Random, prefix_pool: List[str] = None) -> str:
    prefix_pool = prefix_pool or ["10.0", "192.168", "172.16", "203.0.113", "198.51.100"]
    return f"{rng.choice(prefix_pool)}.{rng.randint(1, 254)}.{rng.randint(1, 254)}"


def _host(rng: random.Random, role: str = "ws") -> str:
    return f"{role}-{rng.randint(100, 999):03d}"


def _ts(offset_min: int = 0) -> str:
    t = datetime.utcnow() - timedelta(minutes=offset_min)
    return t.strftime("%Y-%m-%d %H:%M:%S")


def _noise_lines(rng: random.Random, n: int) -> List[str]:
    templates = [
        "[{ts}] [INFO] DHCP renewal on {host}",
        "[{ts}] [INFO] AV signature update on {host}",
        "[{ts}] [INFO] LDAP bind from {ip}",
        "[{ts}] [INFO] proxy GET 200 from {ip}",
        "[{ts}] [WARN] retransmit on {ip} -> {ip2}",
    ]
    out = []
    for _ in range(n):
        tpl = rng.choice(templates)
        out.append(tpl.format(
            ts=_ts(rng.randint(0, 60)),
            host=_host(rng),
            ip=_ip(rng),
            ip2=_ip(rng),
        ))
    return out


# ── Generators ────────────────────────────────────────────────────────────────
def gen_benign_scan(rng: random.Random) -> Scenario:
    """Internet scanner triggers high alert volume but does no harm."""
    decoys = [_ip(rng, ["198.51.100", "203.0.113"]) for _ in range(3)]
    target_hosts = [_host(rng, "edge") for _ in range(2)]
    edge_host = target_hosts[0]
    return Scenario(
        scenario_type="benign_scan",
        is_benign=True,
        attacker_ip=None,
        attacker_host=None,
        decoy_ips=decoys,
        benign_ips=[_ip(rng) for _ in range(2)],
        target_hosts=target_hosts,
        initial_alert=f"IDS: 4,200 SYN packets in 30s from {len(decoys)} external IPs",
        alert_severity="medium",
        correlation_pairs=[],
        summary_keywords=["scan", "benign", "no", "compromise"],
        step_budget=8,
        evidence=[
            Evidence("check_threat_intel", decoys[0],
                     f"ThreatIntel: {decoys[0]} is a known internet scanner (Shodan-like). "
                     "No history of credential abuse or exploit attempts.",
                     weight=0.9, confirms_attacker=False),
            Evidence("query_logs", decoys[0],
                     "PCAP: only TCP-SYN to closed ports 22/80/443/3389. "
                     "Zero connections established. Zero auth attempts.",
                     weight=0.8, confirms_attacker=False),
            Evidence("query_logs", decoys[1],
                     f"Edge logs: {decoys[1]} sent 1,400 SYN packets, all dropped at firewall.",
                     weight=0.6, confirms_attacker=False),
            Evidence("inspect_endpoint", edge_host,
                     "Firewall: connection-tracking shows 0 successful sessions from external scanners.",
                     weight=0.5, confirms_attacker=False),
            Evidence("correlate_events", decoys[0],
                     "Correlation: scan signatures match TCP/IP fingerprint of public scanners. "
                     "No follow-on activity observed.",
                     weight=0.7, confirms_attacker=False),
        ],
        background_logs=_noise_lines(rng, 6),
    )


def gen_phishing_lateral(rng: random.Random) -> Scenario:
    """User clicks phishing -> creds stolen -> attacker pivots laterally."""
    attacker_ip = _ip(rng, ["45.227", "185.220"])  # suspicious ranges
    user_host = _host(rng, "ws")
    pivot_host = _host(rng, "srv")
    decoys = [_ip(rng) for _ in range(2)]  # internal noisy hosts
    return Scenario(
        scenario_type="phishing_lateral",
        is_benign=False,
        attacker_ip=attacker_ip,
        attacker_host=user_host,
        decoy_ips=decoys,
        benign_ips=[_ip(rng) for _ in range(2)],
        target_hosts=[user_host, pivot_host],
        initial_alert=f"EDR: suspicious PowerShell child process on {user_host} after Outlook.exe",
        alert_severity="high",
        correlation_pairs=[(attacker_ip, user_host), (user_host, pivot_host)],
        summary_keywords=["phishing", "lateral", "credential", "pivot", attacker_ip],
        step_budget=10,
        evidence=[
            Evidence("query_logs", user_host,
                     f"Email gw: user opened attachment from spoofed sender 12 min before alert. "
                     f"Outbound C2 to {attacker_ip} from {user_host} on TLS-443 to non-categorised cert.",
                     weight=1.0, confirms_attacker=True),
            Evidence("inspect_endpoint", user_host,
                     "EDR: powershell.exe -enc <base64> spawned by OUTLOOK.EXE. "
                     "AMSI detected reflective DLL load. Mimikatz signature.",
                     weight=1.0, confirms_attacker=True),
            Evidence("check_threat_intel", attacker_ip,
                     f"ThreatIntel: {attacker_ip} is in ProofPoint Emerging Threats - APT-staging. "
                     "Active C2 since 14 days ago.",
                     weight=0.95, confirms_attacker=True),
            Evidence("query_logs", pivot_host,
                     f"AD logs: NTLM auth from {user_host} -> {pivot_host} using svc_backup, "
                     "outside business hours, after credential dump.",
                     weight=0.9, confirms_attacker=True),
            Evidence("correlate_events", user_host,
                     f"Correlation: {attacker_ip} -> {user_host} -> {pivot_host}, "
                     "confirms phishing -> credential theft -> lateral movement.",
                     weight=1.0, confirms_attacker=True),
            Evidence("query_logs", decoys[0],
                     "Logs: backup job failure. Unrelated to incident.",
                     weight=0.2, confirms_attacker=False),
        ],
        background_logs=_noise_lines(rng, 5),
    )


def gen_credential_stuffing(rng: random.Random) -> Scenario:
    """One real attacker hides among many failed-login IPs (residential proxies)."""
    attacker_ip = _ip(rng, ["91.240", "194.165"])
    decoys = [_ip(rng) for _ in range(4)]  # other failed-login sources, all benign
    return Scenario(
        scenario_type="credential_stuffing",
        is_benign=False,
        attacker_ip=attacker_ip,
        attacker_host=None,
        decoy_ips=decoys,
        benign_ips=[_ip(rng) for _ in range(2)],
        target_hosts=[_host(rng, "auth") for _ in range(2)],
        initial_alert=(
            f"Auth: 18,000 failed logins in 1h across SSO from {len(decoys)+1} source IPs. "
            f"One source has a successful auth."
        ),
        alert_severity="high",
        correlation_pairs=[(attacker_ip, "auth-201")],
        summary_keywords=["credential", "stuffing", "successful", attacker_ip],
        step_budget=10,
        evidence=[
            Evidence("query_logs", attacker_ip,
                     f"SSO: {attacker_ip} performed 8,400 failed logins, then 1 SUCCESS for user "
                     "j.smith at 02:14 UTC, followed by token-export action.",
                     weight=1.0, confirms_attacker=True),
            Evidence("query_logs", decoys[0],
                     f"SSO: {decoys[0]} performed 3,100 failed logins, no successes. "
                     "Source: corporate VPN - matches known load test.",
                     weight=0.7, confirms_attacker=False),
            Evidence("check_threat_intel", attacker_ip,
                     f"ThreatIntel: {attacker_ip} appears in Combolist abuse feed. "
                     "Tied to credential-stuffing kits.",
                     weight=0.9, confirms_attacker=True),
            Evidence("check_threat_intel", decoys[1],
                     f"ThreatIntel: {decoys[1]} - residential proxy, mixed reputation.",
                     weight=0.4, confirms_attacker=False),
            Evidence("correlate_events", attacker_ip,
                     f"Correlation: {attacker_ip} is the only IP in this burst with a successful "
                     "auth and follow-on token export.",
                     weight=1.0, confirms_attacker=True),
        ],
        background_logs=_noise_lines(rng, 7),
    )


def gen_data_exfiltration(rng: random.Random) -> Scenario:
    """Slow, low-volume egress to attacker-controlled DNS/HTTPS — easy to miss."""
    attacker_ip = _ip(rng, ["104.21", "172.67"])
    src_host = _host(rng, "db")
    decoys = [_ip(rng) for _ in range(3)]
    return Scenario(
        scenario_type="data_exfiltration",
        is_benign=False,
        attacker_ip=attacker_ip,
        attacker_host=src_host,
        decoy_ips=decoys,
        benign_ips=[_ip(rng) for _ in range(2)],
        target_hosts=[src_host],
        initial_alert=f"DLP: anomalous outbound TLS volume from {src_host} (3.2GB/24h vs 50MB baseline)",
        alert_severity="medium",
        correlation_pairs=[(src_host, attacker_ip)],
        summary_keywords=["exfiltration", "egress", attacker_ip, src_host],
        step_budget=12,
        evidence=[
            Evidence("query_logs", src_host,
                     f"Netflow: {src_host} -> {attacker_ip}:443 over 24h, "
                     "consistent 64KB chunks every 47s. Pattern: covert channel.",
                     weight=1.0, confirms_attacker=True),
            Evidence("inspect_endpoint", src_host,
                     "EDR: scheduled task 'sysmon-update' created 2 days ago, "
                     "spawns powershell + curl to external CDN-fronted URL.",
                     weight=0.95, confirms_attacker=True),
            Evidence("check_threat_intel", attacker_ip,
                     f"ThreatIntel: {attacker_ip} is a Cloudflare-fronted endpoint. "
                     "Domain registered 6 days ago. SSL CN matches DGA pattern.",
                     weight=0.85, confirms_attacker=True),
            Evidence("query_logs", decoys[0],
                     f"Netflow: {decoys[0]} - benign Office365 sync.",
                     weight=0.3, confirms_attacker=False),
            Evidence("correlate_events", src_host,
                     "Correlation: scheduled task -> powershell -> curl -> attacker IP, "
                     "exfil window aligns with off-hours.",
                     weight=1.0, confirms_attacker=True),
        ],
        background_logs=_noise_lines(rng, 6),
    )


def gen_multi_stage_chain(rng: random.Random) -> Scenario:
    """Recon -> exploit -> persistence -> exfil. Multiple correlated indicators."""
    attacker_ip = _ip(rng, ["185.220", "45.227"])
    pivot_host = _host(rng, "srv")
    target_host = _host(rng, "db")
    decoys = [_ip(rng) for _ in range(3)]
    return Scenario(
        scenario_type="multi_stage_chain",
        is_benign=False,
        attacker_ip=attacker_ip,
        attacker_host=pivot_host,
        decoy_ips=decoys,
        benign_ips=[_ip(rng) for _ in range(2)],
        target_hosts=[pivot_host, target_host],
        initial_alert=f"SIEM: kill-chain match - recon then exploit then persistence indicators on {pivot_host}",
        alert_severity="critical",
        correlation_pairs=[(attacker_ip, pivot_host), (pivot_host, target_host)],
        summary_keywords=["kill", "chain", "persistence", "exfil", attacker_ip],
        step_budget=12,
        evidence=[
            Evidence("query_logs", attacker_ip,
                     f"Edge: {attacker_ip} ran nmap-scripts against /24 over 6h, then specific exploit "
                     f"(CVE-2024-1234) against {pivot_host}.",
                     weight=1.0, confirms_attacker=True),
            Evidence("inspect_endpoint", pivot_host,
                     "EDR: webshell dropped in /var/www/uploads/. Cron entry persists shell every 5m. "
                     f"Outbound beacon to {attacker_ip}.",
                     weight=1.0, confirms_attacker=True),
            Evidence("query_logs", target_host,
                     f"DB audit: {pivot_host} extracted 240k rows from customer table to /tmp, "
                     "compressed and exfiltrated via webshell.",
                     weight=1.0, confirms_attacker=True),
            Evidence("check_threat_intel", attacker_ip,
                     f"ThreatIntel: {attacker_ip} attributed to FIN-style actor, "
                     "active in ransomware staging.",
                     weight=0.9, confirms_attacker=True),
            Evidence("correlate_events", pivot_host,
                     "Correlation: full kill-chain visible - "
                     "recon -> exploit -> webshell -> persistence -> data egress.",
                     weight=1.0, confirms_attacker=True),
            Evidence("query_logs", decoys[0],
                     "Edge: scan from public scanner, no follow-on. Unrelated.",
                     weight=0.2, confirms_attacker=False),
        ],
        background_logs=_noise_lines(rng, 5),
    )



def gen_long_horizon_apt(rng: random.Random) -> Scenario:
    """Multi-week APT campaign across 5 hosts with a 20-step budget.

    Implements OpenEnv Theme 2 (Super Long-Horizon Planning): the agent must
    decompose the investigation into a 5-phase kill chain (recon -> initial
    access -> persistence -> lateral movement -> exfiltration), maintain
    state across many tool calls, and resist the temptation to commit
    attribution early. Several decoys (a benign red-team pen tester, a noisy
    backup service, an external scanner) look more suspicious than the real
    attacker if the agent only samples one or two pieces of evidence.
    """
    attacker_ip = _ip(rng, ["203.0.113"])

    # 5-host kill-chain footprint
    edge_host   = _host(rng, "edge")
    user_host   = _host(rng, "ws")
    file_host   = _host(rng, "fs")
    db_host     = _host(rng, "db")
    egress_host = _host(rng, "proxy")
    target_hosts = [edge_host, user_host, file_host, db_host, egress_host]

    # Decoys that look juicy
    redteam_ip = _ip(rng, ["198.51.100"])
    backup_ip  = _ip(rng, ["10.0"])
    scanner_ip = _ip(rng, ["198.51.100"])
    decoys = [redteam_ip, backup_ip, scanner_ip]
    benign_ips = [_ip(rng) for _ in range(4)]

    return Scenario(
        scenario_type="long_horizon_apt",
        is_benign=False,
        attacker_ip=attacker_ip,
        attacker_host=user_host,
        decoy_ips=decoys,
        benign_ips=benign_ips,
        target_hosts=target_hosts,
        initial_alert=(
            f"[APT-SUSPECT] Multi-day low-and-slow activity correlated across "
            f"{len(target_hosts)} hosts (edge -> workstation -> file server -> "
            f"db -> proxy). Initial signal: anomalous TLS SNI patterns from "
            f"{attacker_ip}. Multiple plausible suspects. Investigate end-to-end."
        ),
        alert_severity="critical",
        correlation_pairs=[
            (attacker_ip, edge_host),
            (edge_host, user_host),
            (user_host, file_host),
            (file_host, db_host),
            (db_host, egress_host),
            (egress_host, attacker_ip),
        ],
        summary_keywords=[
            "apt", "kill", "chain", "persistence", "lateral", "exfil",
            attacker_ip,
        ],
        step_budget=20,
        evidence=[
            # Phase 1: Recon
            Evidence("query_logs", attacker_ip,
                     f"Edge IDS: {attacker_ip} performed paced TCP/443 probes "
                     f"against {edge_host} over 14 days, ~1 packet every "
                     "6 minutes. Below standard rate-limit thresholds.",
                     weight=0.7, confirms_attacker=True),
            Evidence("check_threat_intel", attacker_ip,
                     f"ThreatIntel: {attacker_ip} matches infrastructure of "
                     "a known APT cluster (low confidence, sinkholed C2).",
                     weight=0.6, confirms_attacker=True),
            # Phase 2: Initial access via spear-phish landing on user_host
            Evidence("inspect_endpoint", user_host,
                     f"EDR: {user_host} opened a spear-phish PDF that spawned "
                     "powershell with -enc base64 payload. Process tree: "
                     "Outlook -> AcroRd32 -> powershell -> rundll32.",
                     weight=1.0, confirms_attacker=True),
            Evidence("query_logs", user_host,
                     f"DNS: {user_host} resolved 4 typo-squat domains in the "
                     f"15min after the PDF open. Resolved IPs include {attacker_ip}.",
                     weight=0.9, confirms_attacker=True),
            # Phase 3: Persistence
            Evidence("inspect_endpoint", user_host,
                     "Persistence: scheduled task 'OneDriveSync' (mis-spelled) "
                     "calls a base64 powershell stager every 30 minutes.",
                     weight=0.9, confirms_attacker=True),
            # Phase 4: Lateral movement
            Evidence("query_logs", file_host,
                     f"SMB audit: {user_host} authenticated to {file_host} "
                     "with a service account never used from a workstation "
                     "before. Then enumerated ACLs and copied 14 GB out.",
                     weight=1.0, confirms_attacker=True),
            Evidence("inspect_endpoint", db_host,
                     f"DB audit: {file_host} ran SELECT * FROM customer_pii "
                     "with LIMIT 50000 OFFSET k, 10 batches.",
                     weight=1.0, confirms_attacker=True),
            Evidence("correlate_events", user_host,
                     f"Correlation: {user_host} -> {file_host} -> {db_host} "
                     "all touched within a 38-minute window using the same "
                     "Kerberos TGT - unmistakable lateral chain.",
                     weight=1.0, confirms_attacker=True),
            # Phase 5: Exfil via egress proxy
            Evidence("query_logs", egress_host,
                     f"Proxy: {egress_host} sent 12.4 GB outbound to "
                     f"{attacker_ip} over TLS-1.3 with self-signed cert, "
                     "spread across 6 hours, JA3 fingerprint matches phase 1.",
                     weight=1.0, confirms_attacker=True),
            Evidence("correlate_events", attacker_ip,
                     f"Correlation: full kill chain confirmed - recon "
                     f"({attacker_ip}) -> phish ({user_host}) -> persistence "
                     f"-> lateral ({file_host}, {db_host}) -> exfil "
                     f"({egress_host} -> {attacker_ip}). JA3 ties phases 1 "
                     "and 5 together.",
                     weight=1.0, confirms_attacker=True),
            # DECOYS
            Evidence("query_logs", redteam_ip,
                     f"Edge: {redteam_ip} ran an aggressive scan against "
                     "/24 for 48h. Tagged 'authorized red-team' in CMDB.",
                     weight=0.4, confirms_attacker=False),
            Evidence("inspect_endpoint", egress_host,
                     f"Backup service on {backup_ip} pushed nightly backups "
                     "to the egress proxy. High volume but expected.",
                     weight=0.3, confirms_attacker=False),
            Evidence("check_threat_intel", scanner_ip,
                     f"ThreatIntel: {scanner_ip} is a Shodan/Censys scanner. "
                     "No follow-on activity attributable.",
                     weight=0.2, confirms_attacker=False),
        ],
        background_logs=_noise_lines(rng, 8),
    )


_GENERATORS = {
    "benign_scan":          gen_benign_scan,
    "phishing_lateral":     gen_phishing_lateral,
    "credential_stuffing":  gen_credential_stuffing,
    "data_exfiltration":    gen_data_exfiltration,
    "multi_stage_chain":    gen_multi_stage_chain,
    "long_horizon_apt":     gen_long_horizon_apt,
}


def generate_scenario(scenario_type: Optional[str] = None,
                      seed: Optional[int] = None) -> Scenario:
    """Generate a randomized scenario.

    scenario_type=None picks one uniformly at random from SCENARIO_TYPES.
    """
    rng = random.Random(seed)
    if scenario_type is None:
        scenario_type = rng.choice(SCENARIO_TYPES)
    if scenario_type not in _GENERATORS:
        raise ValueError(
            f"Unknown scenario_type: {scenario_type}. Valid: {SCENARIO_TYPES}"
        )
    return _GENERATORS[scenario_type](rng)
