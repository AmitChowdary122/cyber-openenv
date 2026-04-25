"""
Scenario generators for CyberSOC Arena.

A Scenario is a static, replay-safe description of a SOC incident:
  - the alert the analyst sees up front
  - the ground-truth attacker (or None if benign)
  - the evidence that *can* be revealed by investigative tools
  - background log noise
  - terminal-decision metadata (kill chain, summary keywords)

Evidence is keyed by (action_type, target). When the agent runs that
action against that target, the evidence is "revealed" and contributes
to investigation reward + decision-readiness.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import random


# ----------------------------------------------------------------------------
# Data classes
# ----------------------------------------------------------------------------

@dataclass
class Evidence:
    action_type: str           # which investigative tool reveals this
    target: str                # IP or host the tool must be aimed at
    text: str                  # human-readable finding
    weight: float = 0.5        # 0.0 - 1.0, importance to the investigation
    confirms_attacker: bool = True
    phase: int = 1             # kill-chain phase (1..5)
    tags: Tuple[str, ...] = field(default_factory=tuple)


@dataclass
class Scenario:
    scenario_type: str
    step_budget: int
    alert_severity: str
    is_benign: bool
    initial_alert: str
    target_hosts: List[str]
    decoy_ips: List[str]
    correlation_pairs: List[Tuple[str, str]]
    summary_keywords: List[str]
    evidence: List[Evidence]
    background_logs: List[str]
    attacker_ip: Optional[str] = None
    c2_ip: Optional[str] = None
    seed: int = 0

    def evidence_index(self) -> Dict[Tuple[str, str], List[Evidence]]:
        idx: Dict[Tuple[str, str], List[Evidence]] = {}
        for e in self.evidence:
            idx.setdefault((e.action_type, e.target), []).append(e)
        return idx


SCENARIO_TYPES: List[str] = [
    "benign_scan",
    "credential_stuffing",
    "phishing_lateral",
    "data_exfiltration",
    "multi_stage_chain",
    "long_horizon_apt",
    "ransomware_deployment",
    "supply_chain_attack",
]


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _rand_internal_ip(rng: random.Random) -> str:
    return f"10.0.{rng.randint(0, 5)}.{rng.randint(2, 250)}"


def _rand_external_ip(rng: random.Random) -> str:
    return f"{rng.randint(45, 220)}.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(2, 250)}"


_NOISE_TEMPLATES = [
    "[INFO] DHCP lease renewed for {ip}",
    "[INFO] AV scan completed on {host}",
    "[INFO] Backup job finished: {host}",
    "[INFO] HTTP 200 GET /healthz from {ip}",
    "[INFO] LDAP bind succeeded user=svc_monitor src={ip}",
    "[INFO] NTP sync ok on {host}",
    "[WARN] TLS handshake retry from {ip}",
    "[INFO] DNS query A example.internal from {ip}",
    "[INFO] SMTP relay ok from {ip}",
    "[INFO] Patch KB5031234 applied to {host}",
    "[INFO] User helpdesk login OK from {ip}",
    "[INFO] CRL refresh completed",
    "[INFO] Print spool flush on {host}",
    "[INFO] Outlook calendar sync from {ip}",
    "[INFO] OneDrive client heartbeat from {host}",
]


def _noise_lines(rng: random.Random, n: int = 8) -> List[str]:
    hosts = ["wkst-01", "wkst-07", "laptop-23", "print-srv", "mail-01"]
    lines = []
    for _ in range(n):
        tmpl = rng.choice(_NOISE_TEMPLATES)
        ip = _rand_internal_ip(rng)
        host = rng.choice(hosts)
        lines.append(tmpl.format(ip=ip, host=host))
    return lines


# ----------------------------------------------------------------------------
# Scenario 0 — benign_scan (false positive)
# ----------------------------------------------------------------------------

def gen_benign_scan(rng: random.Random) -> Scenario:
    src = _rand_external_ip(rng)
    host = "web-01"
    evidence = [
        Evidence("check_threat_intel", src,
                 f"{src} is a known security vendor scanner (Shodan/Censys range).",
                 weight=0.9, confirms_attacker=False, phase=1, tags=("scanner",)),
        Evidence("query_logs", src,
                 f"Burst of 14 GETs from {src} to /, /robots.txt, /favicon.ico — fingerprinting only.",
                 weight=0.7, confirms_attacker=False, phase=1),
        Evidence("inspect_endpoint", host,
                 f"No new processes / no persistence on {host}. Web server is stock.",
                 weight=0.6, confirms_attacker=False, phase=2),
        Evidence("correlate_events", host,
                 f"All {src} requests received HTTP 404 / 401 — no successful exploitation.",
                 weight=0.6, confirms_attacker=False, phase=2),
        Evidence("query_logs", host,
                 f"{host} access log: same {src} fingerprint repeats hourly, "
                 f"matches scheduled-scan window — benign, no abuse reports.",
                 weight=0.5, confirms_attacker=False, phase=2,
                 tags=("scanner", "scheduled-scan")),
    ]
    return Scenario(
        scenario_type="benign_scan",
        step_budget=6,
        alert_severity="low",
        is_benign=True,
        initial_alert=f"IDS: repeated probes from {src} hitting {host}/ web ports.",
        target_hosts=[host],
        decoy_ips=[],
        correlation_pairs=[(src, host)],
        summary_keywords=["scanner", "benign", "false-positive", src],
        evidence=evidence,
        background_logs=_noise_lines(rng, 6),
        attacker_ip=None,
        seed=rng.randint(0, 1 << 30),
    )


# ----------------------------------------------------------------------------
# Scenario 1 — credential_stuffing
# ----------------------------------------------------------------------------

def gen_credential_stuffing(rng: random.Random) -> Scenario:
    attacker = _rand_external_ip(rng)
    decoy = _rand_external_ip(rng)
    host = "auth-gw"
    evidence = [
        Evidence("query_logs", attacker,
                 f"{attacker}: 1,142 failed logins across 38 accounts in 3 minutes.",
                 weight=0.95, confirms_attacker=True, phase=1),
        Evidence("check_threat_intel", attacker,
                 f"{attacker} flagged on AbuseIPDB (score 92) — credential-stuffing botnet node.",
                 weight=0.9, confirms_attacker=True, phase=1),
        Evidence("inspect_endpoint", host,
                 f"{host} auth log shows brute-force pattern; 1 successful login svc_finance from {attacker}.",
                 weight=0.95, confirms_attacker=True, phase=2),
        Evidence("correlate_events", host,
                 f"Stuffing burst from {attacker} aligned with leaked-cred dump 'collection-1'.",
                 weight=0.8, confirms_attacker=True, phase=2),
        # decoys
        Evidence("query_logs", decoy,
                 f"{decoy}: 4 failed logins, then normal MFA-success — looks like a fat-fingered user.",
                 weight=0.4, confirms_attacker=False, phase=1),
        Evidence("check_threat_intel", decoy,
                 f"{decoy} is a residential ISP range, no abuse reports.",
                 weight=0.4, confirms_attacker=False, phase=1),
    ]
    return Scenario(
        scenario_type="credential_stuffing",
        step_budget=8,
        alert_severity="medium",
        is_benign=False,
        initial_alert=f"Auth gateway: spike in failed logins from {attacker} (and a few from {decoy}).",
        target_hosts=[host],
        decoy_ips=[decoy],
        correlation_pairs=[(attacker, host)],
        summary_keywords=["credential", "stuffing", "brute", attacker],
        evidence=evidence,
        background_logs=_noise_lines(rng, 8),
        attacker_ip=attacker,
        seed=rng.randint(0, 1 << 30),
    )


# ----------------------------------------------------------------------------
# Scenario 2 — phishing_lateral
# ----------------------------------------------------------------------------

def gen_phishing_lateral(rng: random.Random) -> Scenario:
    attacker = _rand_external_ip(rng)
    decoy1 = _rand_external_ip(rng)
    decoy2 = _rand_internal_ip(rng)
    h_user = "wkst-finance-04"
    h_lat = "file-srv"
    evidence = [
        Evidence("query_logs", attacker,
                 f"Inbound email from spoofed-domain to user-04 with macro-laden Excel; clicked at 09:14.",
                 weight=0.85, phase=1),
        Evidence("check_threat_intel", attacker,
                 f"{attacker} resolves to phishing-sender cluster (PROOFPOINT TAP cat=phish).",
                 weight=0.85, phase=1),
        Evidence("inspect_endpoint", h_user,
                 f"EDR on {h_user}: powershell.exe spawned by EXCEL.EXE; downloads to %TEMP%\\inv.dll.",
                 weight=0.95, phase=2),
        Evidence("query_logs", h_user,
                 f"{h_user}: outbound 443 to {attacker} every 60s — implant beacon.",
                 weight=0.9, phase=2),
        Evidence("inspect_endpoint", h_lat,
                 f"{h_lat}: SMB session from {h_user} copying out finance Q4 spreadsheets.",
                 weight=0.9, phase=3),
        Evidence("correlate_events", h_lat,
                 f"Implant on {h_user} -> SMB to {h_lat} -> beacon to {attacker} (full chain).",
                 weight=1.0, phase=3),
        # decoys
        Evidence("check_threat_intel", decoy1,
                 f"{decoy1} is a Microsoft 365 IP — legitimate.",
                 weight=0.3, confirms_attacker=False, phase=1),
        Evidence("query_logs", decoy2,
                 f"{decoy2} is the print server polling the file share — normal.",
                 weight=0.3, confirms_attacker=False, phase=2),
    ]
    return Scenario(
        scenario_type="phishing_lateral",
        step_budget=10,
        alert_severity="high",
        is_benign=False,
        initial_alert=f"EDR: suspicious child-process from EXCEL.EXE on {h_user}.",
        target_hosts=[h_user, h_lat],
        decoy_ips=[decoy1, decoy2],
        correlation_pairs=[(attacker, h_user), (h_user, h_lat)],
        summary_keywords=["phishing", "macro", "lateral", "beacon", attacker],
        evidence=evidence,
        background_logs=_noise_lines(rng, 10),
        attacker_ip=attacker,
        seed=rng.randint(0, 1 << 30),
    )


# ----------------------------------------------------------------------------
# Scenario 3 — data_exfiltration
# ----------------------------------------------------------------------------

def gen_data_exfiltration(rng: random.Random) -> Scenario:
    attacker = _rand_external_ip(rng)
    c2 = _rand_external_ip(rng)
    decoy = _rand_external_ip(rng)
    h_db = "crm-db"
    h_app = "crm-app"
    evidence = [
        Evidence("query_logs", h_db,
                 f"{h_db}: 4.2 GB SELECT * FROM customers exported by svc_app at 02:11.",
                 weight=0.95, phase=1),
        Evidence("inspect_endpoint", h_db,
                 f"{h_db}: rogue cron job tar/gzipping /var/exports nightly.",
                 weight=0.9, phase=2),
        Evidence("query_logs", h_app,
                 f"{h_app}: outbound HTTPS bursts to {c2}, 4.1 GB total, 02:13–02:34.",
                 weight=0.95, phase=3),
        Evidence("check_threat_intel", c2,
                 f"{c2} is a DigitalOcean droplet, sinkhole tagged 'EXFIL/STAGER'.",
                 weight=0.85, phase=3),
        Evidence("correlate_events", h_app,
                 f"DB export size matches outbound burst to {c2} within 90s window.",
                 weight=1.0, phase=4),
        Evidence("check_threat_intel", attacker,
                 f"{attacker} authenticated to {h_app} via stolen svc_app key, then triggered the export.",
                 weight=0.85, phase=1),
        # decoys
        Evidence("check_threat_intel", decoy,
                 f"{decoy} is the CDN edge for static assets — normal.",
                 weight=0.3, confirms_attacker=False, phase=2),
    ]
    return Scenario(
        scenario_type="data_exfiltration",
        step_budget=12,
        alert_severity="high",
        is_benign=False,
        initial_alert=f"DLP: 4 GB outbound to {c2} from {h_app} during off-hours.",
        target_hosts=[h_db, h_app],
        decoy_ips=[decoy],
        correlation_pairs=[(h_db, h_app), (h_app, c2)],
        summary_keywords=["exfiltration", "database", "outbound", c2, attacker],
        evidence=evidence,
        background_logs=_noise_lines(rng, 10),
        attacker_ip=attacker,
        c2_ip=c2,
        seed=rng.randint(0, 1 << 30),
    )


# ----------------------------------------------------------------------------
# Scenario 4 — multi_stage_chain
# ----------------------------------------------------------------------------

def gen_multi_stage_chain(rng: random.Random) -> Scenario:
    attacker = _rand_external_ip(rng)
    c2 = _rand_external_ip(rng)
    decoy1 = _rand_external_ip(rng)
    decoy2 = _rand_internal_ip(rng)
    h_edge = "vpn-edge"
    h_jump = "jump-srv"
    h_app = "billing-app"
    evidence = [
        Evidence("check_threat_intel", attacker,
                 f"{attacker} matches FIN7 infrastructure cluster (high confidence).",
                 weight=0.9, phase=1),
        Evidence("query_logs", h_edge,
                 f"{h_edge}: VPN login as user-49 from {attacker}, no MFA challenge — token replay.",
                 weight=0.95, phase=2),
        Evidence("inspect_endpoint", h_jump,
                 f"{h_jump}: psexec by user-49 onto {h_app}; mimikatz dropped in C:\\Windows\\Temp.",
                 weight=0.95, phase=3),
        Evidence("query_logs", h_app,
                 f"{h_app}: rogue scheduled task 'GoogleUpdateCheck' beaconing to {c2}.",
                 weight=0.9, phase=4),
        Evidence("check_threat_intel", c2,
                 f"{c2}: Cobalt Strike team server on Akamai-fronted domain.",
                 weight=0.9, phase=4),
        Evidence("correlate_events", h_app,
                 f"VPN({attacker}) -> {h_edge} -> {h_jump} -> {h_app} -> beacon({c2}).",
                 weight=1.0, phase=5),
        # decoys
        Evidence("check_threat_intel", decoy1,
                 f"{decoy1} is a corp-VPN concentrator IP — internal traffic.",
                 weight=0.3, confirms_attacker=False, phase=2),
        Evidence("query_logs", decoy2,
                 f"{decoy2} is the patch-management agent polling the jump server.",
                 weight=0.3, confirms_attacker=False, phase=3),
    ]
    return Scenario(
        scenario_type="multi_stage_chain",
        step_budget=15,
        alert_severity="high",
        is_benign=False,
        initial_alert=f"SIEM: anomalous VPN login + lateral movement converging on {h_app}.",
        target_hosts=[h_edge, h_jump, h_app],
        decoy_ips=[decoy1, decoy2],
        correlation_pairs=[(attacker, h_edge), (h_edge, h_jump), (h_jump, h_app), (h_app, c2)],
        summary_keywords=["multi-stage", "vpn", "lateral", "cobalt", "beacon", attacker, c2],
        evidence=evidence,
        background_logs=_noise_lines(rng, 11),
        attacker_ip=attacker,
        c2_ip=c2,
        seed=rng.randint(0, 1 << 30),
    )


# ----------------------------------------------------------------------------
# Scenario 5 — long_horizon_apt  (THE 20-step showcase, Theme 2)
# ----------------------------------------------------------------------------

def gen_long_horizon_apt(rng: random.Random) -> Scenario:
    attacker_ip = _rand_external_ip(rng)
    c2_ip = _rand_external_ip(rng)

    decoy_ips = [_rand_external_ip(rng) for _ in range(3)] + [_rand_internal_ip(rng) for _ in range(2)]

    dmz = "dmz_host"
    jump = "jump_host"
    dc = "dc_host"
    db = "db_host"
    hr = "hr_host"
    hosts = [dmz, jump, dc, db, hr]

    evidence: List[Evidence] = [
        # Phase 1 — Recon / Initial Access
        Evidence("check_threat_intel", attacker_ip,
                 f"{attacker_ip} matches APT-41 infra cluster (TTPs: spearphish + webshell).",
                 weight=0.95, phase=1, tags=("apt", "apt-41")),
        Evidence("query_logs", attacker_ip,
                 f"Spearphish email from {attacker_ip} delivered HTA payload to 3 finance users at 08:42.",
                 weight=0.9, phase=1, tags=("phishing",)),

        # Phase 2 — Foothold (DMZ webshell)
        Evidence("inspect_endpoint", dmz,
                 f"{dmz}: webshell 'cmd.aspx' dropped under /var/www/uploads/ at 09:03; owned by w3wp.exe.",
                 weight=0.95, phase=2, tags=("webshell",)),
        Evidence("query_logs", dmz,
                 f"{dmz}: 47 POST requests to cmd.aspx from {attacker_ip}, base64 'whoami /priv' & 'net group'.",
                 weight=0.9, phase=2, tags=("webshell",)),

        # Phase 3 — Discovery / Lateral
        Evidence("inspect_endpoint", jump,
                 f"{jump}: BloodHound LDAP collector 'SharpHound.exe' executed; full domain map exported.",
                 weight=0.9, phase=3, tags=("lateral", "discovery")),
        Evidence("query_logs", jump,
                 f"{jump}: WMI lateral movement DMZ -> JUMP using stolen svc_iis cred at 09:31.",
                 weight=0.95, phase=3, tags=("lateral",)),

        # Phase 4 — DC compromise (golden ticket)
        Evidence("inspect_endpoint", dc,
                 f"{dc}: mimikatz DCSync against krbtgt account; golden ticket forged for Domain Admins.",
                 weight=1.0, phase=4, tags=("golden", "ticket", "dcsync")),
        Evidence("correlate_events", dc,
                 f"Full kill chain confirmed: {attacker_ip} -> {dmz} -> {jump} -> {dc} -> {db} -> {c2_ip}.",
                 weight=1.0, phase=4, tags=("kill-chain",)),

        # Phase 5 — Exfiltration via DNS tunneling
        Evidence("query_logs", db,
                 f"{db}: 180,000 rows SELECT * FROM customers; chunked over DNS TXT to {c2_ip} (encoded).",
                 weight=1.0, phase=5, tags=("exfiltration", "dns")),
        Evidence("check_threat_intel", c2_ip,
                 f"{c2_ip}: DGA-style domain (entropy 4.2), newly-registered TLD .top, sinkholed yesterday.",
                 weight=0.9, phase=5, tags=("dns", "dga")),

        # Bridging evidence (deepens chain at db / jump)
        Evidence("inspect_endpoint", db,
                 f"{db}: dns2tcp client process 'svchost_dnsupd.exe' spawned by SQL Agent.",
                 weight=0.85, phase=5, tags=("exfiltration", "dns")),
        Evidence("correlate_events", jump,
                 f"{jump}: kerberos golden-ticket TGT used to access {db} as Domain Admin (impersonated).",
                 weight=0.95, phase=4, tags=("golden", "ticket")),
        Evidence("query_logs", dmz,
                 f"{dmz}: webshell file timestamp aligned with phishing-click on user wkst (within 4 min).",
                 weight=0.85, phase=2, tags=("webshell",)),

        # 3 decoys (intentional false leads)
        Evidence("check_threat_intel", decoy_ips[0],
                 f"{decoy_ips[0]} is a Qualys/Tenable scanner IP — internal vuln-mgmt sweep.",
                 weight=0.5, confirms_attacker=False, phase=1, tags=("scanner",)),
        Evidence("check_threat_intel", decoy_ips[1],
                 f"{decoy_ips[1]} is a residential proxy / ISP range — no APT indicators.",
                 weight=0.4, confirms_attacker=False, phase=2),
        Evidence("inspect_endpoint", hr,
                 f"{hr}: Defender blocked an EICAR test file dropped by IT during AV validation.",
                 weight=0.4, confirms_attacker=False, phase=3, tags=("av-block",)),
    ]

    return Scenario(
        scenario_type="long_horizon_apt",
        step_budget=20,
        alert_severity="medium",  # intentionally under-flagged
        is_benign=False,
        initial_alert=(
            "SIEM (medium): unusual outbound DNS volume from db_host. "
            "Single-source phishing email reported by 1 finance user 4 hours earlier."
        ),
        target_hosts=hosts,
        decoy_ips=decoy_ips,
        correlation_pairs=[
            (attacker_ip, dmz),
            (dmz, jump),
            (jump, dc),
            (dc, db),
            (db, c2_ip),
        ],
        summary_keywords=[
            "apt", "phishing", "webshell", "lateral",
            "golden", "ticket", "exfiltration", "dns",
            attacker_ip,
        ],
        evidence=evidence,
        background_logs=_noise_lines(rng, 12),
        attacker_ip=attacker_ip,
        c2_ip=c2_ip,
        seed=rng.randint(0, 1 << 30),
    )


# ----------------------------------------------------------------------------
# Scenario 6 — ransomware_deployment (15-step LockBit-style affiliate)
# ----------------------------------------------------------------------------

def _rand_ip_in_prefix(rng: random.Random, prefix: str) -> str:
    """Pick a random IP whose first 1-2 octets match `prefix` (e.g. '91.92')."""
    octets = prefix.split(".")
    while len(octets) < 4:
        octets.append(str(rng.randint(0, 255)))
    octets[-1] = str(rng.randint(2, 250))
    return ".".join(octets)


def gen_ransomware_deployment(rng: random.Random) -> Scenario:
    attacker = _rand_ip_in_prefix(rng, rng.choice(["91.92", "185.220"]))
    decoys = [_rand_external_ip(rng) for _ in range(3)] + [_rand_internal_ip(rng)]

    ws = "ws_host"
    file_srv = "file_srv"
    backup_srv = "backup_srv"
    dc = "dc_host"
    hosts = [ws, file_srv, backup_srv, dc]

    evidence: List[Evidence] = [
        Evidence("check_threat_intel", attacker,
                 f"{attacker}: RaaS affiliate, LockBit 3.0 infrastructure (high confidence).",
                 weight=0.95, phase=1, tags=("ransomware", "raas", "lockbit")),
        Evidence("query_logs", ws,
                 f"{ws}: phishing PDF opened 47 min before alert; "
                 f"PowerShell download cradle to {attacker} (mshta+iex).",
                 weight=1.0, phase=2, tags=("phishing", "powershell", "implant")),
        Evidence("inspect_endpoint", ws,
                 f"{ws}: Cobalt Strike beacon resident in lsass; "
                 f"LSASS read; lateral SMB sessions to {file_srv} and {backup_srv}.",
                 weight=1.0, phase=2, tags=("cobalt", "beacon", "lateral")),
        Evidence("query_logs", file_srv,
                 f"{file_srv}: 14,312 RENAME *.locked operations in 480 s; "
                 f"vssadmin delete shadows /all /quiet executed.",
                 weight=1.0, phase=3, tags=("ransomware", "encryption", "wiper")),
        Evidence("inspect_endpoint", backup_srv,
                 f"{backup_srv}: backup agent process killed; VSS snapshots wiped; "
                 f"ransom note 'README_RECOVER.txt' dropped in every share.",
                 weight=0.9, phase=3, tags=("ransomware", "backup", "wiper")),
        Evidence("correlate_events", ws,
                 f"Full chain: phish on {ws} -> beacon to {attacker} -> "
                 f"SMB lateral to {file_srv}/{backup_srv} -> mass encrypt + wipe shadows.",
                 weight=1.0, phase=4, tags=("kill-chain",)),
        Evidence("inspect_endpoint", dc,
                 f"{dc}: AD queried for file-share enumeration from {ws} "
                 f"session token (svc_helpdesk).",
                 weight=0.7, phase=2, tags=("discovery",)),
        # decoys
        Evidence("query_logs", decoys[0],
                 f"{decoys[0]}: routine nightly backup job — unrelated to incident.",
                 weight=0.2, confirms_attacker=False, phase=1),
        Evidence("check_threat_intel", decoys[1],
                 f"{decoys[1]}: Cloudflare CDN exit node, mixed reputation but no abuse reports.",
                 weight=0.3, confirms_attacker=False, phase=1),
        Evidence("query_logs", decoys[2],
                 f"{decoys[2]}: Windows Defender AV signature update traffic — benign.",
                 weight=0.1, confirms_attacker=False, phase=1, tags=("av-update",)),
    ]

    return Scenario(
        scenario_type="ransomware_deployment",
        step_budget=15,
        alert_severity="critical",
        is_benign=False,
        initial_alert=(
            f"EDR: mass file rename events (.locked extension) on {file_srv}. "
            f"Shadow copies deleted on {backup_srv}. Estimated 14,000 files "
            f"encrypted in 8 minutes."
        ),
        target_hosts=hosts,
        decoy_ips=decoys,
        correlation_pairs=[(attacker, ws), (ws, file_srv), (ws, backup_srv)],
        summary_keywords=[
            "ransomware", "encryption", "lateral", "backup", "wiper", attacker,
        ],
        evidence=evidence,
        background_logs=_noise_lines(rng, 8),
        attacker_ip=attacker,
        seed=rng.randint(0, 1 << 30),
    )


# ----------------------------------------------------------------------------
# Scenario 7 — supply_chain_attack (18-step nation-state Jenkins compromise)
# ----------------------------------------------------------------------------

def gen_supply_chain_attack(rng: random.Random) -> Scenario:
    attacker = _rand_ip_in_prefix(rng, rng.choice(["45.142", "194.165"]))
    decoys = [_rand_external_ip(rng) for _ in range(3)] + [_rand_internal_ip(rng) for _ in range(2)]

    build = "build_srv"
    repo = "artifact_repo"
    prod_a = "prod_srv_a"
    prod_b = "prod_srv_b"
    hosts = [build, repo, prod_a, prod_b]

    evidence: List[Evidence] = [
        Evidence("check_threat_intel", attacker,
                 f"{attacker}: nation-state tooling cluster (SolarWinds-style TTPs, "
                 f"low-and-slow build-system targeting).",
                 weight=0.9, phase=1, tags=("apt", "nation-state", "supply-chain")),
        Evidence("inspect_endpoint", build,
                 f"{build}: malicious plugin injected into Jenkins pipeline 3 days ago; "
                 f"plugin exfils signing key on every build.",
                 weight=1.0, phase=2, tags=("supply-chain", "build", "key-theft")),
        Evidence("query_logs", build,
                 f"{build}: POST to {attacker}:443 carrying signing-key material "
                 f"at 02:17 UTC (outside business hours).",
                 weight=1.0, phase=2, tags=("exfiltration", "signing-key")),
        Evidence("query_logs", repo,
                 f"{repo}: 3 trojanised packages published with VALID company "
                 f"signature within 6 h of key theft.",
                 weight=0.95, phase=3, tags=("supply-chain", "artifact", "signed")),
        Evidence("inspect_endpoint", prod_a,
                 f"{prod_a}: trojanised package installed via routine update; "
                 f"backdoor beaconing to {attacker}.",
                 weight=1.0, phase=4, tags=("backdoor", "beacon", "compromised")),
        Evidence("inspect_endpoint", prod_b,
                 f"{prod_b}: same trojanised package, same backdoor beacon "
                 f"to {attacker} — second compromised production host.",
                 weight=0.9, phase=4, tags=("backdoor", "beacon", "compromised")),
        Evidence("correlate_events", build,
                 f"Full chain: Jenkins plugin -> signing-key theft -> "
                 f"poisoned artifact published -> {prod_a} & {prod_b} compromised.",
                 weight=1.0, phase=5, tags=("kill-chain",)),
        # decoys
        Evidence("check_threat_intel", decoys[0],
                 f"{decoys[0]}: Fastly CDN node, no abuse history.",
                 weight=0.2, confirms_attacker=False, phase=1),
        Evidence("query_logs", decoys[1],
                 f"{decoys[1]}: developer laptop, normal git pulls and IDE traffic.",
                 weight=0.15, confirms_attacker=False, phase=2),
        Evidence("inspect_endpoint", decoys[2],
                 f"{decoys[2]}: clean prod server, no trojanised package, no backdoor.",
                 weight=0.1, confirms_attacker=False, phase=4),
        Evidence("query_logs", decoys[3],
                 f"{decoys[3]}: routine package-manager (apt / yum) updates — benign.",
                 weight=0.1, confirms_attacker=False, phase=3),
    ]

    return Scenario(
        scenario_type="supply_chain_attack",
        step_budget=18,
        alert_severity="high",
        is_benign=False,
        initial_alert=(
            f"SIEM: anomalous outbound connection from {build} to unknown IP "
            f"during nightly CI build. Package signing key accessed outside "
            f"business hours."
        ),
        target_hosts=hosts,
        decoy_ips=decoys,
        correlation_pairs=[
            (attacker, build),
            (build, repo),
            (repo, prod_a),
            (repo, prod_b),
        ],
        summary_keywords=[
            "supply", "chain", "build", "artifact", "signing", "key", attacker,
        ],
        evidence=evidence,
        background_logs=_noise_lines(rng, 10),
        attacker_ip=attacker,
        seed=rng.randint(0, 1 << 30),
    )


# ----------------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------------

_GENERATORS = {
    "benign_scan": gen_benign_scan,
    "credential_stuffing": gen_credential_stuffing,
    "phishing_lateral": gen_phishing_lateral,
    "data_exfiltration": gen_data_exfiltration,
    "multi_stage_chain": gen_multi_stage_chain,
    "long_horizon_apt": gen_long_horizon_apt,
    "ransomware_deployment": gen_ransomware_deployment,
    "supply_chain_attack": gen_supply_chain_attack,
}


def generate_scenario(scenario_type: Optional[str] = None,
                      seed: Optional[int] = None) -> Scenario:
    """Generate a Scenario. If scenario_type is None, picks uniformly."""
    rng = random.Random(seed)
    if scenario_type is None:
        scenario_type = rng.choice(SCENARIO_TYPES)
    if scenario_type not in _GENERATORS:
        raise ValueError(f"Unknown scenario_type: {scenario_type}. "
                         f"Valid: {SCENARIO_TYPES}")
    return _GENERATORS[scenario_type](rng)
