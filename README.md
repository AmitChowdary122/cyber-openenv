# 🔐 CyberEnv — Cybersecurity Incident Response OpenEnv

**Built for the Meta × Hugging Face OpenEnv Hackathon**

A realistic cybersecurity simulation environment where AI agents learn to detect, analyze, and respond to cyber attacks through structured reasoning.

---

## 🧠 Why this environment is realistic

- **SOC‑grade logs**: Timestamps, IDS signatures (e.g., "ET SCAN Potential Port Scan"), and CVE references (CVE‑2024‑1234) mirror real security operations.
- **Multi‑stage attack simulation**: Attackers progress through `scan → probe → exploit`, generating distinct log patterns at each stage.
- **Ambiguity and false positives**: Decoy IPs and benign suspicious IPs produce overlapping alerts, forcing the agent to correlate information across steps – no single log reveals the attacker.
- **Investigation as a first‑class action**: `investigate_ip` enriches state and rewards correlation, not blind blocking.

---

## 🔥 Why the HARD task is challenging for frontier models

The **hard** task requires **genuine multi‑step reasoning**:

1. **Large hypothesis space**: 6–8 suspicious IPs, only one is the real attacker.
2. **Overlapping evidence**: Decoys mimic real attack progression; benign IPs generate false positives.
3. **Information gathering**: The agent must decide which IPs to investigate, balancing exploration and exploitation within **only 5 steps**.
4. **No retries**: A wrong block terminates the episode with a severe penalty (-0.6).
5. **Penalty for guessing**: Blocking an IP without any investigation incurs an extra -0.05.

**Example reasoning trace** (successful agent):

- **Step 1**: `analyze_log` → sees port scans from IP_A and IP_B, login failures from IP_C.
- **Step 2**: `investigate_ip(IP_A)` → enriched data shows IP_A is a decoy (high suspicion but no exploit stage).
- **Step 3**: `investigate_ip(IP_C)` → enrichment reveals IP_C has consistent failed logins but no scan phase – decoy.
- **Step 4**: `investigate_ip(IP_B)` → enrichment shows IP_B progressed from scan to probe to exploit – real attacker.
- **Step 5**: `block_ip(IP_B)` → success, reward +0.5, threat level drops, episode ends.

Without investigation, the agent would likely block a decoy and fail.

---

## 🎯 Tasks

| Task    | Description                                                                 | Success Condition                                          | Max Steps |
|---------|-----------------------------------------------------------------------------|------------------------------------------------------------|-----------|
| Easy    | Identify the attacker IP from logs                                          | `identified_attacker is not None`                          | 10        |
| Medium  | Identify and block the attacker                                             | `identified_attacker in blocked_ips`                       | 8         |
| Hard    | Investigate, correlate, and block only the true attacker (no false positives) | `identified_attacker in blocked_ips and threat_level < 0.1 and no false positives blocked` | 5         |

---

## 📊 Reward Shaping (Encourages Reasoning)

| Action                         | Reward                     | Rationale                                     |
|--------------------------------|----------------------------|-----------------------------------------------|
| `investigate_ip` (attacker)    | +0.05                      | Gathering intelligence is valuable            |
| `investigate_ip` (non‑attacker)| +0.01                      | Still helps eliminate suspects                |
| `identify_attacker` (correct)  | +0.5                       | Good, but blocking is the final goal          |
| `block_ip` (correct)           | +0.5 (+0.2 if identified)  | Major success                                 |
| `block_ip` (wrong)             | -0.6                       | Severe penalty – episode ends                 |
| Block without investigation    | -0.05 extra                | Discourages guessing                          |
| Multi‑step correlation         | +0.02 (identify after invest) | Rewards using past information              |
| Step penalty                   | -0.05 per step             | Prevents brute‑force                          |

---

## 🚀 Quick Start

```bash
# Install
pip install -r requirements.txt

# Run server
python api/main.py

# Run baseline agent (easy/medium only – hard requires reasoning)
python inference.py