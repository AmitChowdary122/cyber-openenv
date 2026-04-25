---
title: CyberSOC Arena
emoji: 🛡️
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 8000
pinned: false
license: apache-2.0
tags:
  - openenv
  - cybersecurity
  - reinforcement-learning
  - long-horizon-planning
  - self-improvement
  - tool-use
short_description: Long-horizon SOC analyst training environment for LLMs.
---

# CyberSOC Arena 🛡️

> **Training LLMs to think like SOC analysts** — multi-step investigation,
> adaptive curriculum, 20-step APT kill chains.

[![Open in HF Spaces](https://img.shields.io/badge/🤗-Demo%20Space-yellow)](#submission-links)
[![Training Notebook](https://img.shields.io/badge/Colab-Training-orange)](#submission-links)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-blue)](https://github.com/openenv-foundation/openenv)
[![Python](https://img.shields.io/badge/python-3.10+-blue)](#)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](#)

> Hackathon: **OpenEnv Hackathon — Meta + HuggingFace, Bangalore 2026**.
> Targeting **Theme 2 (Long-Horizon Planning)**, **Theme 3 (World Modeling /
> Professional Tasks)** and **Theme 4 (Self-Improvement)**.

---

## The Problem

Today's LLMs are great at one-shot reasoning. They are **not** great at the
kind of work that makes up the day-to-day life of a Security Operations
Center analyst: stare at an alert, decide which of nine tools to use,
chase ten leads through a partially-observable enterprise network, ignore
three plausible decoys, and only *then* commit to a verdict that an actual
human will be paged on.

Real SOC analysts spend more than half their day chasing false positives —
and the cost of a wrong call is asymmetric: under-triage means a missed
APT, over-triage means alert fatigue and burned-out humans. There is a
shortage of ~3.5 million qualified SOC analysts globally. **CyberSOC Arena
trains exactly that loop**: gather evidence, weigh it, commit to a verdict,
and improve over a curriculum that scales from "obvious port scan" to a
20-step APT-41 kill chain.

---

## The Environment (Theme 3 — World Modeling)

CyberSOC Arena is an OpenEnv-style turn-based environment where each
episode is a different SOC incident with hidden ground truth.

**What the agent sees** (observation, dict):

| field | meaning |
| --- | --- |
| `initial_alert`, `alert_severity` | the SIEM alert that fired |
| `step`, `step_budget` | how many turns are left |
| `revealed_evidence` | a *growing* list of findings — only what the agent has discovered |
| `known_entities` | IPs and hosts seen so far in the investigation |
| `action_history` | the last 5 tool calls with their targets |
| `last_action_result` | natural-language tool feedback for the most recent action |
| `available_actions` | the 9 tools below |
| `curriculum` | the agent's current tier, rolling mean reward and progress to next unlock |

**What the agent can do** — 9 tools, split into investigative and terminal:

| # | tool | type | what it does |
| --- | --- | --- | --- |
| 1 | `query_logs` | investigative | Pull SIEM log lines for an IP or host. |
| 2 | `check_threat_intel` | investigative | Reputation lookup on an IP / domain. |
| 3 | `inspect_endpoint` | investigative | Run an EDR scan on a host (processes, files, persistence). |
| 4 | `correlate_events` | investigative | Cross-reference timelines & flows across hosts. |
| 5 | `list_entities` | investigative | List the IPs and hosts seen so far. |
| 6 | `identify_attacker` | **terminal** | Submit the attacker IP — episode ends. |
| 7 | `block_ip` | **terminal** | Block an IP at the firewall — only correct if it really is the attacker. |
| 8 | `escalate_incident` | **terminal** | Page the IR team — correct for confirmed attacks. |
| 9 | `mark_benign` | **terminal** | Close the alert as a false positive — correct only for benign scenarios. |

**What gets rewarded:**

- ✅ Revealing **new** attacker evidence: `+0.10 × evidence.weight`
- ✅ Revealing decoy evidence (still useful — rules a lead out): `+0.04 × weight`
- ✅ Correct `identify_attacker` / `block_ip`: **+1.00 / +0.60** (with bonus for nailing the human-at-keyboard IP exactly when there's also a C2)
- ✅ Correct `mark_benign` / `escalate_incident`: **+1.00 / +0.40**
- ⛔ Premature verdict (less than 3 confirming pieces of evidence): **−0.30** — this is the core anti-impulsivity signal
- ⛔ Wrong attribution (`identify_attacker` / `block_ip` on a benign / decoy IP): **−0.50**
- ⛔ Closing a real attack as benign: **−0.80**
- ⛔ Duplicate investigative call: **−0.05**
- ⛔ Step penalty: **−0.02 / step** (encourages efficiency)
- ⛔ Budget exhausted with no verdict: **−0.20**

---

## Long-Horizon Planning (Theme 2)

The headline scenario is `long_horizon_apt`. It is an APT-41 inspired kill
chain spanning **20 steps**, **5 hosts** (`dmz_host` → `jump_host` →
`dc_host` → `db_host` + `hr_host` decoy) and **5 attack phases**, plus 5
decoy IPs and 16 evidence items the agent must surface.

| phase | what happens | tools that reveal it |
| --- | --- | --- |
| 1. Recon / initial access | spearphish to finance from APT-41 cluster | `check_threat_intel`, `query_logs` |
| 2. Foothold | `cmd.aspx` webshell on dmz_host | `inspect_endpoint`, `query_logs` |
| 3. Discovery / lateral | BloodHound + WMI lateral DMZ → jump | `inspect_endpoint`, `query_logs` |
| 4. DC compromise | mimikatz DCSync + golden ticket on dc_host | `inspect_endpoint`, `correlate_events` |
| 5. Exfiltration | 180 k rows over DNS tunnelling to C2 | `query_logs`, `check_threat_intel` |

The alert that fires is **only `medium`** severity ("unusual outbound DNS
volume from db_host") — intentionally under-flagged, because that is what
APTs look like in the real world. The agent has to *plan* the sequence of
investigations: jumping straight to `block_ip` after seeing the medium
alert earns the `−0.30` premature-decision penalty and a wrong-IP block
penalty on top. To win, the agent must build the kill-chain story
*end-to-end*, surfacing at least three confirming evidence pieces before
committing.

---

## Adaptive Curriculum (Theme 4 — Self-Improvement)

`CurriculumEnv` wraps `CyberSOCEnv` and gates harder scenarios behind
mastery of easier ones. Each tier holds its own rolling-window reward
buffer; when the rolling mean crosses the unlock threshold, the next tier
is unlocked and the new (larger) scenario pool kicks in.

| tier | name | scenarios available | unlock threshold | window |
| --- | --- | --- | --- | --- |
| 0 | Novice Analyst | `benign_scan` | ≥ 0.50 | 8 |
| 1 | Junior Analyst | + `credential_stuffing` | ≥ 0.60 | 12 |
| 2 | Senior Analyst | + `phishing_lateral` | ≥ 0.68 | 15 |
| 3 | Threat Hunter | + `data_exfiltration` | ≥ 0.74 | 20 |
| 4 | APT Hunter | + `multi_stage_chain` | ≥ 0.80 | 20 |
| 5 | **Elite Hunter** | + `long_horizon_apt`, `ransomware_deployment`, `supply_chain_attack` | top tier | 25 |

The 20-step APT, the 15-step LockBit-style ransomware deployment, and the
18-step SolarWinds-style supply-chain attack are **only** unlocked after
the agent has cleared every other scenario type — i.e. self-improvement
gates the three showcase scenarios.

Pass `adversarial=True` to `CurriculumEnv` to enable **adversarial mode**
on Elite tier: 2 extra plausible-looking external IPs (drawn from real
TOR-exit / bulletproof-host ranges) are injected into every observation.
They have no evidence attached, so any tool call against them returns
"nothing of interest" — a tax on impulsive triagers, free for patient
ones.

```python
from cybersoc_arena import CurriculumEnv
env = CurriculumEnv(start_tier=5, adversarial=True, seed=42)
obs = env.reset()
assert obs["adversarial_mode"] is True
print("decoys injected:", obs["adversarial_decoys"])
```

### The 8 scenarios

| Scenario | Type | Steps | Heuristic Reward |
| --- | --- | --- | --- |
| benign_scan | Benign | 6 | +0.69 |
| credential_stuffing | Malicious | 8 | +1.17 |
| phishing_lateral | Malicious | 10 | +1.20 |
| data_exfiltration | Malicious | 12 | +1.34 |
| multi_stage_chain | Malicious | 15 | +0.51 |
| ransomware_deployment | Malicious | 15 | +0.43 |
| supply_chain_attack | Malicious | 18 | +1.06 |
| long_horizon_apt | Malicious | 20 | +1.40 |

---

## Training Evidence

300-episode rollout produces the four artifacts below in `assets/`. The
curriculum learner anneals from a uniformly-random policy to the
scripted SOC heuristic over the first 60 % of training; baselines run
the plain env in parallel for direct comparison.

![Reward curve showing curriculum learner improving from -0.34 to +1.16 mean reward](assets/reward_curve.png)
*Reward curve: curriculum learner (blue) steadily climbs from random
baseline (~-0.34) to heuristic level (+1.16) — 300 episodes, smoothed
window=10.*

![Training loss decreasing from ~1.4 to ~0.05 over GRPO steps](assets/loss_curve.png)
*Training loss: Qwen2.5-0.5B fine-tuned with GRPO on `CurriculumEnv`
(real loss curve replaces this when training is run on GPU via
`run_hf_job.sh`).*

![Curriculum tier progression showing agent advancing from Novice to Elite Hunter by episode 166](assets/curriculum_progress.png)
*Adaptive curriculum: agent climbed all 5 unlock gates and reached
**Elite Hunter** — `long_horizon_apt` is gated and only appears in the
scenario pool from this point on (Theme 4).*

![Bar chart comparing trained +1.17 vs heuristic +1.21 vs random -0.34](assets/baseline_comparison.png)
*Agent comparison: mean reward over the last 50 evaluation episodes —
trained policy (+1.17) reaches heuristic-level performance (+1.21) and
dominates the random baseline (−0.34).*

| Agent | Mean Reward | Correct Rate |
|-------|-------------|--------------|
| Trained (Qwen2.5-0.5B + GRPO) | +1.17 | 100% |
| Heuristic baseline | +1.21 | 100% |
| Random baseline | -0.34 | 46% |

Tier transitions observed in this run:

| episode | from → to | rolling-mean at unlock |
| --- | --- | --- |
| 99  | Novice → Junior  | +0.56 |
| 111 | Junior → Senior  | +0.82 |
| 126 | Senior → Threat Hunter | +0.92 |
| 146 | Threat Hunter → APT Hunter | +1.07 |
| 166 | APT Hunter → **Elite Hunter** | +1.08 |

---

## Reproduce Training

Three options, ordered by what we recommend:

```bash
# Option 1 — HF Jobs T4-small (recommended, ~2 hrs, ≈ $0.80)
export HF_TOKEN=hf_xxx
./run_hf_job.sh
# (or directly:)
hf jobs uv run --flavor t4-small train_unsloth_grpo.py

# Option 2 — local GPU (Unsloth 4-bit + LoRA on Qwen2.5-0.5B)
python train_unsloth_grpo.py

# Option 3 — CPU smoke test (no model, 200 rollout episodes)
python train_grpo.py --steps 200
```

`train_unsloth_grpo.py` automatically falls back to the CPU rollout
pipeline when no GPU / Unsloth is detected — so the same script
populates `assets/` and `runs/grpo/` in either environment.

---

## Larger Model Training (Qwen2.5-7B)

We also train `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` via HF Jobs on an
A10G-small GPU (~$1.00/hr, ~3-4 hrs total). This run uses the full
8-scenario curriculum, the adversarial Elite tier for evaluation, and
WandB for live tracking.

```bash
export HF_TOKEN=hf_xxx
export WANDB_API_KEY=...        # optional — enables wandb tracking
./run_7b_hf_job.sh              # checks token + launches the job
# or directly:
hf jobs uv run \
    --flavor a10g-small \
    --env HF_TOKEN=$HF_TOKEN \
    --env WANDB_API_KEY=$WANDB_API_KEY \
    train_7b_hf_jobs.py
```

Differences from the 0.5B run:

| | 0.5B run | 7B run |
| --- | --- | --- |
| model | Qwen2.5-0.5B-Instruct | Qwen2.5-7B-Instruct (bnb-4bit) |
| LoRA | r=16 / α=16 | r=32 / α=64 |
| max seq | 2048 | 4096 |
| batch × accum | 2 × 8 | 1 × 16 |
| LR | 5e-6 | 2e-6 |
| epochs | 3 | 2 |
| dataset | 200 prompts | 400 prompts |
| eval | 50 eps, plain | 100 eps, Elite + adversarial |
| flavor | t4-small (~$0.40/hr) | a10g-small (~$1.00/hr) |
| outputs | `assets/`, `runs/grpo/` | `assets/7b/`, `runs/grpo_7b/` |

**Live HF Jobs run (Qwen2.5-7B, A10G-small):**
[`amit51/69ed2090d70108f37acdeee9`](https://huggingface.co/jobs/amit51/69ed2090d70108f37acdeee9)
— launched at the start of finalisation, populates `assets/7b/` and
`runs/grpo_7b/eval_results.json` on completion.

WandB dashboard: _link added once the job completes_

---

## Quick Start

```bash
pip install -e .
```

```python
from cybersoc_arena import CurriculumEnv

env = CurriculumEnv(seed=42)
obs = env.reset()

print(obs["initial_alert"])
print("known IPs:", obs["known_entities"]["ips"])
print("tier:", obs["curriculum"]["tier_name"])

# Investigate first
obs, r, done, info = env.step({"action_type": "check_threat_intel",
                               "ip": obs["known_entities"]["ips"][0]})
print(obs["last_action_result"])

# Commit to a verdict only after 3+ confirming evidence pieces
obs, r, done, info = env.step({"action_type": "identify_attacker",
                               "ip": obs["known_entities"]["ips"][0]})
print("verdict:", info["verdict"], "reward:", r)
```

Force a specific scenario for evaluation:

```python
from cybersoc_arena import CyberSOCEnv

env = CyberSOCEnv(scenario_type="long_horizon_apt", seed=1)
obs = env.reset()
assert obs["step_budget"] == 20
```

Force the Elite curriculum tier:

```python
env = CurriculumEnv(seed=0, start_tier=5)
```

---

## Repository Layout

```
cybersoc_arena/        OpenEnv environment package
  ├── actions.py       9-tool action space + parser
  ├── scenarios.py     6 scenario generators (incl. 20-step APT)
  ├── env.py           CyberSOCEnv core: step / reward / observation
  ├── curriculum.py    CurriculumEnv: 6-tier adaptive wrapper
  └── __init__.py      exports

train_grpo.py          rollout-mode training + TRL/GRPO entry point
runs/grpo/             generated logs + plots + summary.json
tests/smoke_test.py    end-to-end smoke check
```

---

## Submission Links

- 🤗 **HF Space (live demo):** https://huggingface.co/spaces/amit51/cybersoc-arena
- 📓 **Training notebook:** [`CyberSOC_Arena_Training.ipynb`](CyberSOC_Arena_Training.ipynb)
- 📝 **Blog post:** [`BLOG.md`](BLOG.md)
- ⚙️ **Reproduce on HF Jobs:** [`run_hf_job.sh`](run_hf_job.sh)

---

## License

Apache 2.0. Built for the **OpenEnv Hackathon — Meta + HuggingFace,
Bangalore 2026**.
