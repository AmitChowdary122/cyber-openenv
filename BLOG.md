# CyberSOC Arena: teaching LLMs the discipline of a Tier-2 SOC analyst

*An OpenEnv hackathon Round 2 submission (Meta x Hugging Face x PyTorch, Bangalore 2026).*

## TL;DR

We built **CyberSOC Arena** -- an OpenEnv environment where an LLM acts as
a Tier-2 SOC analyst. It ships with **6 stochastic scenarios** (5 short
incidents plus a 20-step long-horizon APT across 5 hosts), **9 SOC tools**,
a **dense + bounded + anti-gaming reward**, and an **adaptive 6-tier
curriculum wrapper** for self-improvement. A tiny CPU REINFORCE policy
trained for 12 seconds against the live `CurriculumEnv` doubles success
rate over a random meta-policy and reaches near-perfect performance on
benign scenarios -- the analyst skill that's *most* expensive to get
wrong in production.

## Why SOC investigation

The hackathon brief is sharp about innovation: "judges have seen a lot of
chess, snake, tic-tac-toe, and grid-world clones." So we picked a domain
that simultaneously demands four behaviours LLMs are bad at out of the box:

1. **Long-horizon planning under a step budget** -- 20 steps, 5 hosts, 5 phases.
2. **Real tool use** -- 9 distinct tools, each with its own target type.
3. **Evidence-grounded decision-making** -- the worst possible action is
   committing early when the strongest-looking signal is a decoy.
4. **False-positive suppression** -- internet scanners and authorised
   red-team scans look like attacks until you check threat intel.

Real Tier-2 analysts get paid for exactly this. A trained LLM that scores
well on CyberSOC Arena has measurably learned a *professional* skill --
and there's a published research line on it (LLM-driven SOC playbooks,
SOAR copilots, "agentic" incident response) that's wide open.

## What the environment looks like

Every reset draws a fresh, randomized scenario from one of six archetypes.
The agent sees:

- the original alert text + severity,
- an asset inventory (hosts visible to the analyst, plus the IPs the
  agent can target),
- a step budget,
- the evidence findings revealed so far (text only -- *never* the hidden
  `confirms_attacker` flags or weights),
- the action history,
- a small slice of background noise (unrelated log lines), and
- the JSON action schema.

Each step the agent emits one of:

| Tool | Targets | Notes |
|---|---|---|
| `query_logs` | IP | edge / DNS / SMB / proxy logs |
| `investigate_ip` | IP | broad cross-source pull |
| `inspect_endpoint` | host | EDR + persistence + process tree |
| `check_threat_intel` | IP | external attribution data |
| `correlate_events` | entity | links pairs of indicators |
| `identify_attacker` | IP | **terminal**: attribute the attack |
| `isolate_host` | host | **terminal**: contain a specific host |
| `escalate_incident` | summary text | **terminal**: keyword-graded handoff |
| `close_as_benign` | summary text | **terminal**: declare no incident |

Reward is dense and *intentionally hard to game*:

- **Per-step shaping**: -0.05 step penalty, -0.10 repeat penalty,
  +0.20 x weight for new attacker evidence, +0.05 x weight for
  decoy evidence (still positive -- exploration is good),
  +0.20 correlation bonus when a `correlate_events` lands on a real pair.
- **Terminal**: +/-1.5 for attribution, +1.20 for correctly closing a
  benign incident, **-1.50 for closing a real incident as benign** (the
  worst single action, exactly as in a real SOC), -0.30 premature-decision
  penalty if the agent commits with <2 evidence, +0.30 evidence-quality
  bonus only when >=3 attacker-confirming pieces are gathered.
- All per-step rewards clipped to **[-2, 2]** so a single bad action
  cannot poison a batch.

## Why the long-horizon APT scenario matters

The marquee scenario, `long_horizon_apt`, is a 20-step investigation across
5 hosts (edge -> workstation -> file server -> DB -> egress proxy) with a
5-phase kill chain (recon, initial access, persistence, lateral movement,
exfiltration). It carries **three carefully tuned decoys** -- an authorised
red-team scanner, a noisy internal backup service, and an external Shodan
crawler -- each with non-zero evidence weight, so the obvious early-game
moves all converge on the wrong attribution.

Reading the trajectory of a heuristic-driven walk-through (see
[`training_runs/demo_long_horizon.log`](training_runs/demo_long_horizon.log)):
the SOC playbook hits +0.35 at step 11 (correlation lands), and then
+1.80 at step 13 (`identify_attacker` fires correctly with the +0.30
evidence-quality bonus). Final: **+2.06 cumulative reward, 6 evidence
pieces revealed (5 attacker-confirming), correct attribution in a 20-step
budget.**

## Curriculum (Theme 4)

`CurriculumEnv` wraps the env and tracks a rolling mean of episode rewards.
When the mean crosses a tier threshold, the next tier unlocks:

| Tier | Name | Pool | Promote at rolling mean |
|---:|---|---|---:|
| 0 | Novice analyst   | benign_scan only                         | +0.50 |
| 1 | Junior responder | + phishing_lateral                       | +0.70 |
| 2 | Mid-level        | + credential_stuffing                    | +0.85 |
| 3 | Senior responder | + data_exfiltration                      | +0.95 |
| 4 | Incident lead    | + multi_stage_chain                      | +1.05 |
| 5 | APT hunter       | + long_horizon_apt (full mix, 20 steps)  | --    |

This implements the hackathon's Theme 4 ("agents that learn to drive their
own capability growth") in 230 lines of Python with no extra dependencies.

## Training

We ran two passes:

**1. Pure-CPU numpy REINFORCE (the headline curve, 12 seconds).**
A softmax over 4 *meta-actions* (INVESTIGATE / CORRELATE / IDENTIFY /
CLOSE_BENIGN) with hand-extracted features. Targets are picked by an
SOC-analyst heuristic that reads the finding text. After 3,000 episodes
on `CurriculumEnv`:

| Agent | Mean reward (60 eps) | Success | benign_scan reward |
|---|---:|---:|---:|
| Random meta-policy | -1.57 | 8.3% | -0.32 |
| **REINFORCE-trained** | **-1.23** | **16.7%** | **+1.17** |

The aggregate hides where the policy actually shines: on `benign_scan`
it goes from -0.32 to **+1.17** -- it has learned the most expensive
analyst skill, *don't isolate the internet scanner*.

**2. TRL `GRPOTrainer` + Qwen2.5-0.5B-Instruct + LoRA.**
Provided as a Colab notebook (`notebooks/CyberSOC_Arena_GRPO.ipynb`,
~25 min on a free T4) and as a Hugging Face Jobs launcher
(`scripts/run_hf_job.sh`, `~$0.10` on the $30 hackathon credit). The
reward function is the live env: each completion is parsed, replayed
on a fresh reset, and the per-step reward is returned to GRPO.

Both runs train against the *same* `CyberSOCEnv`, so the curves are
directly comparable.

## Engineering notes

- `CyberSOCEnv` subclasses `openenv.core.env_server.Environment[CyberAction,
  CyberObservation, CyberState]`.
- `cybersoc_arena.server` builds the FastAPI app via `create_fastapi_app(...)`,
  so the standard `/reset`, `/step`, `/state`, `/health`, `/docs`, `/web`,
  and `/ws` endpoints work without any custom HTTP code.
- `CyberSOCAsyncClient` subclasses `openenv.core.EnvClient` so TRL/Unsloth
  training loops can drive the env over WebSocket sessions.
- All Pydantic models inherit from the official `Action`, `Observation`,
  `State` base classes.
- Reserved MCP tool names (`reset`, `step`, `state`, `close`) are *not*
  used for any of the 9 SOC tools.
- `openenv.yaml` validates against `openenv-core >= 0.2.3`.

## What's next

- **Adaptive attackers**: have the scenario's attacker react to the agent's
  isolation moves so the kill chain branches mid-episode.
- **Multi-agent**: blue-team analyst vs. red-team simulator, with both
  trained against the same arena.
- **Bigger curriculum**: insider threat, supply-chain compromise,
  ransomware deployment, cloud-IAM abuse.

Code, plots, and the deployable HF Space are linked from the
[README](README.md).

-- *Built for the OpenEnv Hackathon, Round 2, Bangalore 2026.*
