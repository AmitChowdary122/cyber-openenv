# 2-Minute Pitch -- CyberSOC Arena

> Speaker script for a 2-minute submission video. Aim for energetic, pause
> on each section break.

---

**[0:00 - 0:15] Hook**

"Random gets 8 percent. A trained 4-action meta-policy, in twelve seconds
of CPU training, doubles that. We built **CyberSOC Arena** -- an OpenEnv
environment where an LLM acts as a Tier-2 SOC analyst, triages
real-world-style alerts, and learns to investigate *before* deciding."

**[0:15 - 0:35] Why this and not another grid world**

"The hackathon brief is sharp: judges have seen too many chess and tic-tac-toe
clones. So we picked the one professional skill LLMs are spectacularly bad at
out of the box: discipline under uncertainty. Real Tier-2 analysts get paid
for picking the right one of nine tools, on the right IP, long enough to
gather corroborating evidence, and not committing on the loudest signal --
because the loudest signal is usually a decoy. CyberSOC Arena is the first
OpenEnv environment that tests all four behaviours at once."

**[0:35 - 1:00] Environment**

"Six stochastic scenarios. A benign internet scan that punishes you with
minus 1.5 if you isolate it. A 12-step phishing-and-lateral chain. A
credential-stuffing flood with one real attacker hidden among decoys. Slow
TLS exfiltration. A short kill chain. And the marquee: a **20-step
long-horizon APT across 5 hosts** with three carefully-tuned decoys
including an authorised red-team scanner. Nine analyst tools: investigate,
query logs, inspect endpoint, threat intel, correlate, isolate, escalate,
identify, close-as-benign."

**[1:00 - 1:25] Reward signal + curriculum**

"Reward is dense, bounded, and hard to game. Per step you get paid for
new evidence, penalised for repeating yourself. The terminal carries the
bulk: plus or minus 1.5 for attribution, plus a 0.30 evidence-quality
bonus that *only* activates with three or more attacker-confirming
pieces. Closing a real incident as benign costs you 1.5 -- the worst
single action, exactly as in a real SOC. And we ship a `CurriculumEnv`
wrapper that *unlocks* harder scenarios as the agent's rolling reward
crosses tier thresholds: Novice analyst, Junior responder, Mid-level,
Senior, Lead, APT hunter. That's our Theme 4 self-improvement story."

**[1:25 - 1:50] Results**

"On the held-out evaluation set, the random meta-policy gets minus 1.57
mean reward and 8.3 percent success. Our REINFORCE-trained meta-policy
gets minus 1.23 and 16.7 percent -- success rate doubles. And the
aggregate hides where the policy *really* shines: on the benign-scan
scenario, the trained policy reaches plus 1.17 cumulative reward, versus
random's minus 0.32 -- a one-and-a-half-point lift. It learned the most
expensive analyst skill there is: *don't isolate the internet scanner*.
The plots, the per-scenario breakdown, and the curriculum-tier-over-time
chart are all in the README, regenerable from `python train_reinforce.py`
in twelve seconds."

**[1:50 - 2:00] Close**

"Pip install the env. Run `uv run server` for the standard /reset, /step,
/state OpenEnv surface plus the /web HumanAgent UI. The Colab notebook
trains a real Qwen2.5-0.5B with TRL GRPO on a free T4. Code, plots,
HF Space, and submission checklist are linked below. Thanks."

---

## Talking-points cheat sheet (Q&A)

- **Innovation (40%)**: 6 stochastic scenarios + 20-step APT + decoys
  with non-zero evidence weight + adaptive curriculum + a domain
  underexplored in RL/LLM training (cybersecurity defence).
- **Reward signal**: dense, bounded to [-2, 2] per step, anti-gaming
  (premature-decision penalty, repeat penalty, evidence-quality bonus
  only at >=3 attacker pieces, decoy reward strictly less than real-evidence
  reward).
- **Improvement evidence**: REINFORCE 8.3% -> 16.7% success in 12 seconds
  CPU; benign_scan goes -0.32 -> +1.17. Curves committed as PNG in
  `assets/`, embedded in README.
- **Pipeline**: TRL GRPOTrainer + Qwen2.5-0.5B + LoRA on free Colab T4
  (notebook); HF Jobs launcher (PEP-723 inline deps) for the same run on
  the $30 hackathon credit.
- **Engineering**: subclasses `openenv.core.env_server.Environment`,
  uses `create_fastapi_app(...)`, `EnvClient` subclass for async, valid
  `openenv.yaml`, no reserved tool names, client/server separation
  preserved.
- **Reproducibility**: every plot regenerable from
  `python train_reinforce.py --episodes 3000`; demos via
  `python demo_run.py`, `python demo_long_horizon.py`,
  `python demo_curriculum.py`.

## Demo flow if asked to live-demo

1. `python demo_long_horizon.py` -- 20-step APT walkthrough (~1s).
2. `python train_reinforce.py --episodes 1000` -- live training,
   prints curves, finishes in ~5 sec.
3. `uv run server` -- open `http://localhost:8000/web` and click through
   the HumanAgent UI for one episode.
