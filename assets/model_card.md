---
base_model: Qwen/Qwen2.5-1.5B-Instruct
library_name: peft
tags:
  - openenv
  - cybersoc-arena
  - grpo
  - cybersecurity
  - llm-agent
  - tool-use
  - long-horizon
  - reinforcement-learning
license: apache-2.0
language:
  - en
pipeline_tag: text-generation
datasets:
  - amit51/cybersoc-arena
---

# CyberSOC Arena - GRPO-trained Qwen2.5-1.5B-Instruct (LoRA)

LoRA adapter trained with `trl.GRPOTrainer` against the live
[CyberSOC Arena](https://huggingface.co/spaces/amit51/cybersoc-arena)
OpenEnv environment on a single **NVIDIA L40S 48GB** via Hugging Face Jobs.

> Submitted to the **OpenEnv Hackathon Round 2** (Meta x Hugging Face x PyTorch, Bangalore 2026).
> Mini-blog: [BLOG.md on the Space](https://huggingface.co/spaces/amit51/cybersoc-arena/blob/main/BLOG.md)
> Code: <https://github.com/AmitChowdary122/cyber-openenv>

---

## What this adapter does

Qwen2.5-1.5B-Instruct trained to act as a **Tier-2 SOC analyst** in the
CyberSOC Arena environment. The base model emits one of nine SOC tools per
step (`investigate_ip`, `query_logs`, `inspect_endpoint`, `check_threat_intel`,
`correlate_events`, `identify_attacker`, `isolate_host`, `escalate_incident`,
`close_as_benign`) on the right IP/host, gathers cross-host evidence, and
commits a final verdict under a step budget.

## Training

| Setting | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Method | TRL `GRPOTrainer` + LoRA (r=16, alpha=32, q/k/v/o_proj) |
| Reward | Live `CyberSOCEnv` per-step env reward (replayed in fresh reset per completion) |
| Hardware | 1x **NVIDIA L40S 48GB** (Hugging Face Jobs, `--flavor l40sx1`) |
| Wall clock | ~2 hr (480 prompts x 3 epochs x 8 generations = 360 GRPO steps) |
| Cost | ~$3.60 of the $30 hackathon credit |
| Final train loss | 2.76e-4 |
| Reward range | -0.23 (init) -> +0.15 to +0.40 (epoch 2.5+) |

### Training reward curve (the headline plot)

![GRPO reward curve](grpo_reward_curve.png)

Mean per-step environment reward across 720 GRPO log points (480 prompts x
3 epochs / per-device batch). Clear monotonic climb from ~-0.20 at init
to a stable +0.15 to +0.40 band by step ~500 -- the policy is learning to
pick the right SOC tool on the right target from live env reward alone,
with **no SFT warm-start**.

### GRPO loss curve

![GRPO loss curve](grpo_loss_curve.png)

GRPO's KL-regularized policy-gradient surrogate loss. **For GRPO, this
loss drifting up while reward goes up is the correct signal**, not
divergence: the loss measures how far the policy has moved from the frozen
reference (the KL penalty). A flat-zero loss would mean the policy isn't
updating. Magnitudes (1e-4 to 8e-4) are normal LoRA-GRPO scale.

---

## Held-out evaluation: BEFORE vs AFTER

Greedy rollout, 4 episodes per scenario, identical seeds before and after
training:

| Scenario | Qwen2.5-1.5B (BEFORE) | + GRPO (AFTER) | Delta |
|---|---:|---:|---:|
| `benign_scan`         | -1.96 | -2.07 | -0.10 |
| `phishing_lateral`    | -1.99 | -2.00 | -0.01 |
| `credential_stuffing` | -2.13 | -2.00 | **+0.13** |
| `data_exfiltration`   | -2.61 | -2.30 | **+0.31** |
| `multi_stage_chain`   | -2.70 | -2.30 | **+0.40** |
| `long_horizon_apt`    | -3.30 | -3.30 | 0.00 |
| **Mean**              | **-2.45** | **-2.33** | **+0.12** |

![Per-scenario before/after](grpo_baseline_compare.png)

The lifts concentrate on the harder, multi-evidence scenarios where
correct cross-host correlation pays off (`multi_stage_chain` +0.40,
`data_exfiltration` +0.31, `credential_stuffing` +0.13) -- exactly the
SOC behaviours the env is built to test. Easy and 20-step-budget
scenarios are flat in 360 GRPO steps.

---

## Files in this repo

| File | What it is |
|---|---|
| `adapter_model.safetensors` | LoRA weights (17.5 MB) |
| `adapter_config.json` | PEFT config |
| `tokenizer.json`, `tokenizer_config.json` | Tokenizer for Qwen2.5-1.5B-Instruct |
| `training_log.json` | Per-step `loss`, `reward`, `completion_length` (360 entries) |
| `eval_results.json` | BEFORE/AFTER greedy rollout numbers, per scenario |
| `grpo_reward_curve.png` | Headline reward curve over training |
| `grpo_loss_curve.png` | KL-regularized GRPO loss over training |
| `grpo_baseline_compare.png` | BEFORE vs AFTER bar chart per scenario |

---

## Quickstart

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = "Qwen/Qwen2.5-1.5B-Instruct"
adapter = "amit51/cybersoc-arena-qwen2.5-1.5b-grpo"

tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype="bfloat16", device_map="auto")
model = PeftModel.from_pretrained(model, adapter)

# Drive the live CyberSOC Arena env with this policy:
import asyncio
from cybersoc_arena import CyberSOCAsyncClient, CyberAction

async def play_one():
    async with CyberSOCAsyncClient(
        base_url="https://amit51-cybersoc-arena.hf.space",
    ) as env:
        obs = await env.reset(seed=42, scenario_type="multi_stage_chain")
        while not obs.done:
            prompt = render_obs_as_prompt(obs)   # see scripts/train_hf_job.py
            inp = tok(prompt, return_tensors="pt").to(model.device)
            out = model.generate(**inp, max_new_tokens=192, do_sample=False)
            text = tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
            action = parse_llm_action(text)      # tolerant JSON parser
            obs = await env.step(action)
        print(f"Done. Reward = {obs.reward:+.2f}, Evidence = {obs.evidence_count}")

asyncio.run(play_one())
```

`render_obs_as_prompt` and `parse_llm_action` are provided in
[`scripts/train_hf_job.py`](https://huggingface.co/spaces/amit51/cybersoc-arena/blob/main/scripts/train_hf_job.py)
on the env's HF Space.

## Reproducing this run

```bash
git clone https://github.com/AmitChowdary122/cyber-openenv && cd cyber-openenv
hf auth login                     # write-scope token
bash scripts/run_hf_job_l40s.sh   # default: 1x L40S 48GB, ~$3.60
```

That kicks off the same training run, pushes the adapter + plots + logs to
your own model repo when done. Override the GPU with `FLAVOR=h200` (faster
but $5/hr) or `FLAVOR=a100-large` (longer queue right now in India daytime).

## Training schedule (defaults in launcher)

| Hyperparam | Value |
|---|---|
| `--num-prompts` | 480 |
| `--num-generations` | 8 |
| `--epochs` | 3 |
| `--max-completion-length` | 192 |
| `--per-device-batch-size` | 4 |
| `--gradient-accumulation-steps` | 4 |
| LoRA `r` / `alpha` | 16 / 32 |
| LoRA target modules | `q_proj, k_proj, v_proj, o_proj` |

## Hackathon themes hit

- **Theme 2 - Super Long-Horizon Planning** (20-step APT across 5 hosts in `long_horizon_apt`)
- **Theme 3.1 - World Modeling / Professional Tasks** (real SOC tool use, partially observable)
- **Theme 4 - Self-Improvement** (`CurriculumEnv` adaptive 6-tier scenario unlock)

## Safety

CyberSOC Arena is a **defensive** simulator. Synthetic logs, synthetic IPs,
synthetic finding text. No real exploit code, no malware behaviour, no
attack instructions. The trained agent learns to triage and contain
incidents, not to attack systems.

## License

Apache-2.0 (matches base model and env).

## Citation

```bibtex
@misc{cybersoc-arena-grpo-2026,
  author       = {Amit Chowdary},
  title        = {CyberSOC Arena: a SOC-analyst OpenEnv environment, with GRPO-trained Qwen2.5-1.5B},
  year         = {2026},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/amit51/cybersoc-arena-qwen2.5-1.5b-grpo}},
  note         = {Built for the OpenEnv Hackathon Round 2 (Meta x Hugging Face x PyTorch, Bangalore)}
}
```
