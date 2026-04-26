# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "cyber-openenv @ git+https://huggingface.co/spaces/amit51/cybersoc-arena",
#   "openenv-core>=0.2.3",
#   "transformers==4.55.2",
#   "trl==0.18.0",
#   "datasets==3.6.0",
#   "accelerate>=1.0,<1.10",
#   "peft>=0.13,<0.16",
#   "bitsandbytes>=0.43",
#   "matplotlib>=3.7",
#   "pydantic>=2.0",
#   "huggingface_hub>=0.34",
# ]
# ///

"""GRPO training for CyberSOC Arena, runnable on Hugging Face Jobs (L40S).

This is the headline cloud training run. It trains Qwen2.5-1.5B-Instruct
on a single L40S 48GB (default) and pushes EVERYTHING back to a
persistent HF Hub repo before the job's container is torn down:

  * LoRA adapter weights (so the trained agent can be reloaded)
  * training_log.json              -- per-step reward, loss, completion length
  * eval_results.json              -- before/after random vs trained
  * grpo_loss_curve.png            -- TRL GRPO loss over steps
  * grpo_reward_curve.png          -- mean per-step env reward over steps
  * grpo_baseline_compare.png      -- trained vs untrained on each scenario

Launch with:

    bash scripts/run_hf_job_a100.sh

or directly:

    hf jobs uv run --flavor l40sx1 --secrets HF_TOKEN \\
      https://huggingface.co/spaces/amit51/cybersoc-arena/raw/main/scripts/train_hf_job.py \\
      --base-model Qwen/Qwen2.5-1.5B-Instruct \\
      --num-prompts 480 --num-generations 8 --epochs 3 \\
      --max-completion-length 192 \\
      --output-dir /tmp/cybersoc_grpo \\
      --push-to-hub amit51/cybersoc-arena-qwen2.5-1.5b-grpo

Cost on the $30 hackathon credit
--------------------------------
- l40sx1 (1x L40S 48GB) is $1.80/hr.
- A 480-prompt, 3-epoch, 8-generation run finishes in ~2 hr on L40S: about
  $3.60 per run, well within the $30 hackathon credit. The L40S queue is
  almost always empty in India daytime, while a100-large can sit 30+ min.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys


os.environ["TRANSFORMERS_VERBOSITY"] = "error"
# ─────────────────────────────────────────────────────────────────────────────
# Prompt rendering and reward function
# ─────────────────────────────────────────────────────────────────────────────
def render_obs(obs) -> str:
    """Render a CyberObservation as the LLM-facing prompt string."""
    lines = [
        "You are a Tier-2 SOC analyst. Investigate the alert and decide.",
        "",
        f"ALERT [{obs.alert.severity.upper()}]: {obs.alert.summary}",
        f"Step {obs.step}/{obs.step_budget}  (remaining: {obs.remaining_steps})",
        f"Hosts: {obs.asset_inventory.hosts}",
        f"Visible IPs: {obs.asset_inventory.visible_ips}",
        "",
        f"Evidence so far ({obs.evidence_count} items):",
    ]
    for e in obs.evidence_collected[-6:]:
        lines.append(f"  - [{e.action}({e.target})] {e.finding[:140]}")
    lines += [
        "",
        f"Available actions: {obs.available_actions}",
        "",
        "Reply with ONE JSON action, e.g. "
        '{"action_type": "query_logs", "ip": "10.0.0.5"}',
    ]
    return "\n".join(lines)


def build_prompt_dataset(SCENARIO_TYPES, CyberSOCEnv, n_prompts: int, seed: int):
    """Build a prompt dataset by sampling fresh starting observations."""
    from datasets import Dataset
    random.seed(seed)
    rows = []
    for k in range(n_prompts):
        env = CyberSOCEnv()
        scen = random.choice(SCENARIO_TYPES)
        obs = env.reset(seed=seed + k, scenario_type=scen)
        rows.append({
            "prompt": render_obs(obs),
            "scenario": scen,
            "seed": seed + k,
        })
    return Dataset.from_list(rows)


def make_reward_function(SCENARIO_TYPES, CyberSOCEnv, CyberAction, parse_action):
    """The TRL `reward_funcs` callback. Replays each completion in a fresh env."""
    def env_reward(prompts, completions, **kwargs):
        rewards = []
        scenarios = kwargs.get("scenario", [None] * len(completions))
        seeds = kwargs.get("seed", [None] * len(completions))
        for i, completion in enumerate(completions):
            text = completion if isinstance(completion, str) \
                else completion[0]["content"]
            try:
                a = parse_action(text)
                ca = CyberAction(
                    action_type=a.action_type, ip=a.ip, host=a.host,
                    entity=a.entity, summary=a.summary,
                )
            except Exception:
                rewards.append(-0.10)
                continue
            try:
                env = CyberSOCEnv()
                scen = scenarios[i] if i < len(scenarios) and scenarios[i] else \
                    random.choice(SCENARIO_TYPES)
                seed = seeds[i] if i < len(seeds) and seeds[i] is not None else \
                    random.randint(0, 1_000_000)
                env.reset(seed=int(seed), scenario_type=scen)
                obs = env.step(ca)
                rewards.append(float(obs.reward))
            except Exception:
                rewards.append(-0.20)
        return rewards
    return env_reward


# ─────────────────────────────────────────────────────────────────────────────
# Eval rollouts (before/after) for the comparison plot
# ─────────────────────────────────────────────────────────────────────────────
def rollout(model, tok, SCENARIO_TYPES, CyberSOCEnv, CyberAction, parse_action,
            n_per_scenario: int = 6, max_steps_per_ep: int = 25,
            seed_base: int = 999):
    """Run greedy rollouts of `model` on every scenario; return per-scenario rewards."""
    import torch
    out = {s: [] for s in SCENARIO_TYPES}
    for scen in SCENARIO_TYPES:
        for k in range(n_per_scenario):
            env = CyberSOCEnv()
            obs = env.reset(seed=seed_base + k, scenario_type=scen)
            total = 0.0
            steps = 0
            while not obs.done and steps < max_steps_per_ep:
                prompt = render_obs(obs)
                ids = tok(prompt, return_tensors="pt", truncation=True,
                          max_length=900).to(model.device)
                with torch.no_grad():
                    gen = model.generate(
                        **ids, max_new_tokens=80, do_sample=False,
                        pad_token_id=tok.eos_token_id,
                    )
                text = tok.decode(gen[0][ids["input_ids"].shape[1]:],
                                   skip_special_tokens=True)
                try:
                    a = parse_action(text)
                    ca = CyberAction(
                        action_type=a.action_type, ip=a.ip, host=a.host,
                        entity=a.entity, summary=a.summary,
                    )
                    obs = env.step(ca)
                except Exception:
                    obs = env.step(CyberAction(
                        action_type="query_logs",
                        ip=obs.asset_inventory.visible_ips[0]
                            if obs.asset_inventory.visible_ips else "10.0.0.1",
                    ))
                total += obs.reward
                steps += 1
            out[scen].append(total)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────
def plot_curves(log, eval_results, out_dir, model_label="Qwen2.5-1.5B"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    os.makedirs(out_dir, exist_ok=True)

    steps_l = [r["step"] for r in log if "loss" in r]
    losses = [r["loss"] for r in log if "loss" in r]
    rsteps = [r["step"] for r in log if "reward" in r]
    rewards = [r["reward"] for r in log if "reward" in r]

    if losses:
        fig, ax = plt.subplots(figsize=(8.4, 4.4))
        ax.plot(steps_l, losses, color="#7570b3", lw=1.6, label="GRPO loss")
        ax.set_xlabel("Training step (#)")
        ax.set_ylabel("Loss")
        ax.set_title(f"CyberSOC Arena - GRPO loss ({model_label}, L40S)")
        ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "grpo_loss_curve.png"), dpi=130)
        plt.close(fig)

    if rewards:
        fig, ax = plt.subplots(figsize=(8.4, 4.4))
        ax.plot(rsteps, rewards, color="#1b9e77", lw=1.6, label="mean reward")
        ax.set_xlabel("Training step (#)")
        ax.set_ylabel("Mean per-step env reward")
        ax.set_title(f"CyberSOC Arena - GRPO reward ({model_label}, L40S)")
        ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "grpo_reward_curve.png"), dpi=130)
        plt.close(fig)

    if eval_results and "before" in eval_results and "after" in eval_results:
        scens = list(eval_results["before"].keys())
        before = [float(np.mean(eval_results["before"][s]))
                  if eval_results["before"][s] else 0.0 for s in scens]
        after = [float(np.mean(eval_results["after"][s]))
                 if eval_results["after"][s] else 0.0 for s in scens]
        fig, ax = plt.subplots(figsize=(9.5, 4.7))
        x = np.arange(len(scens)); w = 0.36
        ax.bar(x - w / 2, before, w, label=f"Untrained {model_label}",
               color="#d95f02")
        ax.bar(x + w / 2, after, w, label=f"GRPO-trained {model_label}",
               color="#1b9e77")
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("_", "\n") for s in scens], fontsize=9)
        ax.set_ylabel("Mean episode reward (greedy rollout)")
        ax.set_title(f"Per-scenario before/after GRPO  ({model_label})")
        ax.axhline(0, color="black", lw=0.8)
        ax.legend(loc="lower right"); ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "grpo_baseline_compare.png"), dpi=130)
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Push artifacts to a persistent HF Hub repo so the user keeps them
# ─────────────────────────────────────────────────────────────────────────────
def push_artifacts(repo_id: str, out_dir: str, model, tok,
                   model_label: str, base_model: str):
    """Push LoRA adapter + logs + plots to the model repo."""
    from huggingface_hub import HfApi, create_repo

    api = HfApi()
    create_repo(repo_id, repo_type="model", exist_ok=True)

    # 1. Push LoRA adapter + tokenizer
    print(f"[push] uploading LoRA adapter to {repo_id}")
    model.push_to_hub(repo_id)
    tok.push_to_hub(repo_id)

    # 2. Push training artifacts (JSON + PNGs)
    for fname in ("training_log.json", "eval_results.json",
                  "grpo_loss_curve.png", "grpo_reward_curve.png",
                  "grpo_baseline_compare.png"):
        p = os.path.join(out_dir, fname)
        if not os.path.exists(p):
            continue
        api.upload_file(
            path_or_fileobj=p,
            path_in_repo=fname,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Add {fname} from HF Jobs L40S run",
        )
        print(f"[push] uploaded {fname}")

    # 3. Auto-generate a model-card README.md describing the run
    md = f"""---
base_model: {base_model}
library_name: peft
tags:
  - openenv
  - cybersoc-arena
  - grpo
  - cybersecurity
  - llm-agent
license: apache-2.0
---

# CyberSOC Arena - GRPO-trained {model_label}

LoRA adapter trained with `trl.GRPOTrainer` against the live
[CyberSOC Arena](https://huggingface.co/spaces/amit51/cybersoc-arena)
OpenEnv environment on a single L40S 48GB via Hugging Face Jobs.

## Training run

- Base: `{base_model}`
- Method: GRPO + LoRA (r=16, alpha=32)
- Reward: live env per-step reward (replayed in a fresh `CyberSOCEnv`
  per completion, with the prompt's scenario+seed)
- Hardware: 1x L40S 48GB
- See `training_log.json` for per-step loss + reward.
- See `grpo_loss_curve.png`, `grpo_reward_curve.png`, and
  `grpo_baseline_compare.png` for the headline plots.

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("{base_model}", torch_dtype="auto")
model = PeftModel.from_pretrained(base, "{repo_id}")
tok = AutoTokenizer.from_pretrained("{repo_id}")
```

To roll out the trained model against the env over WebSocket:

```python
from cybersoc_arena import CyberSOCAsyncClient, CyberAction
async with CyberSOCAsyncClient(
    base_url="https://amit51-cybersoc-arena.hf.space"
) as env:
    obs = await env.reset(seed=42, scenario_type="long_horizon_apt")
    while not obs.done:
        text = your_inference(model, tok, obs)        # model.generate(...)
        action = CyberAction(**json.loads(text))
        obs = await env.step(action)
```

Submission for the OpenEnv Hackathon, Round 2 (Bangalore 2026).
"""
    api.upload_file(
        path_or_fileobj=md.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Add model card",
    )
    print(f"[push] uploaded README.md to {repo_id}")


# ─────────────────────────────────────────────────────────────────────────────
# main()
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--num-prompts", type=int, default=320)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--max-prompt-length", type=int, default=900)
    ap.add_argument("--max-completion-length", type=int, default=192)
    ap.add_argument("--per-device-batch-size", type=int, default=4)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=4)
    ap.add_argument("--output-dir", default="/tmp/cybersoc_grpo")
    ap.add_argument("--push-to-hub", default=None,
                    help="HF repo id for trained adapter + logs + plots")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-eval", action="store_true",
                    help="Skip the before/after rollout (saves ~3 min)")
    ap.add_argument("--skip-push", action="store_true",
                    help="Don't push artifacts (logs only locally)")
    args = ap.parse_args()

    # cybersoc_arena is declared in the PEP-723 dependencies block at the top
    # of this script, so `uv run` already installs it before main() executes.
    # If the import still fails here, fall back to `uv pip install` because
    # `uv run` venvs do NOT ship the standard `pip` module.
    try:
        import cybersoc_arena  # noqa
    except ImportError:
        import subprocess
        subprocess.check_call([
            "uv", "pip", "install", "--quiet",
            "git+https://huggingface.co/spaces/amit51/cybersoc-arena",
        ])
        import cybersoc_arena  # noqa: E402,F401

    from cybersoc_arena import (  # noqa: E402
        CyberSOCEnv, CyberAction, SCENARIO_TYPES,
    )
    from cybersoc_arena.actions import parse_action  # noqa: E402

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[hf-job] device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"[hf-job] loading {args.base_model}")
    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto",
    )

    # Pre-training rollout (untrained baseline)
    eval_results = {}
    if not args.skip_eval:
        print("[hf-job] eval rollout BEFORE training")
        eval_results["before"] = rollout(
            model, tok, SCENARIO_TYPES, CyberSOCEnv, CyberAction, parse_action,
            n_per_scenario=4, seed_base=10_000,
        )
        before_means = {s: (sum(v) / max(1, len(v)))
                        for s, v in eval_results["before"].items()}
        print("[hf-job] BEFORE per-scenario means:", before_means)

    # Build training set
    train_ds = build_prompt_dataset(
        SCENARIO_TYPES, CyberSOCEnv, args.num_prompts, args.seed,
    )
    print(f"[hf-job] dataset rows: {len(train_ds)}")

    reward_fn = make_reward_function(
        SCENARIO_TYPES, CyberSOCEnv, CyberAction, parse_action,
    )

    config = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        learning_rate=args.lr,
        logging_steps=2,
        report_to="none",
        bf16=True,
        save_strategy="no",
    )
    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tok,
        reward_funcs=reward_fn,
        args=config,
        train_dataset=train_ds,
        peft_config=peft_config,
    )
    print("[hf-job] training start")
    trainer.train()
    print("[hf-job] training done")

    # Save log
    log = trainer.state.log_history
    with open(os.path.join(args.output_dir, "training_log.json"), "w") as f:
        json.dump(log, f, indent=2)

    # Post-training rollout (trained baseline)
    if not args.skip_eval:
        print("[hf-job] eval rollout AFTER training")
        eval_results["after"] = rollout(
            trainer.model, tok, SCENARIO_TYPES, CyberSOCEnv, CyberAction,
            parse_action, n_per_scenario=4, seed_base=10_000,
        )
        after_means = {s: (sum(v) / max(1, len(v)))
                       for s, v in eval_results["after"].items()}
        print("[hf-job] AFTER per-scenario means:", after_means)
        with open(os.path.join(args.output_dir, "eval_results.json"), "w") as f:
            json.dump(eval_results, f, indent=2)

    # Plots
    label = args.base_model.split("/")[-1]
    plot_curves(log, eval_results, args.output_dir, model_label=label)

    # Push to Hub
    if args.push_to_hub and not args.skip_push:
        try:
            push_artifacts(args.push_to_hub, args.output_dir, trainer.model,
                            tok, model_label=label, base_model=args.base_model)
            print(f"[hf-job] artifacts pushed to https://huggingface.co/{args.push_to_hub}")
        except Exception as e:
            print(f"[hf-job][warn] push_to_hub failed: {e}")

    print("[hf-job] done. Artifacts in", args.output_dir)


if __name__ == "__main__":
    main()
