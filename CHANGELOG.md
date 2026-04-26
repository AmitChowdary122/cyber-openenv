# Changelog

## v0.3.0-hackathon -- 2026-04-26

OpenEnv Hackathon Round 2 submission (Meta x Hugging Face x PyTorch, Bangalore 2026).

### Environment
- 6 stochastic scenario archetypes: `benign_scan`, `phishing_lateral`,
  `credential_stuffing`, `data_exfiltration`, `multi_stage_chain`,
  `long_horizon_apt` (the marquee 20-step APT across 5 hosts with 3 decoys).
- 9 SOC tools across 5 investigative + 4 terminal actions.
- Dense + bounded + anti-gaming reward function with 17 named components,
  per-step reward clipped to [-2, 2].
- `CyberSOCEnv` subclasses `openenv.core.env_server.Environment`.
- `cybersoc_arena.server.create_fastapi_app(...)` provides standard
  `/reset`, `/step`, `/state`, `/health`, `/docs`, `/web`, `/ws` endpoints.
- Pydantic `CyberAction` / `CyberObservation` / `CyberState` models inherit
  from the official OpenEnv base classes.

### Self-Improvement (Theme 4)
- `CurriculumEnv` adaptive 6-tier wrapper:
  Novice analyst -> Junior responder -> Mid-level -> Senior -> Incident lead -> APT hunter.
- Tier unlocks driven by rolling-mean reward against per-tier thresholds
  (+0.50 / +0.70 / +0.85 / +0.95 / +1.05).
- Optional drop-back via `ratchet=False`; default is monotone for benchmarks.

### Composable rewards (RFC 004)
- `cybersoc_arena.rubric.CyberSOCRubric`: idiomatic
  `openenv.core.rubrics.Rubric` tree with 17 introspectable leaves across
  two named subtrees (`step` x 6, `terminal` x 11).
- `rubric.named_breakdown(action, obs)` for credit assignment / ablation.
- `rubric.get_rubric("terminal.wrong_benign_close")` for path-based pulls.
- Non-invasive: does not replace `rewards.py`; `observation.reward` remains
  the canonical scalar.

### Training
- **CPU REINFORCE baseline** (`train_reinforce.py`): numpy-only, 12 sec on
  CPU, 3,000 episodes. Beats random meta-policy on all 6 scenarios.
- **GRPO on Qwen2.5-1.5B-Instruct + LoRA** (`scripts/train_hf_job.py`):
  - 1x NVIDIA L40S 48GB via Hugging Face Jobs (`--flavor l40sx1`)
  - 480 prompts x 3 epochs x 8 generations = 360 GRPO steps
  - 2-hour wall clock, ~$3.60 of the $30 hackathon credit
  - Training reward climbed from -0.23 -> +0.15-0.40 band by epoch 2.5+
  - Adapter + plots + logs pushed to
    [`amit51/cybersoc-arena-qwen2.5-1.5b-grpo`](https://huggingface.co/amit51/cybersoc-arena-qwen2.5-1.5b-grpo)
- **Per-scenario BEFORE/AFTER**: +0.40 on `multi_stage_chain`, +0.31 on
  `data_exfiltration`, +0.13 on `credential_stuffing`. Mean reward
  -2.45 -> -2.33.

### Plots committed
- `assets/grpo_reward_curve.png` -- headline GRPO reward curve (L40S)
- `assets/grpo_loss_curve.png` -- KL-regularized policy-gradient surrogate loss (L40S)
- `assets/grpo_baseline_compare.png` -- per-scenario BEFORE/AFTER bar chart
- `assets/curriculum_progress.png` -- 6-tier curriculum staircase (T0 -> T5)
- `assets/curriculum_combined.png` -- staircase + rolling-mean threshold panel
- `assets/reward_curve.png`, `loss_curve.png`, `baseline_comparison.png` -- REINFORCE baseline

### Tooling
- `scripts/run_hf_job_l40s.sh` -- one-command HF Jobs launcher (default `l40sx1`).
- `scripts/regenerate_plots.py` -- replot from `training_log.json` without re-training.
- `scripts/plot_curriculum_full.py` -- generate the 6-tier staircase plot.
- `scripts/push_all_to_space.py` -- one-shot HF Space + model repo uploader.
- `scripts/cleanup_obsolete.py` -- delete obsolete files from local + git + Space.

### Documentation
- `README.md` -- live-env curl block, Quickstart, OpenEnv compliance notes,
  REINFORCE + GRPO results, Rubric usage example, repository layout, submission links.
- `BLOG.md` -- separate standalone writeup for the hackathon's "mini-blog
  on Hugging Face" deliverable.

### Removed in v0.3.0-hackathon
- `notebooks/CyberSOC_Arena_GRPO.ipynb` (Colab T4 + Qwen2.5-0.5B path -- not used in submission)
- `scripts/run_hf_job.sh` (T4-medium launcher; superseded by `scripts/run_hf_job_l40s.sh`)
- `PRESENTATION.md` (slide-deck script -- subsumed by `BLOG.md`)
- `SUBMIT.md` (internal handoff doc -- not a hackathon deliverable)

### Submission surfaces
- HF Space (env): <https://huggingface.co/spaces/amit51/cybersoc-arena>
- Trained adapter: <https://huggingface.co/amit51/cybersoc-arena-qwen2.5-1.5b-grpo>
- GitHub mirror: <https://github.com/AmitChowdary122/cyber-openenv>
- Stable submission tag: <https://github.com/AmitChowdary122/cyber-openenv/tree/v0.3.0-hackathon>
