# Submission checklist -- CyberSOC Arena

Deadline: **5 PM IST, 26 April 2026** (changes after deadline are not considered).

## Maps every minimum requirement to a file/link

| Requirement | Where it lives in this repo | Status |
|---|---|---|
| Use OpenEnv (latest release) | `cybersoc_arena/env.py` inherits `openenv.core.env_server.Environment`; `cybersoc_arena/server.py` builds via `create_fastapi_app(...)`; `pyproject.toml` declares `openenv-core>=0.2.3` | done |
| Working training script (Unsloth or HF TRL) | `notebooks/CyberSOC_Arena_GRPO.ipynb` (TRL `GRPOTrainer` + Qwen2.5-0.5B-Instruct + LoRA, runnable on free T4) and `scripts/train_hf_job.py` (PEP-723 inline-deps for `hf jobs uv run --flavor t4-medium`) | done |
| Real loss + reward plots | `assets/reward_curve.png`, `assets/loss_curve.png`, `assets/curriculum_progress.png`, `assets/baseline_comparison.png` -- all from a real REINFORCE run committed to repo | done |
| Mini-blog OR <2 min video OR slides | `BLOG.md` (Hugging Face-style writeup) and `PRESENTATION.md` (2-minute pitch script) | done |
| Push environment to HF Space | See **Push to HF Space** section below | TODO before deadline |
| README that motivates problem, explains env, shows results | `README.md` -- all 4 plots embedded, results table, scenario list, quickstart, OpenEnv compliance notes | done |
| README links to HF Space + materials | `README.md` "Submission links" section (Space, BLOG, PRESENTATION, notebook, HF Jobs script) | done (Space URL placeholder, fill in after push) |
| No big video files in env submission | nothing >100 KB in `assets/`; no video files committed | done |
| Valid `openenv.yaml` | `openenv.yaml` lists 6 scenarios, standard endpoints, declares Pydantic action/observation/state classes | done |
| No reserved tool names | the 9 SOC tools (`investigate_ip`, `query_logs`, `inspect_endpoint`, `check_threat_intel`, `correlate_events`, `identify_attacker`, `isolate_host`, `escalate_incident`, `close_as_benign`) avoid `reset`, `step`, `state`, `close` | done |
| Client/server separation | `cybersoc_arena/client.py` does not import server internals | done |
| Gym-style API | `reset(seed=, episode_id=, ...)`, `step(action)`, `state` property | done |

## Push to HF Space (do this before 5 PM IST)

1. Create the Space (one-time):
   ```bash
   pip install -U huggingface_hub
   hf auth login                 # paste your HF token (you have write access)
   ```

2. Push the env using the OpenEnv CLI (preferred -- handles Dockerfile + entry point):
   ```bash
   cd /path/to/cyber-openenv
   openenv push --repo-id amit51/cybersoc-arena
   ```
   If `openenv push` is unavailable, fall back to:
   ```bash
   hf upload amit51/cybersoc-arena . \
        --repo-type space \
        --commit-message "CyberSOC Arena v0.3.0 - OpenEnv hackathon submission"
   ```

3. Verify the Space build is green:
   - <https://huggingface.co/spaces/amit51/cybersoc-arena>
   - `https://amit51-cybersoc-arena.hf.space/health` should return `{"status": "healthy"}`
   - `https://amit51-cybersoc-arena.hf.space/docs` should render Swagger UI
   - `https://amit51-cybersoc-arena.hf.space/web` should render the HumanAgent UI

4. Update `README.md` Submission links if the URL differs from the placeholder.

5. Commit and push the README change. **This must finish before 5 PM IST.**

## Final hand-check before submitting the URL

- [ ] `pytest -q` passes locally
- [ ] `python demo_run.py` finishes without errors
- [ ] `python demo_long_horizon.py` shows correct attribution (+2.06 reward on seed=314)
- [ ] `python train_reinforce.py --episodes 600` finishes in under 10 sec and writes all 4 plots
- [ ] `https://huggingface.co/spaces/amit51/cybersoc-arena` build is green
- [ ] `assets/reward_curve.png`, `assets/loss_curve.png`, `assets/curriculum_progress.png`, `assets/baseline_comparison.png` are all present and committed
- [ ] `BLOG.md` and `PRESENTATION.md` are present and linked from README
- [ ] `notebooks/CyberSOC_Arena_GRPO.ipynb` opens and runs the first 2 cells without error
- [ ] `scripts/train_hf_job.py` and `scripts/run_hf_job.sh` are present
- [ ] `openenv.yaml` validates (visible in HF Space settings UI)
- [ ] Submit the **HF Space URL** in the form: `https://huggingface.co/spaces/amit51/cybersoc-arena`

## Known risks + mitigations

| Risk | Mitigation |
|---|---|
| HF Space build OOMs on `pip install openenv-core` | Dockerfile pins minimal runtime deps only (no torch/trl in the Space image) |
| `openenv push` CLI unavailable | Fallback to `hf upload --repo-type space` is documented above |
| Colab notebook errors on free T4 | Notebook has small batch sizes (per_device_train_batch_size=2) and short max_completion_length=128 |
| HF Jobs run exceeds the $30 credit | `t4-medium` flavor is ~$0.60/hr; the launcher requests num_prompts=160 epochs=1 (~10 min, ~$0.10/run) |

## What the Space exposes

| Endpoint | Purpose |
|---|---|
| `POST /reset`        | start a new episode (`{"seed": 42, "scenario_type": "long_horizon_apt"}`) |
| `POST /step`         | send one `CyberAction` |
| `GET  /state`        | current `CyberState` snapshot (no ground truth leak) |
| `GET  /health`       | liveness probe |
| `GET  /docs`         | OpenAPI / Swagger UI |
| `GET  /web`          | interactive HumanAgent UI (web form for trying the env) |
| `WS   /ws`           | persistent WebSocket session for `EnvClient`-driven training |

These are all set up by `openenv.core.env_server.create_fastapi_app` -- we
do not write any custom HTTP code.
