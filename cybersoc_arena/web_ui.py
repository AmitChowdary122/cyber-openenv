"""Custom Gradio /web UI for CyberSOC Arena.

This is a hand-built ``gradio_builder`` that replaces openenv-core's
generic auto-generated form with a SOC-analyst-themed dashboard:

  * One button per scenario archetype (one click to spawn an episode).
  * Prominent alert summary with severity-coloured banner.
  * Asset inventory + step budget + running totals strip.
  * Investigative tools and terminal actions visually grouped, each with
    its own target input field (IP / host / entity / summary).
  * Live evidence log table that scrolls as the agent acts.
  * Reward breakdown panel showing which named rubric components fired.

Plumbed into ``cybersoc_arena.server._create_app`` via the
``gradio_builder=`` kwarg of
``openenv.core.env_server.web_interface.create_web_interface_app``.

If gradio is missing or this builder errors at import time,
``server.py`` falls back to the plain ``create_fastapi_app`` so the
JSON ``/reset``, ``/step``, ``/state`` surface keeps working.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


SCENARIO_LABELS = [
    ("benign_scan",         "Benign internet scan"),
    ("phishing_lateral",    "Phishing -> lateral movement"),
    ("credential_stuffing", "Credential stuffing flood"),
    ("data_exfiltration",   "Slow data exfiltration"),
    ("multi_stage_chain",   "Multi-stage kill chain"),
    ("long_horizon_apt",    "20-step APT (the marquee scenario)"),
]

SEVERITY_COLOR = {
    "critical": "#7a1f1f",
    "high":     "#a14a1a",
    "medium":   "#7a6a1a",
    "low":      "#1a4f7a",
    "info":     "#444444",
}


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print helpers
# ─────────────────────────────────────────────────────────────────────────────
def _alert_md(obs: Dict[str, Any]) -> str:
    if not obs:
        return "*No active episode. Pick a scenario above.*"
    alert = obs.get("alert", {}) or {}
    sev = (alert.get("severity") or "info").lower()
    color = SEVERITY_COLOR.get(sev, "#444")
    summary = alert.get("summary", "(no summary)")
    return (
        f"<div style='background:{color};color:white;padding:14px 18px;"
        f"border-radius:8px;font-weight:600;font-size:15px'>"
        f"<span style='font-size:11px;opacity:0.85;letter-spacing:1px'>"
        f"ALERT &middot; SEV {sev.upper()}</span><br>{summary}</div>"
    )


def _inventory_md(obs: Dict[str, Any]) -> str:
    if not obs:
        return ""
    inv = obs.get("asset_inventory", {}) or {}
    ips = inv.get("visible_ips", []) or []
    hosts = inv.get("hosts", []) or []
    step = obs.get("step", 0)
    budget = obs.get("step_budget", 0)
    rem = obs.get("remaining_steps", budget)
    ev_n = obs.get("evidence_count", 0)
    return (
        f"**Step {step}/{budget}** &nbsp; ({rem} remaining) &nbsp;|&nbsp; "
        f"**Evidence collected: {ev_n}**\n\n"
        f"**Visible IPs:** `{', '.join(ips[:8]) or '(none)'}`\n\n"
        f"**Hosts:** `{', '.join(hosts[:8]) or '(none)'}`"
    )


def _evidence_html(obs: Dict[str, Any]) -> str:
    """HTML table of evidence collected (no pandas dep)."""
    rows = (obs or {}).get("evidence_collected", []) or []
    if not rows:
        return ("<div style='color:#888;font-style:italic'>"
                "(no evidence yet -- pick a tool and target above)</div>")
    body = "".join(
        "<tr>"
        f"<td style='padding:4px 8px;color:#888'>{ev.get('step','')}</td>"
        f"<td style='padding:4px 8px'><code>{ev.get('action','')[:24]}</code></td>"
        f"<td style='padding:4px 8px'><code>{ev.get('target','')[:24]}</code></td>"
        f"<td style='padding:4px 8px'>{(ev.get('finding','') or '')[:160]}</td>"
        "</tr>"
        for ev in rows
    )
    return (
        "<table style='border-collapse:collapse;width:100%;font-size:13px'>"
        "<thead><tr style='background:#f0f0f3'>"
        "<th style='padding:6px 8px;text-align:left'>step</th>"
        "<th style='padding:6px 8px;text-align:left'>action</th>"
        "<th style='padding:6px 8px;text-align:left'>target</th>"
        "<th style='padding:6px 8px;text-align:left'>finding</th>"
        "</tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )


def _history_html(obs: Dict[str, Any]) -> str:
    rows = (obs or {}).get("action_history", []) or []
    if not rows:
        return ("<div style='color:#888;font-style:italic'>"
                "(no actions yet)</div>")
    body = "".join(
        "<tr>"
        f"<td style='padding:4px 8px'><code>{h.get('action_type','')[:24]}</code></td>"
        f"<td style='padding:4px 8px'><code>{h.get('target','')[:24]}</code></td>"
        f"<td style='padding:4px 8px;color:{'#1a7a3a' if h.get('success') else '#a14a1a'}'>"
        f"{'OK' if h.get('success') else 'FAIL'}</td>"
        "</tr>"
        for h in rows
    )
    return (
        "<table style='border-collapse:collapse;width:100%;font-size:13px'>"
        "<thead><tr style='background:#f0f0f3'>"
        "<th style='padding:6px 8px;text-align:left'>action</th>"
        "<th style='padding:6px 8px;text-align:left'>target</th>"
        "<th style='padding:6px 8px;text-align:left'>status</th>"
        "</tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )


def _reward_md(obs: Dict[str, Any], last_reward: float) -> str:
    if not obs:
        return ""
    info = obs.get("info", {}) or {}
    breakdown = info.get("breakdown", {}) or {}
    if not breakdown:
        return f"**Last step reward:** `{last_reward:+.3f}`"
    parts = ", ".join(f"`{k}: {v:+.2f}`" for k, v in breakdown.items())
    done = obs.get("done", False)
    done_md = "  &nbsp; **EPISODE DONE**" if done else ""
    return f"**Last step reward:** `{last_reward:+.3f}` &nbsp; ({parts}){done_md}"


def _initial_obs_placeholder() -> Dict[str, Any]:
    """Empty-state observation so the UI renders before first reset."""
    return {
        "alert": {"summary": "Pick a scenario above to spawn an episode.",
                  "severity": "info"},
        "asset_inventory": {"visible_ips": [], "hosts": []},
        "step": 0, "step_budget": 0, "remaining_steps": 0,
        "evidence_count": 0, "evidence_collected": [], "action_history": [],
        "info": {}, "done": False,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Custom Gradio builder
# ─────────────────────────────────────────────────────────────────────────────
def build_cybersoc_gradio_ui(
    web_manager: Any,
    action_fields: List[Dict[str, Any]],
    metadata: Optional[Any],
    is_chat_env: bool,
    title: str = "CyberSOC Arena",
    quick_start_md: Optional[str] = None,
):
    """Build the SOC-analyst dashboard. Returns a ``gradio.Blocks`` instance."""
    import gradio as gr

    css = """
    .soc-card { padding: 10px 14px; border-radius: 8px; background: #f7f7f8;
                border: 1px solid #e1e1e3; }
    .gradio-container { max-width: 1280px !important; }
    """

    async def _do_reset(scenario: str, seed_str: str):
        try:
            seed = int(seed_str) if (seed_str and seed_str.strip()) else None
        except ValueError:
            seed = None
        try:
            data = await web_manager.reset_environment(
                {"scenario_type": scenario, "seed": seed}
                if seed is not None
                else {"scenario_type": scenario}
            )
            obs = data.get("observation", {}) or _initial_obs_placeholder()
        except Exception as exc:
            obs = _initial_obs_placeholder()
            obs["alert"] = {"summary": f"Reset failed: {exc.__class__.__name__}: {exc}",
                            "severity": "critical"}
        last_reward = obs.get("reward", 0.0) or 0.0
        return (
            _alert_md(obs),
            _inventory_md(obs),
            _evidence_html(obs),
            _history_html(obs),
            _reward_md(obs, last_reward),
            f"Reset OK. Scenario: {scenario}",
        )

    async def _do_step(action_type: str, ip: str, host: str,
                       entity: str, summary: str):
        action: Dict[str, Any] = {"action_type": action_type}
        if ip and ip.strip():       action["ip"] = ip.strip()
        if host and host.strip():   action["host"] = host.strip()
        if entity and entity.strip(): action["entity"] = entity.strip()
        if summary and summary.strip(): action["summary"] = summary.strip()
        try:
            data = await web_manager.step_environment(action)
            obs = data.get("observation", {}) or _initial_obs_placeholder()
            status = f"Step OK ({action_type})."
        except Exception as exc:
            obs = _initial_obs_placeholder()
            obs["alert"] = {"summary": f"Step failed: {exc.__class__.__name__}: {exc}",
                            "severity": "critical"}
            status = f"Step failed: {exc.__class__.__name__}"
        last_reward = obs.get("reward", 0.0) or 0.0
        return (
            _alert_md(obs),
            _inventory_md(obs),
            _evidence_html(obs),
            _history_html(obs),
            _reward_md(obs, last_reward),
            status,
        )

    with gr.Blocks(title=title, css=css, theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# CyberSOC Arena &mdash; Tier-2 SOC Analyst Sandbox\n"
            "*An OpenEnv environment where an LLM (or you) acts as a SOC analyst: "
            "triage the alert, pick the right tool on the right target, gather "
            "cross-host evidence, and commit a verdict before the step budget runs out. "
            "[Read the BLOG](https://huggingface.co/spaces/amit51/cybersoc-arena/blob/main/BLOG.md)*"
        )

        # ── Top: Scenario picker ───────────────────────────────────────────
        gr.Markdown("### 1. Pick a scenario")
        with gr.Row():
            scenario_dd = gr.Dropdown(
                choices=[k for k, _ in SCENARIO_LABELS],
                value="long_horizon_apt",
                label="Scenario archetype",
                scale=3,
            )
            seed_box = gr.Textbox(
                value="314", label="Seed (blank = random)", scale=1,
            )
            reset_btn = gr.Button("Reset episode", variant="primary", scale=1)

        with gr.Row():
            for key, lbl in SCENARIO_LABELS:
                btn = gr.Button(f"-> {lbl}", scale=1, size="sm")

                def _make_quick_reset(scenario_key):
                    async def _quick(seed):
                        return await _do_reset(scenario_key, seed)
                    return _quick

                # quick-fire scenario button: fixed scenario, current seed box
                # populated. Replace the dropdown so the user sees what was picked.
                btn.click(fn=_make_quick_reset(key), inputs=[seed_box],
                          outputs=None, queue=False).then(
                    fn=lambda k=key: k, inputs=None, outputs=scenario_dd, queue=False)

        # ── Middle: Alert + inventory + reward ─────────────────────────────
        gr.Markdown("### 2. The current alert")
        alert_html = gr.HTML(_alert_md(_initial_obs_placeholder()))

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("**Asset inventory & step counter**")
                inventory_md = gr.Markdown(_inventory_md(_initial_obs_placeholder()))
            with gr.Column(scale=1):
                gr.Markdown("**Reward (last step)**")
                reward_md = gr.Markdown("")

        # ── Action panel ──────────────────────────────────────────────────
        gr.Markdown("### 3. Pick a SOC tool")
        with gr.Row():
            with gr.Column():
                gr.Markdown("**Investigative (5 tools)**")
                ip_in = gr.Textbox(label="Target IP",
                                   placeholder="e.g. 198.51.100.7")
                with gr.Row():
                    inv_btn      = gr.Button("investigate_ip")
                    log_btn      = gr.Button("query_logs")
                    intel_btn    = gr.Button("check_threat_intel")
                host_in = gr.Textbox(label="Target host",
                                     placeholder="e.g. ws-laptop-04")
                inspect_btn = gr.Button("inspect_endpoint")
                entity_in = gr.Textbox(label="Entity to correlate (IP or host)",
                                       placeholder="e.g. 198.51.100.7")
                corr_btn = gr.Button("correlate_events")
            with gr.Column():
                gr.Markdown("**Terminal (4 tools, end the episode)**")
                ident_btn = gr.Button("identify_attacker (uses Target IP)",
                                      variant="primary")
                iso_btn = gr.Button("isolate_host (uses Target host)",
                                    variant="primary")
                summary_in = gr.Textbox(
                    label="Analyst summary (for escalate / close_as_benign)",
                    placeholder="e.g. Internet scanner; no internal pivot. Closing.",
                    lines=2,
                )
                with gr.Row():
                    esc_btn = gr.Button("escalate_incident", variant="primary")
                    close_btn = gr.Button("close_as_benign", variant="primary")

        status_box = gr.Textbox(label="Status", interactive=False, value="")

        # ── Logs (HTML tables to avoid pandas/jinja2 dep on gr.Dataframe) ──
        gr.Markdown("### 4. What you've gathered so far")
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("**Evidence collected**")
                evidence_html = gr.HTML(_evidence_html(_initial_obs_placeholder()))
            with gr.Column(scale=2):
                gr.Markdown("**Action history**")
                history_html = gr.HTML(_history_html(_initial_obs_placeholder()))

        outputs = [alert_html, inventory_md, evidence_html,
                   history_html, reward_md, status_box]

        # ── Wire reset ────────────────────────────────────────────────────
        reset_btn.click(fn=_do_reset, inputs=[scenario_dd, seed_box],
                        outputs=outputs)

        # ── Wire each tool button ─────────────────────────────────────────
        # Each button passes its action_type as a constant, plus the relevant
        # target field(s). We pass *all* target inputs to keep the function
        # signature uniform; _do_step ignores empties.
        target_inputs = [ip_in, host_in, entity_in, summary_in]

        def _bind(btn, action_type: str):
            btn.click(
                fn=lambda a, b, c, d, _at=action_type: _do_step(_at, a, b, c, d),
                inputs=target_inputs,
                outputs=outputs,
            )

        _bind(inv_btn,    "investigate_ip")
        _bind(log_btn,    "query_logs")
        _bind(intel_btn,  "check_threat_intel")
        _bind(inspect_btn,"inspect_endpoint")
        _bind(corr_btn,   "correlate_events")
        _bind(ident_btn,  "identify_attacker")
        _bind(iso_btn,    "isolate_host")
        _bind(esc_btn,    "escalate_incident")
        _bind(close_btn,  "close_as_benign")

        gr.Markdown(
            "---\n"
            "*The 9 tools, 6 scenarios, and 17 named reward components are "
            "described in the [README](https://huggingface.co/spaces/amit51/cybersoc-arena/blob/main/README.md) "
            "and [BLOG](https://huggingface.co/spaces/amit51/cybersoc-arena/blob/main/BLOG.md). "
            "The trained Qwen2.5-1.5B-Instruct adapter "
            "([model card](https://huggingface.co/amit51/cybersoc-arena-qwen2.5-1.5b-grpo)) "
            "drives this same env over the WebSocket session at /ws.*"
        )

    return demo
