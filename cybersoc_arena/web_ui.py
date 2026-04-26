"""Custom Gradio /web UI for CyberSOC Arena -- wizard-style layout.

Designed for first-time visitors. Hard-coded UX flow:

  STEP 1 -- pick a scenario (6 buttons, each one resets the env)
  STEP 2 -- read the alert + asset inventory + step counter
  STEP 3 -- pick a tool from a dropdown; the SINGLE target field auto-labels
            and shows a description of what the tool does; click "Take Step"
  STEP 4 -- watch evidence + action history fill in, see reward breakdown

The single-target-field design hides the IP / host / entity / summary
mapping from the user. Behind the scenes, the action_type determines
which Pydantic field the target value gets routed to.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


SCENARIOS: List[Tuple[str, str, str]] = [
    ("benign_scan",
     "Benign internet scan",
     "Internet scanner pinging your edge. The right call is to recognise it as noise and NOT isolate."),
    ("phishing_lateral",
     "Phishing -> lateral movement",
     "Phishing email -> credential reuse -> lateral movement. 8 steps, 3 hosts."),
    ("credential_stuffing",
     "Credential stuffing flood",
     "Pick the real attacker IP out of a flood of failed logins. 8 steps, 2 hosts."),
    ("data_exfiltration",
     "Slow data exfiltration",
     "Slow, low-volume covert TLS egress. 10 steps, 3 hosts."),
    ("multi_stage_chain",
     "Multi-stage kill chain",
     "Recon -> exploit -> persist -> exfil short kill chain. 12 steps, 4 hosts."),
    ("long_horizon_apt",
     "20-step APT (the marquee scenario)",
     "Full multi-phase APT across 5 hosts with 3 carefully tuned decoys. 20 steps."),
]

# (key, friendly_label, target_field, target_label, target_placeholder, description, is_terminal)
TOOLS: List[Tuple[str, str, str, str, str, str, bool]] = [
    ("investigate_ip", "investigate_ip",
     "ip", "Target IP", "e.g. 198.51.100.7",
     "Broad cross-source pull on an IP (edge logs + threat intel + flow data). Best as a first move when you don't know what you're looking at yet.",
     False),
    ("query_logs", "query_logs",
     "ip", "Target IP", "e.g. 198.51.100.7",
     "Targeted log pull for an IP (edge / DNS / SMB / proxy). Cheaper than investigate_ip if you already know which log source matters.",
     False),
    ("inspect_endpoint", "inspect_endpoint",
     "host", "Target host", "e.g. ws-laptop-04",
     "EDR + persistence + process tree on a specific host. Use after you've narrowed down which workstation or server is suspect.",
     False),
    ("check_threat_intel", "check_threat_intel",
     "ip", "Target IP", "e.g. 198.51.100.7",
     "External attribution data for an IP (known scanner? authorised red team? real attacker infrastructure?). Cheap and fast.",
     False),
    ("correlate_events", "correlate_events",
     "entity", "Entity to correlate (IP or host)", "e.g. 198.51.100.7",
     "Link an entity (IP OR host) to other indicators. Lands the +0.20 correlation bonus when it matches a real attacker pair.",
     False),
    ("identify_attacker", "identify_attacker  [TERMINAL]",
     "ip", "Attacker IP", "e.g. 198.51.100.7",
     "TERMINAL: attribute the attack to a specific IP. +1.50 if right, -1.50 if wrong. Don't fire until you have >=3 attacker-confirming evidence pieces.",
     True),
    ("isolate_host", "isolate_host  [TERMINAL]",
     "host", "Host to isolate", "e.g. ws-laptop-04",
     "TERMINAL: contain a host (cuts its network). +0.80 if it is genuinely compromised, -1.00 if it is clean.",
     True),
    ("escalate_incident", "escalate_incident  [TERMINAL]",
     "summary", "Analyst summary (handoff text)", "e.g. Confirmed lateral movement; recommend tier-3 IR.",
     "TERMINAL: hand off to L3 with a 1-2 sentence summary. Reward scales with keyword overlap against the reference summary.",
     True),
    ("close_as_benign", "close_as_benign  [TERMINAL]",
     "summary", "Why this is benign", "e.g. Authorised internet scanner. No internal pivot observed.",
     "TERMINAL: declare no incident. +1.20 if scenario IS benign, -1.50 if you closed a real incident as benign (the worst single SOC mistake).",
     True),
]

TOOL_BY_KEY = {t[0]: t for t in TOOLS}
TOOL_KEYS = [t[0] for t in TOOLS]
TOOL_LABELS = [t[1] for t in TOOLS]

SEVERITY_COLOR = {
    "critical": "#7a1f1f", "high": "#a14a1a", "medium": "#7a6a1a",
    "low": "#1a4f7a", "info": "#3a3a3a",
}


def _alert_html(obs: Dict[str, Any]) -> str:
    if not obs or not obs.get("alert"):
        return ("<div style='background:#3a3a3a;color:white;padding:14px 18px;"
                "border-radius:8px;font-weight:600;font-size:14px'>"
                "<span style='font-size:11px;opacity:0.85;letter-spacing:1px'>"
                "WAITING FOR EPISODE</span><br>"
                "Pick a scenario above to spawn an episode.</div>")
    alert = obs["alert"]
    sev = (alert.get("severity") or "info").lower()
    color = SEVERITY_COLOR.get(sev, "#3a3a3a")
    return (f"<div style='background:{color};color:white;padding:14px 18px;"
            f"border-radius:8px;font-weight:600;font-size:15px;line-height:1.4'>"
            f"<span style='font-size:11px;opacity:0.85;letter-spacing:1px'>"
            f"ALERT &middot; SEV {sev.upper()}</span><br>"
            f"{alert.get('summary', '(no summary)')}</div>")


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
    done_badge = " &nbsp; <b style='color:#7a1f1f'>EPISODE OVER</b>" if obs.get("done") else ""
    return (
        f"**Step {step} of {budget}** ({rem} remaining) "
        f"&middot; **Evidence: {ev_n}**{done_badge}\n\n"
        f"**Visible IPs** &nbsp; `{', '.join(ips[:8]) or '(none yet)'}`\n\n"
        f"**Hosts** &nbsp; `{', '.join(hosts[:8]) or '(none yet)'}`"
    )


def _evidence_html(obs: Dict[str, Any]) -> str:
    rows = (obs or {}).get("evidence_collected", []) or []
    if not rows:
        return ("<div style='color:#888;font-style:italic;padding:12px'>"
                "(no evidence yet -- pick a tool below and click Take Step)</div>")
    body = "".join(
        "<tr style='border-bottom:1px solid #ececef'>"
        f"<td style='padding:6px 8px;color:#888;width:40px'>{ev.get('step','')}</td>"
        f"<td style='padding:6px 8px;width:140px'><code>{ev.get('action','')[:24]}</code></td>"
        f"<td style='padding:6px 8px;width:140px'><code>{ev.get('target','')[:24]}</code></td>"
        f"<td style='padding:6px 8px'>{(ev.get('finding','') or '')[:200]}</td></tr>"
        for ev in rows
    )
    return ("<table style='border-collapse:collapse;width:100%;font-size:13px'>"
            "<thead><tr style='background:#f0f0f3'>"
            "<th style='padding:8px;text-align:left'>step</th>"
            "<th style='padding:8px;text-align:left'>tool</th>"
            "<th style='padding:8px;text-align:left'>target</th>"
            "<th style='padding:8px;text-align:left'>finding</th></tr></thead>"
            f"<tbody>{body}</tbody></table>")


def _history_html(obs: Dict[str, Any]) -> str:
    rows = (obs or {}).get("action_history", []) or []
    if not rows:
        return ("<div style='color:#888;font-style:italic;padding:12px'>"
                "(no actions yet)</div>")
    body = "".join(
        "<tr style='border-bottom:1px solid #ececef'>"
        f"<td style='padding:6px 8px'><code>{h.get('action_type','')[:24]}</code></td>"
        f"<td style='padding:6px 8px'><code>{h.get('target','')[:24]}</code></td>"
        f"<td style='padding:6px 8px;color:{'#1a7a3a' if h.get('success') else '#a14a1a'};font-weight:bold'>"
        f"{'OK' if h.get('success') else 'FAIL'}</td></tr>"
        for h in rows
    )
    return ("<table style='border-collapse:collapse;width:100%;font-size:13px'>"
            "<thead><tr style='background:#f0f0f3'>"
            "<th style='padding:8px;text-align:left'>tool</th>"
            "<th style='padding:8px;text-align:left'>target</th>"
            "<th style='padding:8px;text-align:left'>status</th></tr></thead>"
            f"<tbody>{body}</tbody></table>")


def _reward_md(obs: Dict[str, Any], last_reward: float) -> str:
    if not obs:
        return "*Take a step and the reward breakdown will appear here.*"
    info = obs.get("info", {}) or {}
    breakdown = info.get("breakdown", {}) or {}
    if not breakdown:
        return f"**Last step reward:** `{last_reward:+.3f}`"
    parts = ", ".join(f"`{k}: {v:+.2f}`" for k, v in breakdown.items())
    return f"**Last step reward:** `{last_reward:+.3f}` &nbsp; ({parts})"


def _initial_obs() -> Dict[str, Any]:
    return {
        "alert": None,
        "asset_inventory": {"visible_ips": [], "hosts": []},
        "step": 0, "step_budget": 0, "remaining_steps": 0,
        "evidence_count": 0, "evidence_collected": [], "action_history": [],
        "info": {}, "done": False, "reward": 0.0,
    }


def _tool_help_md(tool_key: str) -> str:
    if tool_key not in TOOL_BY_KEY:
        return ""
    _, label, _, _, _, desc, is_terminal = TOOL_BY_KEY[tool_key]
    badge = ("<span style='background:#a14a1a;color:white;padding:2px 8px;"
             "border-radius:4px;font-size:11px;letter-spacing:1px'>TERMINAL</span> "
             if is_terminal else "")
    return f"{badge}**{label}** &mdash; {desc}"


def build_cybersoc_gradio_ui(
    web_manager: Any,
    action_fields: List[Dict[str, Any]],
    metadata: Optional[Any],
    is_chat_env: bool,
    title: str = "CyberSOC Arena",
    quick_start_md: Optional[str] = None,
):
    import gradio as gr

    async def _do_reset(scenario: str, seed_str: str):
        try:
            seed = int(seed_str) if (seed_str and seed_str.strip()) else None
        except ValueError:
            seed = None
        kwargs: Dict[str, Any] = {"scenario_type": scenario}
        if seed is not None:
            kwargs["seed"] = seed
        try:
            data = await web_manager.reset_environment(kwargs)
            obs = data.get("observation", {}) or _initial_obs()
            # OpenEnv puts reward + done at the TOP level of the response,
            # not inside observation. Inject them so helpers find them.
            obs["reward"] = data.get("reward", 0.0)
            obs["done"] = data.get("done", False)
            status = (f"Reset OK -- scenario `{scenario}`. Now read the alert, "
                      f"then pick a tool below.")
        except Exception as exc:
            obs = _initial_obs()
            obs["alert"] = {"summary": f"Reset failed: {exc.__class__.__name__}: {exc}",
                            "severity": "critical"}
            status = f"Reset FAILED: {exc.__class__.__name__}"
        last_reward = float(obs.get("reward", 0.0) or 0.0)
        return (_alert_html(obs), _inventory_md(obs),
                _evidence_html(obs), _history_html(obs),
                _reward_md(obs, last_reward), status)

    async def _do_step(tool_key: str, target_value: str):
        if tool_key not in TOOL_BY_KEY:
            obs = _initial_obs()
            return (_alert_html(obs), _inventory_md(obs),
                    _evidence_html(obs), _history_html(obs),
                    _reward_md(obs, 0.0), f"Unknown tool: {tool_key}")
        _, _, target_field, _, _, _, _ = TOOL_BY_KEY[tool_key]
        action: Dict[str, Any] = {"action_type": tool_key}
        if target_value and target_value.strip():
            action[target_field] = target_value.strip()
        try:
            data = await web_manager.step_environment(action)
            obs = data.get("observation", {}) or _initial_obs()
            # OpenEnv puts reward + done at the top level of the response.
            obs["reward"] = data.get("reward", 0.0)
            obs["done"] = data.get("done", False)
            last_reward = float(obs["reward"])
            done = bool(obs["done"])
            status = (
                f"Step OK -- {tool_key}({target_field}={target_value or '(empty)'}) "
                f"-> reward {last_reward:+.3f}"
                + ("  |  EPISODE OVER" if done else "")
            )
        except Exception as exc:
            obs = _initial_obs()
            obs["alert"] = {"summary": f"Step failed: {exc.__class__.__name__}: {exc}",
                            "severity": "critical"}
            last_reward = 0.0
            status = f"Step FAILED: {exc.__class__.__name__}: {exc}"
        return (_alert_html(obs), _inventory_md(obs),
                _evidence_html(obs), _history_html(obs),
                _reward_md(obs, last_reward), status)

    def _on_tool_change(tool_key: str):
        if tool_key not in TOOL_BY_KEY:
            return _tool_help_md(tool_key), gr.update(label="Target", placeholder="")
        _, _, _, target_label, target_placeholder, _, _ = TOOL_BY_KEY[tool_key]
        return (_tool_help_md(tool_key),
                gr.update(label=target_label, placeholder=target_placeholder))

    with gr.Blocks(title=title) as demo:
        gr.Markdown(
            "# CyberSOC Arena - Tier-2 SOC Analyst Sandbox\n"
            "*An OpenEnv environment where an LLM (or you) acts as a SOC analyst: "
            "triage the alert, pick the right tool on the right target, gather "
            "cross-host evidence, and commit a verdict before the step budget runs out.*\n\n"
            "**How to use this page:** click a scenario in **Step 1** to spawn an "
            "episode. Read the alert in **Step 2**. Pick a tool from the dropdown "
            "in **Step 3** -- the target field auto-relabels for whichever tool "
            "you choose. Watch evidence accumulate in **Step 4**. The episode "
            "ends when you commit a terminal action or run out of steps."
        )

        gr.Markdown("## Step 1 - Pick a scenario")
        seed_box = gr.Textbox(
            value="314",
            label="Seed (314 is the canonical APT example; blank = random)",
        )

        scenario_buttons: List[Any] = []
        for key, lbl, blurb in SCENARIOS:
            with gr.Row():
                with gr.Column(scale=2):
                    btn = gr.Button(f"-> {lbl}", variant="primary", size="lg")
                with gr.Column(scale=4):
                    gr.Markdown(blurb)
                scenario_buttons.append((btn, key))

        gr.Markdown("## Step 2 - Read the alert")
        alert_html = gr.HTML(_alert_html(_initial_obs()))
        inventory_md = gr.Markdown(_inventory_md(_initial_obs()))

        gr.Markdown("## Step 3 - Pick a tool, fill in the target, click Take Step")
        tool_dd = gr.Dropdown(
            choices=list(zip(TOOL_LABELS, TOOL_KEYS)),
            value="investigate_ip",
            label="Tool",
        )
        tool_help_md = gr.Markdown(_tool_help_md("investigate_ip"))
        target_in = gr.Textbox(
            label=TOOL_BY_KEY["investigate_ip"][3],
            placeholder=TOOL_BY_KEY["investigate_ip"][4],
            value="",
        )
        with gr.Row():
            step_btn = gr.Button("Take Step", variant="primary", size="lg", scale=4)
            clear_btn = gr.Button("Clear target", scale=1)

        status_box = gr.Textbox(label="Status", interactive=False, value="", lines=2)
        reward_md = gr.Markdown("*Take a step and the reward breakdown will appear here.*")

        gr.Markdown("## Step 4 - What you've gathered so far")
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("### Evidence collected")
                evidence_html = gr.HTML(_evidence_html(_initial_obs()))
            with gr.Column(scale=2):
                gr.Markdown("### Action history")
                history_html = gr.HTML(_history_html(_initial_obs()))

        outputs = [alert_html, inventory_md, evidence_html,
                   history_html, reward_md, status_box]

        for btn, scenario_key in scenario_buttons:
            def _make(scn_key):
                async def _go(seed):
                    return await _do_reset(scn_key, seed)
                return _go
            btn.click(fn=_make(scenario_key), inputs=[seed_box], outputs=outputs)

        tool_dd.change(fn=_on_tool_change, inputs=[tool_dd],
                       outputs=[tool_help_md, target_in])

        step_btn.click(fn=_do_step, inputs=[tool_dd, target_in], outputs=outputs)
        clear_btn.click(fn=lambda: "", inputs=None, outputs=target_in)

        gr.Markdown(
            "---\n"
            "*9 tools, 6 scenarios, 17 named reward components. Read the "
            "[README](https://huggingface.co/spaces/amit51/cybersoc-arena/blob/main/README.md) "
            "and the [BLOG](https://huggingface.co/spaces/amit51/cybersoc-arena/blob/main/BLOG.md). "
            "The Qwen2.5-1.5B-Instruct LoRA adapter trained against this env is at "
            "[amit51/cybersoc-arena-qwen2.5-1.5b-grpo](https://huggingface.co/amit51/cybersoc-arena-qwen2.5-1.5b-grpo).*"
        )

    return demo
