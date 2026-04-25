"""Trained policy agent.

Loads a fine-tuned causal LM (HF Transformers) and emits an action JSON per
step. If transformers/torch are unavailable, or if no model is provided, falls
back to an enhanced heuristic that uses the same prompt template — so this
agent always runs and the benchmark stays comparable.

The model is expected to have been fine-tuned with train_sft.py and/or
train_grpo.py to produce CyberSOC action JSON given the rendered observation.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from baselines.heuristic_agent import HeuristicAgent
from cybersoc_arena.actions import parse_action
from cybersoc_arena.observations import render_observation_text


_SYSTEM_PROMPT = (
    "You are a Tier-2 SOC analyst. You have a budget of investigation steps.\n"
    "Each turn you must respond with a SINGLE JSON object describing one action.\n"
    "Valid action_types: investigate_ip, query_logs, inspect_endpoint, "
    "check_threat_intel, correlate_events, isolate_host, escalate_incident, "
    "identify_attacker, close_as_benign.\n"
    "For ip-targeted actions, set 'ip'. For host-targeted, set 'host'. "
    "For correlate_events set 'entity'. For escalate_incident or close_as_benign "
    "set 'summary' to a 1-2 sentence note.\n"
    "Only output the JSON object — no prose, no markdown."
)


class TrainedPolicyAgent:
    name = "trained_policy"

    def __init__(self, model_path: Optional[str] = None, max_new_tokens: int = 96):
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._tokenizer = None
        self._fallback = HeuristicAgent()
        self._loaded = False
        if model_path:
            try:
                self._load()
            except Exception as e:
                print(f"[trained_policy] Model load failed ({e}); using heuristic fallback.")

    def _load(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        self._model.eval()
        self._loaded = True

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        if not self._loaded:
            return self._fallback.act(obs)
        try:
            return self._infer(obs)
        except Exception as e:
            print(f"[trained_policy] inference error: {e}; falling back.")
            return self._fallback.act(obs)

    def _infer(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        import torch
        prompt = self._build_prompt(obs)
        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        text = self._tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        try:
            action = parse_action(text)
            return action.to_dict()
        except ValueError:
            return self._fallback.act(obs)

    def _build_prompt(self, obs: Dict[str, Any]) -> str:
        body = render_observation_text(obs)
        return (
            f"<|system|>\n{_SYSTEM_PROMPT}\n<|end|>\n"
            f"<|user|>\n{body}\n\nRespond with a single JSON action.\n<|end|>\n"
            f"<|assistant|>\n"
        )
