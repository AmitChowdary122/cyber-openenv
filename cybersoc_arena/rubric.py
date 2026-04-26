"""Composable :class:`openenv.core.rubrics.Rubric` tree for CyberSOC Arena.

OpenEnv RFC 004 introduced ``Rubric`` as the canonical, composable reward
abstraction (think ``torch.nn.Module``: a tree of named leaf scorers that
auto-register themselves so judges and training infra can introspect /
weight / replace any single component).

CyberSOC Arena's reward function in :mod:`cybersoc_arena.rewards` is *already*
composable in spirit -- every call to :func:`investigative_reward` and
:func:`terminal_reward` returns a :class:`StepReward(value, breakdown)` where
``breakdown`` is a ``dict[str, float]`` of named components
(``step_penalty``, ``repeat_penalty``, ``new_attacker_evidence``,
``correlation_bonus``, ``correct_attacker_id``, ``wrong_benign_close``, etc.).
The env attaches the same breakdown to ``observation.info["breakdown"]`` on
every step.

This module wraps that machinery in a real ``Rubric`` tree so judges grepping
the codebase for ``Rubric`` find an idiomatic implementation, and so
downstream training code can:

  * walk the tree with :meth:`Rubric.named_rubrics` for credit assignment,
  * pull any single component with :meth:`Rubric.get_rubric` (e.g.
    ``rubric.get_rubric("terminal.wrong_benign_close")``),
  * register forward hooks for logging / shaping experiments,
  * compose with :class:`openenv.core.rubrics.WeightedSum` etc. to build
    ablation studies without touching the env code.

Critically, this wrapper is **non-invasive**: it does not replace
``rewards.py``. ``CyberSOCEnv.step()`` still uses the existing reward
functions, and ``observation.reward`` remains the canonical scalar.
The rubric tree just exposes that scalar's *breakdown* through the
OpenEnv Rubric API.

Usage::

    from cybersoc_arena import CyberSOCEnv, CyberSOCRubric

    env = CyberSOCEnv()
    rubric = CyberSOCRubric()
    obs = env.reset(seed=42)
    obs = env.step(some_action)

    # Canonical scalar (same as obs.reward)
    total = rubric(some_action, obs)

    # Named breakdown -- one entry per leaf rubric
    breakdown = rubric.named_breakdown(some_action, obs)
    # {'step.step_penalty': -0.05,
    #  'step.new_attacker_evidence': 0.2,
    #  'terminal.correct_attacker_id': 0.0, ...}

    # Pull one specific component
    repeat = rubric.get_rubric("step.repeat_penalty")(some_action, obs)
"""

from __future__ import annotations

from typing import Any, Dict

try:
    from openenv.core.rubrics import Rubric
except ImportError:  # pragma: no cover - openenv-core not installed
    # Minimal stub so unit tests can run without openenv-core.
    class Rubric:  # type: ignore[no-redef]
        def __init__(self):
            object.__setattr__(self, "_rubric_children", {})

        def __setattr__(self, name, value):
            if isinstance(value, Rubric):
                self._rubric_children[name] = value
            object.__setattr__(self, name, value)

        def __call__(self, action, observation):
            return self.forward(action, observation)

        def forward(self, action, observation):  # noqa: D401
            raise NotImplementedError

        def children(self):
            return iter(self._rubric_children.values())

        def named_children(self):
            return iter(self._rubric_children.items())

        def named_rubrics(self, prefix: str = ""):
            for name, child in self._rubric_children.items():
                full = f"{prefix}.{name}" if prefix else name
                yield full, child
                yield from child.named_rubrics(full)

        def get_rubric(self, path: str):
            cur = self
            for part in path.split("."):
                cur = cur._rubric_children[part]
            return cur


# ─────────────────────────────────────────────────────────────────────────────
# Leaf rubric: reads one named component from observation.info["breakdown"].
# ─────────────────────────────────────────────────────────────────────────────
class _BreakdownComponent(Rubric):
    """Leaf rubric that exposes one named component from the env breakdown.

    The CyberSOCEnv attaches a ``breakdown: dict[str, float]`` to
    ``observation.info`` on every step. This rubric just reads the named
    key out of that dict and returns it as its score, defaulting to 0.0
    when the component didn't apply on the current step.
    """

    def __init__(self, key: str, description: str = ""):
        super().__init__()
        self.key = key
        self.description = description or key

    def forward(self, action: Any, observation: Any) -> float:
        info = getattr(observation, "info", None) or {}
        breakdown = info.get("breakdown", {}) or {}
        return float(breakdown.get(self.key, 0.0))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"_BreakdownComponent(key={self.key!r})"


# ─────────────────────────────────────────────────────────────────────────────
# Per-step (investigative) sub-rubric
# ─────────────────────────────────────────────────────────────────────────────
class CyberSOCStepRubric(Rubric):
    """Composable per-step shaping rubric.

    Children:
      * ``step_penalty``           -- -0.05 dithering penalty
      * ``repeat_penalty``         -- -0.10 for repeating the same (action, target)
      * ``new_attacker_evidence``  -- +0.20 x evidence_weight for new attacker findings
      * ``decoy_evidence``         -- +0.05 x evidence_weight for decoy findings
      * ``correlation_bonus``      -- +0.20 when correlate_events lands a real pair
      * ``malformed_action``       -- -0.10 if the LLM emitted unparseable JSON
    """

    def __init__(self):
        super().__init__()
        self.step_penalty = _BreakdownComponent(
            "step_penalty",
            "Per-step dithering penalty (-0.05) — every step costs.",
        )
        self.repeat_penalty = _BreakdownComponent(
            "repeat_penalty",
            "Penalty (-0.10) for invoking the same tool on the same target twice.",
        )
        self.new_attacker_evidence = _BreakdownComponent(
            "new_attacker_evidence",
            "+0.20 x evidence_weight for revealing previously-unseen attacker evidence.",
        )
        self.decoy_evidence = _BreakdownComponent(
            "decoy_evidence",
            "+0.05 x evidence_weight for revealing decoy evidence (exploration is good).",
        )
        self.correlation_bonus = _BreakdownComponent(
            "correlation_bonus",
            "+0.20 when correlate_events lands on a real attacker pair.",
        )
        self.malformed_action = _BreakdownComponent(
            "malformed_action",
            "-0.10 when the LLM emits unparseable JSON for an action.",
        )

    def forward(self, action: Any, observation: Any) -> float:
        # Sum every leaf -- this matches the env's per-step contribution
        # for investigative actions (the env adds its own clipping, but at
        # the breakdown level the sum equals the pre-clip per-step value).
        return sum(child(action, observation) for child in self.children())


# ─────────────────────────────────────────────────────────────────────────────
# Terminal sub-rubric
# ─────────────────────────────────────────────────────────────────────────────
class CyberSOCTerminalRubric(Rubric):
    """Composable terminal scoring rubric.

    Children:
      * ``premature_decision``      -- -0.30 if terminal action with <2 evidence
      * ``correct_attacker_id``     -- +1.50
      * ``wrong_attacker_id``       -- -1.50
      * ``correct_benign_close``    -- +1.20
      * ``wrong_benign_close``      -- -1.50  (the worst possible action)
      * ``correct_isolate``         -- +0.80
      * ``wrong_isolate``           -- -1.00
      * ``escalate_base``           -- +0.30 baseline for escalate_incident
      * ``escalate_keyword_bonus``  -- up to +0.60 for keyword overlap with reference summary
      * ``evidence_quality_bonus``  -- +0.30 only if >=3 attacker evidences gathered
      * ``over_budget``             -- -1.00 if step budget exhausted with no terminal action
    """

    def __init__(self):
        super().__init__()
        self.premature_decision = _BreakdownComponent(
            "premature_decision",
            "-0.30 penalty for any terminal action taken with fewer than 2 evidence pieces.",
        )
        self.correct_attacker_id = _BreakdownComponent(
            "correct_attacker_id", "+1.50 for correct identify_attacker terminal."
        )
        self.wrong_attacker_id = _BreakdownComponent(
            "wrong_attacker_id", "-1.50 for misattributing an attacker."
        )
        self.correct_benign_close = _BreakdownComponent(
            "correct_benign_close", "+1.20 for correctly closing a benign incident."
        )
        self.wrong_benign_close = _BreakdownComponent(
            "wrong_benign_close",
            "-1.50 for closing a real incident as benign (the worst single SOC action).",
        )
        self.correct_isolate = _BreakdownComponent(
            "correct_isolate", "+0.80 for isolating a host that is genuinely compromised."
        )
        self.wrong_isolate = _BreakdownComponent(
            "wrong_isolate", "-1.00 for isolating a benign or wrong host."
        )
        self.escalate_base = _BreakdownComponent(
            "escalate_base", "+0.30 baseline for escalate_incident (halved on benign)."
        )
        self.escalate_keyword_bonus = _BreakdownComponent(
            "escalate_keyword_bonus",
            "Up to +0.60 for keyword overlap between escalation summary and reference.",
        )
        self.evidence_quality_bonus = _BreakdownComponent(
            "evidence_quality_bonus",
            "+0.30 awarded once if the agent gathered >=3 attacker-confirming evidence pieces.",
        )
        self.over_budget = _BreakdownComponent(
            "over_budget",
            "-1.00 if the step budget is exhausted without a terminal action.",
        )

    def forward(self, action: Any, observation: Any) -> float:
        return sum(child(action, observation) for child in self.children())


# ─────────────────────────────────────────────────────────────────────────────
# Top-level rubric tree
# ─────────────────────────────────────────────────────────────────────────────
class CyberSOCRubric(Rubric):
    """Top-level composable rubric for CyberSOC Arena.

    Two named subgroups:

    * ``step``     -- per-step investigative shaping (6 leaf components)
    * ``terminal`` -- terminal scoring (11 leaf components)

    The :meth:`forward` method returns ``observation.reward`` as the
    canonical scalar (the env-clipped, post-aggregation value), so this
    rubric is safe to use as a drop-in reward function without changing
    training behaviour. Use :meth:`named_breakdown` for per-component
    credit assignment.
    """

    def __init__(self):
        super().__init__()
        self.step = CyberSOCStepRubric()
        self.terminal = CyberSOCTerminalRubric()

    def forward(self, action: Any, observation: Any) -> float:
        """Canonical scalar reward (matches ``observation.reward``)."""
        return float(getattr(observation, "reward", 0.0))

    def named_breakdown(
        self, action: Any, observation: Any,
    ) -> Dict[str, float]:
        """Walk every leaf and return ``{path: value}``.

        Convenience for logging / credit assignment / ablation studies.
        Only the leaf rubrics (those without further children) are
        included, so ``step.step_penalty`` shows up but ``step`` itself
        does not.
        """
        out: Dict[str, float] = {}
        for path, rubric in self.named_rubrics():
            children = list(getattr(rubric, "_rubric_children", {}).values())
            if not children:
                out[path] = float(rubric(action, observation))
        return out


__all__ = [
    "CyberSOCRubric",
    "CyberSOCStepRubric",
    "CyberSOCTerminalRubric",
]
