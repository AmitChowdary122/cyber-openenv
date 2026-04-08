# app/env.py

from app.state import SimpleState
from app.reward import compute_reward
from app.models import Observation, Action, Reward
from typing import Tuple, Dict, Any
from app.tasks import TASKS


class CyberEnv:
    def __init__(self, difficulty: str = 'easy'):
        self.difficulty = difficulty
        self._state = SimpleState()

        # Match task with difficulty
        self.current_task = next(t for t in TASKS if t["id"] == difficulty)

        if difficulty == 'easy':
            self.max_steps = 10
        elif difficulty == 'medium':
            self.max_steps = 8
        elif difficulty == 'hard':
            self.max_steps = 5
        else:
            raise ValueError("difficulty must be 'easy', 'medium', or 'hard'")

        self.reset()

    def reset(self) -> Observation:
        self._state.reset(difficulty=self.difficulty)

        # Ensure correct task
        self.current_task = next(t for t in TASKS if t["id"] == self.difficulty)

        return self._get_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        self._state.apply_action(action.action_type, action.parameters)
        self._state.generate_step_logs(difficulty=self.difficulty)

        obs = self._get_observation()
        task_id = self.current_task["id"]

        reward = compute_reward(action.action_type, action.parameters, self._state)

        # ✅ ONLY use state logic (no overrides)
        done = self._state.is_terminal(
            max_steps=self.max_steps,
            task_id=task_id
        )

        info = {"steps": self._state.steps}

        return obs, reward, done, info

    def state(self) -> Dict[str, Any]:
        return {
            **self._state.to_dict(),
            "current_task": self.current_task,
            "suspicion_scores": self._state.suspicion_scores,
            "attackers": self._state.attackers,  # only for grader
        }

    def _get_observation(self) -> Observation:
        suspicion_scores = self._state.suspicion_scores

        top_suspicious = sorted(
            suspicion_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]

        top_suspicious_ips = [ip for ip, _ in top_suspicious]

        # ✅ FIX: NO attacker leakage
        inv_results = {}
        for ip, count in self._state.investigation_history.items():
            inv_results[ip] = {
                "investigated_count": count,
                "suspicion": suspicion_scores.get(ip, 0.0)
            }

        return Observation(
            logs=self._state.logs[-5:],
            alerts=self._state.alerts[-5:],
            system_state=self._state.system_state,
            goal=self.current_task["goal"],
            suspicion_scores=suspicion_scores,
            top_suspicious_ips=top_suspicious_ips,
            investigation_result=inv_results if inv_results else None,
        )