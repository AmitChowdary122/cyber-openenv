# app/env.py
"""
CyberEnv – main environment class.
Handles reset, step, observation generation, and state access.
Difficulty levels: easy (10 steps), medium (8 steps), hard (5 steps).
"""
from app.state import SimpleState
from app.reward import compute_reward
from app.models import Observation, Action, Reward
from typing import Tuple, Dict, Any, List
from app.tasks import TASKS
import random

class CyberEnv:
    """
    Cybersecurity incident response environment.
    - observation: logs, alerts, system state, suspicion scores, top suspicious IPs.
    - action: analyze_log, identify_attacker, block_ip, quarantine_system, investigate_ip.
    - reward: shaped to encourage reasoning (see reward.py).
    """

    def __init__(self, difficulty: str = 'easy'):
        """
        Args:
            difficulty: 'easy' (10 steps), 'medium' (8 steps), 'hard' (5 steps)
        """
        self.difficulty = difficulty
        self._state = SimpleState()
        self.current_task = None

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
        """Reset environment, select a random task (force hard if difficulty==hard)."""
        self._state.reset(difficulty=self.difficulty)
        if self.difficulty == 'hard':
            self.current_task = next(t for t in TASKS if t['id'] == 'hard')
        else:
            self.current_task = random.choice([t for t in TASKS if t['id'] != 'hard'])
        return self._get_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Apply action, update state, generate logs, compute reward and termination.
        Returns (observation, reward, done, info).
        """
        self._state.apply_action(action.action_type, action.parameters)
        self._state.generate_step_logs(difficulty=self.difficulty)
        obs = self._get_observation()
        task_id = self.current_task["id"]
        reward = compute_reward(action.action_type, action.parameters, self._state)
        done = self._state.is_terminal(max_steps=self.max_steps, task_id=task_id)
        info = {"steps": self._state.steps}
        return obs, reward, done, info

    def state(self) -> Dict[str, Any]:
        """Return full internal state (includes attackers list for grading)."""
        return {
            **self._state.to_dict(),
            "current_task": self.current_task,
            "suspicion_scores": self._state.suspicion_scores,
            "attackers": self._state.attackers,
        }

    def _get_observation(self) -> Observation:
        """Build observation from current state."""
        suspicion_scores = self._state.suspicion_scores
        top_suspicious = sorted(suspicion_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        top_suspicious_ips = [ip for ip, _ in top_suspicious]

        inv_results = {}
        for ip, count in self._state.investigation_history.items():
            inv_results[ip] = {
                "investigated_count": count,
                "is_attacker": ip in self._state.attackers,
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