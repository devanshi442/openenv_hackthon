"""
CyberDefend-X: SOC Decision Intelligence Environment
Full OpenEnv spec implementation.

API:
  POST /reset     → ResetResult
  POST /step      → StepResult
  GET  /state     → Observation
  GET  /health    → {"status": "ok"}
"""

from __future__ import annotations
import json
from typing import Any, Dict, List, Optional

from models import Observation, Action, StepResult, ResetResult
from tasks import task1_alert_prioritization as t1
from tasks import task2_threat_detection as t2
from tasks import task3_incident_response as t3


TASK_IDS = ["alert_prioritization", "threat_detection", "incident_response"]
MAX_STEPS_PER_TASK = {"alert_prioritization": 1, "threat_detection": 1, "incident_response": 4}


class CyberDefendEnv:
    """
    CyberDefend-X environment state machine.
    Implements the OpenEnv interface: step(), reset(), state().
    """

    def __init__(self, task_id: str = "alert_prioritization", scenario_index: int = 0):
        assert task_id in TASK_IDS, f"Unknown task_id: {task_id}. Choose from {TASK_IDS}"
        self.task_id = task_id
        self.scenario_index = scenario_index
        self._step_count = 0
        self._done = False
        self._cumulative_reward = 0.0
        self._history: List[Dict[str, Any]] = []
        self._scenario: Dict[str, Any] = {}
        self._current_obs: Optional[Observation] = None

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self) -> ResetResult:
        """Reset the environment and return the initial observation."""
        self._step_count = 0
        self._done = False
        self._cumulative_reward = 0.0
        self._history = []
        self._scenario = self._load_scenario()
        self._current_obs = self._build_initial_obs()
        return ResetResult(observation=self._current_obs)

    def step(self, action: Action) -> StepResult:
        """Process an agent action and return the next state."""
        if self._done:
            return StepResult(
                observation=self._current_obs,
                reward=0.0,
                done=True,
                info={"error": "Episode already done. Call reset()."},
            )

        reward, info = self._process_action(action)
        self._cumulative_reward = round(self._cumulative_reward + reward, 4)
        self._step_count += 1

        max_steps = MAX_STEPS_PER_TASK[self.task_id]
        if self._step_count >= max_steps:
            self._done = True

        self._history.append({
            "step": self._step_count,
            "action": action.action,
            "reason": action.reason,
            "reward": reward,
            "feedback": info.get("feedback", ""),
        })

        self._current_obs = self._build_obs()
        return StepResult(
            observation=self._current_obs,
            reward=reward,
            done=self._done,
            info=info,
        )

    def state(self) -> Observation:
        """Return current environment state without advancing."""
        if self._current_obs is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        return self._current_obs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_scenario(self) -> Dict[str, Any]:
        if self.task_id == "alert_prioritization":
            return t1.get_scenario(self.scenario_index)
        elif self.task_id == "threat_detection":
            return t2.get_scenario(self.scenario_index)
        else:
            return t3.get_scenario(self.scenario_index)

    def _build_initial_obs(self) -> Observation:
        if self.task_id == "alert_prioritization":
            alerts = self._scenario["alerts"]
            return Observation(
                task_id=self.task_id,
                step=0,
                context={"scenario_id": self._scenario["id"]},
                alerts=alerts,
                logs=[],
                instructions=(
                    "You are a SOC analyst. Rank the following security alerts from most "
                    "to least critical. Return a JSON object with key 'ranking' containing "
                    "an ordered list of alert strings (most critical first), and a 'reason' "
                    "explaining your prioritization.\n\n"
                    f"Alerts: {json.dumps([a['alert'] for a in alerts], indent=2)}"
                ),
                history=[],
                done=False,
            )

        elif self.task_id == "threat_detection":
            logs = self._scenario["logs"]
            return Observation(
                task_id=self.task_id,
                step=0,
                context={"scenario_id": self._scenario["id"]},
                alerts=[],
                logs=logs,
                instructions=(
                    "You are a SOC threat analyst. Analyze the following correlated log entries "
                    "and classify the attack. Return a JSON object with:\n"
                    "  'attack_type': string — your attack classification\n"
                    "  'signals': list of strings — key signals you identified\n"
                    "  'reason': string — your reasoning\n\n"
                    f"Logs:\n" + "\n".join(f"  - {l}" for l in logs)
                ),
                history=[],
                done=False,
            )

        else:  # incident_response
            step_data = self._scenario["steps"][0]
            return Observation(
                task_id=self.task_id,
                step=0,
                context={
                    "scenario_id": self._scenario["id"],
                    "title": self._scenario["title"],
                    "total_steps": len(self._scenario["steps"]),
                    "valid_actions": t3.VALID_ACTIONS,
                },
                alerts=[],
                logs=[step_data["log"]],
                instructions=(
                    f"INCIDENT: {self._scenario['title']}\n\n"
                    f"Current situation: {step_data['log']}\n\n"
                    "You are the incident responder. Choose ONE action from the valid_actions list. "
                    "Return a JSON object with:\n"
                    "  'action': string — one of the valid actions\n"
                    "  'reason': string — why you chose this action\n\n"
                    f"Valid actions: {t3.VALID_ACTIONS}"
                ),
                history=[],
                done=False,
            )

    def _build_obs(self) -> Observation:
        """Build observation for steps after the first."""
        if self.task_id in ("alert_prioritization", "threat_detection"):
            # Single-step tasks — just mark done
            return Observation(
                task_id=self.task_id,
                step=self._step_count,
                context={"scenario_id": self._scenario["id"]},
                alerts=[],
                logs=[],
                instructions="Task complete.",
                history=self._history,
                done=True,
            )

        # Multi-step incident response
        steps = self._scenario["steps"]
        if self._step_count >= len(steps):
            return Observation(
                task_id=self.task_id,
                step=self._step_count,
                context={
                    "scenario_id": self._scenario["id"],
                    "title": self._scenario["title"],
                    "cumulative_reward": self._cumulative_reward,
                },
                alerts=[],
                logs=["Incident contained. Episode complete."],
                instructions="All response steps complete. Well done.",
                history=self._history,
                done=True,
            )

        step_data = steps[self._step_count]
        return Observation(
            task_id=self.task_id,
            step=self._step_count,
            context={
                "scenario_id": self._scenario["id"],
                "title": self._scenario["title"],
                "total_steps": len(steps),
                "valid_actions": t3.VALID_ACTIONS,
                "cumulative_reward": self._cumulative_reward,
            },
            alerts=[],
            logs=[step_data["log"]],
            instructions=(
                f"INCIDENT: {self._scenario['title']} — Step {self._step_count + 1}/{len(steps)}\n\n"
                f"Current situation: {step_data['log']}\n\n"
                "Choose ONE action. Return JSON with 'action' and 'reason'.\n"
                f"Valid actions: {t3.VALID_ACTIONS}"
            ),
            history=self._history,
            done=False,
        )

    def _process_action(self, action: Action) -> tuple[float, Dict[str, Any]]:
        if self.task_id == "alert_prioritization":
            predicted = action.action if isinstance(action.action, list) else []
            score, breakdown, feedback = t1.compute_reward(predicted, self._scenario)
            return score, {"breakdown": breakdown, "feedback": feedback}

        elif self.task_id == "threat_detection":
            act = action.action if isinstance(action.action, dict) else {}
            predicted_type = act.get("attack_type", "")
            predicted_signals = act.get("signals", [])
            score, breakdown, feedback = t2.compute_reward(
                predicted_type, predicted_signals, self._scenario
            )
            return score, {"breakdown": breakdown, "feedback": feedback}

        else:  # incident_response
            steps = self._scenario["steps"]
            step_data = steps[min(self._step_count, len(steps) - 1)]
            act = action.action if isinstance(action.action, dict) else {}
            chosen_action = act.get("action", "") if isinstance(act, dict) else str(act)
            score, breakdown, feedback = t3.compute_step_reward(
                chosen_action, action.reason, step_data
            )
            return score, {"breakdown": breakdown, "feedback": feedback}
