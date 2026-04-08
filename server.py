"""
CyberDefend-X FastAPI Server
Exposes the OpenEnv REST interface.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env import CyberDefendEnv
from models import Action, Observation, StepResult, ResetResult

logger = logging.getLogger(__name__)

app = FastAPI(
    title="CyberDefend-X: SOC Decision Intelligence Environment",
    description="OpenEnv-compliant cybersecurity RL environment for SOC analyst training.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global env registry keyed by (task_id, scenario_index)
_envs: Dict[str, CyberDefendEnv] = {}

TASKS = [
    {
        "id": "alert_prioritization",
        "name": "Alert Prioritization",
        "difficulty": "easy",
        "max_steps": 1,
        "max_scenarios": 3,
    },
    {
        "id": "threat_detection",
        "name": "Multi-Signal Threat Detection",
        "difficulty": "medium",
        "max_steps": 1,
        "max_scenarios": 4,
    },
    {
        "id": "incident_response",
        "name": "Sequential Incident Response",
        "difficulty": "hard",
        "max_steps": 4,
        "max_scenarios": 2,
    },
]

VALID_TASK_IDS = {t["id"] for t in TASKS}


def _get_env(task_id: str, scenario_index: int = 0) -> CyberDefendEnv:
    key = f"{task_id}_{scenario_index}"
    if key not in _envs:
        _envs[key] = CyberDefendEnv(task_id=task_id, scenario_index=scenario_index)
    return _envs[key]


def _validate_task(task_id: str, scenario_index: int) -> None:
    task = next((t for t in TASKS if t["id"] == task_id), None)
    if task is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown task_id '{task_id}'. Valid options: {sorted(VALID_TASK_IDS)}",
        )
    max_scenarios = task["max_scenarios"]
    if not (0 <= scenario_index < max_scenarios):
        raise HTTPException(
            status_code=400,
            detail=f"scenario_index must be in [0, {max_scenarios - 1}] for task '{task_id}'.",
        )


# ------------------------------------------------------------------
# Request schemas
# ------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: int = 0
    scenario_index: int = 0


class StepRequest(BaseModel):
    task_id: str = "alert_prioritization"
    scenario_index: int = 0
    action: Any
    reason: str = ""


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "env": "CyberDefend-X"}


@app.post("/reset", response_model=ResetResult)
def reset(req: ResetRequest = None):
    if req is None:
        req = ResetRequest()

    _validate_task(req.task_id, req.scenario_index)
    try:
        env = _get_env(req.task_id, req.scenario_index)
        return env.reset()
    except Exception as e:
        logger.exception("Error in /reset")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step", response_model=StepResult)
def step(req: StepRequest = None ):
    if req is None:
        req = StepRequest()
    _validate_task(req.task_id, req.scenario_index)
    env = _get_env(req.task_id, req.scenario_index)
    if env._current_obs is None:
        raise HTTPException(status_code=400, detail="Call /reset before /step.")
    try:
        action = Action(action=req.action, reason=req.reason)
        return env.step(action)
    except Exception as e:
        logger.exception("Error in /step")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state", response_model=Observation)
def state(task_id: str = "alert_prioritization", scenario_index: int = 0):
    _validate_task(task_id, scenario_index)
    env = _get_env(task_id, scenario_index)
    try:
        return env.state()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/tasks")
def list_tasks():
    return {"tasks": TASKS}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=7860, reload=False)
