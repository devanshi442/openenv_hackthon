"""CyberDefend-X: Typed Pydantic models for OpenEnv compliance."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Observation(BaseModel):
    """Current environment state presented to the agent."""

    task_id: str = Field(..., description="Active task identifier")
    step: int = Field(..., description="Current step index (0-based)")
    context: Dict[str, Any] = Field(..., description="Task-specific context data")
    logs: List[str] = Field(default_factory=list, description="Security log entries")
    alerts: List[Dict[str, Any]] = Field(default_factory=list, description="Active security alerts")
    instructions: str = Field(..., description="Natural language instructions for the agent")
    history: List[Dict[str, Any]] = Field(default_factory=list, description="Previous steps taken")
    done: bool = Field(default=False, description="Whether the episode has ended")


class Action(BaseModel):
    """Agent action — every action must include a reason (explainability bonus)."""

    action: Any = Field(..., description="The action payload (varies by task)")
    reason: str = Field(..., description="Explanation for why this action was chosen")


class Reward(BaseModel):
    """Structured reward with partial breakdown."""

    total: float = Field(..., ge=0.0, le=1.0, description="Total reward [0.0, 1.0]")
    breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-component reward breakdown for transparency",
    )
    feedback: str = Field(default="", description="Human-readable feedback on the action")


class StepResult(BaseModel):
    """Full result returned by step()."""

    observation: Observation
    reward: float = Field(..., ge=0.0, le=1.0, description="Step reward [0.0, 1.0]")
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class ResetResult(BaseModel):
    """Result returned by reset()."""

    observation: Observation
