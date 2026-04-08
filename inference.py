"""
CyberDefend-X — Baseline Inference Script
Uses the OpenAI-compatible client to run an LLM agent against all 3 tasks.
Required environment variables:
  API_BASE_URL   — e.g. https://api.x.ai/v1  (Grok / xAI)
  MODEL_NAME     — e.g. grok-3-mini
  HF_TOKEN       — Your Hugging Face token (or API key if using xAI)
  ENV_BASE_URL   — e.g. "https://devanshi86-cyberdefend-x.hf.space"
Logging format: strictly [START], [STEP], [END] as required by the spec.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI

print("🚀 Starting inference...")

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.x.ai/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "grok-3-mini")
import os
HF_TOKEN = os.getenv("HF_TOKEN")
ENV_BASE_URL = "https://devanshi86-cyberdefend-x.hf.space"

SUCCESS_SCORE_THRESHOLD = float(os.environ.get("SUCCESS_SCORE_THRESHOLD", "0.5"))

TASKS = [
    {"task_id": "alert_prioritization", "scenario_index": 0, "max_steps": 1},
    {"task_id": "threat_detection",     "scenario_index": 0, "max_steps": 1},
    {"task_id": "incident_response",    "scenario_index": 0, "max_steps": 4},
]

if not HF_TOKEN:
    print("[WARN] HF_TOKEN is not set. API calls may fail.", flush=True)

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN or "placeholder",
)

# ------------------------------------------------------------------
# Logging helpers — strict format required by evaluator
# ------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(json.dumps({
        "type": "START",
        "task": task,
        "env": env,
        "model": model,
        "timestamp": time.time(),
    }), flush=True)


def log_step(step: int, action: Any, reward: float, done: bool, error: Optional[str]) -> None:
    print(json.dumps({
        "type": "STEP",
        "step": step,
        "action": action,
        "reward": reward,
        "done": done,
        "error": error,
        "timestamp": time.time(),
    }), flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    print(json.dumps({
        "type": "END",
        "success": success,
        "steps": steps,
        "score": score,
        "rewards": rewards,
        "timestamp": time.time(),
    }), flush=True)

print("Calling /reset...")

# ------------------------------------------------------------------
# LLM call
# ------------------------------------------------------------------

def get_llm_action(instructions: str, history: List[Dict], task_id: str) -> Dict[str, Any]:
    """Call the LLM and parse a JSON action response."""
    system_prompt = (
        "You are an expert SOC (Security Operations Center) analyst AI. "
        "You MUST respond with valid JSON only — no markdown, no explanation outside the JSON. "
        "Every response must include a 'reason' field explaining your decision."
    )

    history_text = ""
    if history:
        history_text = "\n\nPrevious steps:\n" + "\n".join(
            f"  Step {h['step']}: action={h['action']}, reward={h['reward']:.2f}, feedback={h.get('feedback', '')}"
            for h in history
        )

    user_msg = instructions + history_text

    ACTION_HINTS = {
        "alert_prioritization": (
            '{"ranking": ["most critical alert", "second", ...], "reason": "..."}'
        ),
        "threat_detection": (
            '{"attack_type": "...", "signals": ["...", "..."], "reason": "..."}'
        ),
    }
    hint = ACTION_HINTS.get(task_id, '{"action": "one_of_valid_actions", "reason": "..."}')
    user_msg += f"\n\nRespond with JSON: {hint}"

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=512,
            temperature=0.2,
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception as e:
        print(f"[DEBUG] LLM call error: {e}", flush=True)
        fallbacks: Dict[str, Dict[str, Any]] = {
            "alert_prioritization": {"ranking": [], "reason": f"LLM error: {e}"},
            "threat_detection":     {"attack_type": "Unknown", "signals": [], "reason": f"LLM error: {e}"},
        }
        return fallbacks.get(task_id, {"action": "alert_soc", "reason": f"LLM error: {e}"})

# ------------------------------------------------------------------
# HTTP helpers
# ------------------------------------------------------------------

def api_reset(task_id: str, scenario_index: int) -> Dict:
    r = httpx.post(
        f"{ENV_BASE_URL}/reset",
        json={"task_id": task_id, "scenario_index": scenario_index},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def api_step(task_id: str, scenario_index: int, action: Any, reason: str) -> Dict:
    r = httpx.post(
        f"{ENV_BASE_URL}/step",
        json={
            "task_id": task_id,
            "scenario_index": scenario_index,
            "action": action,
            "reason": reason,
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


# ------------------------------------------------------------------
# Action payload builder
# ------------------------------------------------------------------

def build_action_payload(task_id: str, llm_response: Dict[str, Any]) -> Any:
    if task_id == "alert_prioritization":
        return llm_response.get("ranking", [])
    if task_id == "threat_detection":
        return {
            "attack_type": llm_response.get("attack_type", ""),
            "signals": llm_response.get("signals", []),
        }
    return {"action": llm_response.get("action", "alert_soc")}


# ------------------------------------------------------------------
# Run one task episode
# ------------------------------------------------------------------

def run_task(task_cfg: Dict[str, Any]) -> float:
    task_id: str = task_cfg["task_id"]
    scenario_index: int = task_cfg["scenario_index"]
    max_steps: int = task_cfg["max_steps"]

    log_start(task=task_id, env="CyberDefend-X", model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    history: List[Dict] = []
    score = 0.0
    success = False

    try:
        reset_result = api_reset(task_id, scenario_index)
        obs = reset_result["observation"]
        done = obs.get("done", False)

        for step in range(1, max_steps + 1):
            if done:
                break

            instructions = obs.get("instructions", "")
            llm_response = get_llm_action(instructions, history, task_id)
            reason = llm_response.pop("reason", "No reason provided.")
            action_payload = build_action_payload(task_id, llm_response)

            step_result = api_step(task_id, scenario_index, action_payload, reason)
            reward = step_result.get("reward", 0.0)
            done = step_result.get("done", True)
            obs = step_result.get("observation", obs)
            info = step_result.get("info", {})

            rewards.append(reward)
            steps_taken = step

            history.append({
                "step": step,
                "action": action_payload,
                "reward": reward,
                "feedback": info.get("feedback", ""),
            })

            log_step(step=step, action=action_payload, reward=reward, done=done, error=None)

        # Score normalised over this task's max_steps (not summed across all tasks)
        score = round(sum(rewards) / max(max_steps, 1), 4)
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Task {task_id} error: {e}", flush=True)
        log_step(step=steps_taken, action=None, reward=0.0, done=True, error=str(e))

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    print(f"[INFO] Starting CyberDefend-X baseline inference", flush=True)
    print(f"[INFO] Model: {MODEL_NAME} @ {API_BASE_URL}", flush=True)
    print(f"[INFO] Environment: {ENV_BASE_URL}", flush=True)

    all_scores: Dict[str, float] = {}

    for task_cfg in TASKS:
        task_id = task_cfg["task_id"]
        print(f"\n{'='*60}", flush=True)
        print(f"[INFO] Running task: {task_id}", flush=True)
        score = run_task(task_cfg)
        all_scores[task_id] = score
        print(f"[INFO] Task {task_id} final score: {score:.4f}", flush=True)

    print(f"\n{'='*60}", flush=True)
    print("[SUMMARY] All task scores:", flush=True)
    for tid, s in all_scores.items():
        print(f"  {tid}: {s:.4f}", flush=True)
    overall = round(sum(all_scores.values()) / len(all_scores), 4)
    print(f"  OVERALL: {overall:.4f}", flush=True)


if __name__ == "__main__":
    main()
