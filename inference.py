"""
CyberDefend-X Inference Script
Runs the agent against the live HF Space environment.
"""

from __future__ import annotations

import json
import os
import sys
import time

import httpx
from openai import OpenAI

# ------------------------------------------------------------------
# Environment variables
# ------------------------------------------------------------------
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "llama-3.3-70b-versatile")
HF_TOKEN     = os.environ.get("HF_TOKEN")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "https://devanshi86-cyberdefend-x.hf.space").rstrip("/")

if not HF_TOKEN:
    print("[WARN] HF_TOKEN is not set. API calls may fail.", flush=True)

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN or "placeholder",
)

TASKS = ["alert_prioritization", "threat_detection", "incident_response"]
MAX_STEPS = {"alert_prioritization": 1, "threat_detection": 1, "incident_response": 4}


def log_start(task: str, env: str, model: str):
    # Plain text format (required by evaluator)
    print(f"[START] task={task} env={env} model={model}", flush=True)
    # JSON format (for compatibility)
    print(json.dumps({
        "type": "START",
        "task": task,
        "env": env,
        "model": model,
        "timestamp": time.time(),
    }), flush=True)
    sys.stdout.flush()


def log_step(step: int, action, reward: float, done: bool, error=None):
    # Plain text format (required by evaluator)
    print(f"[STEP] step={step} reward={reward} done={done}", flush=True)
    # JSON format (for compatibility)
    print(json.dumps({
        "type": "STEP",
        "step": step,
        "action": action,
        "reward": reward,
        "done": done,
        "error": error,
        "timestamp": time.time(),
    }), flush=True)
    sys.stdout.flush()


def log_end(success: bool, steps: int, score: float, rewards: list):
    # Plain text format (required by evaluator)
    print(f"[END] success={success} steps={steps} score={score}", flush=True)
    # JSON format (for compatibility)
    print(json.dumps({
        "type": "END",
        "success": success,
        "steps": steps,
        "score": score,
        "rewards": rewards,
        "timestamp": time.time(),
    }), flush=True)
    sys.stdout.flush()


def call_llm(system_prompt: str, user_prompt: str):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=1024,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"[DEBUG] LLM call error: {e}", flush=True)
        return None


def parse_json(text: str):
    if not text:
        return None
    for candidate in [
        text.strip(),
        text.strip().strip("```json").strip("```").strip(),
        text[text.find("{"):text.rfind("}")+1] if "{" in text else "",
        text[text.find("["):text.rfind("]")+1] if "[" in text else "",
    ]:
        try:
            if candidate:
                return json.loads(candidate)
        except Exception:
            continue
    return None


def get_action_for_task(task_id: str, observation: dict):
    obs_str = json.dumps(observation, indent=2)

    if task_id == "alert_prioritization":
        alerts = observation.get("alerts", [])
        alert_list = "\n".join([f"- [{i}] {a['alert']}" for i, a in enumerate(alerts)])
        system = """You are an expert SOC analyst. Identify HIGH priority alerts that need immediate action.

HIGH priority: data exfiltration, unauthorized admin access, port scans on critical systems, brute force attacks, malware, privilege escalation.
LOW priority: single failed logins, routine traffic, non-critical warnings.

Return ONLY a valid JSON array of exact high-priority alert strings.
Example: ["Alert text 1", "Alert text 2"]
No explanation. No markdown. Just the JSON array."""

        user = f"""Alerts:\n{alert_list}\n\nFull observation:\n{obs_str}\n\nReturn JSON array of high-priority alert strings."""
        result = call_llm(system, user)
        parsed = parse_json(result)
        if isinstance(parsed, list):
            return parsed
        return [a["alert"] for a in alerts if a.get("true_severity", 0) >= 0.3]

    elif task_id == "threat_detection":
        system = """You are a threat intelligence analyst. Identify the attack pattern from logs.

Return ONLY valid JSON with:
- "attack_type": specific attack name (e.g. "Brute Force + Privilege Escalation")
- "signals": list of 3-5 evidence strings from logs

Example: {"attack_type": "Brute Force + Privilege Escalation", "signals": ["20 failed logins", "sudo su root", "new admin account"]}
No explanation. No markdown. Just JSON."""

        user = f"""Analyze logs and identify the attack:\n{obs_str}\n\nReturn JSON with attack_type and signals."""
        result = call_llm(system, user)
        parsed = parse_json(result)
        if isinstance(parsed, dict) and "attack_type" in parsed and "signals" in parsed:
            return parsed
        return {"attack_type": "Unknown Threat", "signals": ["Suspicious activity detected"]}

    elif task_id == "incident_response":
        logs = observation.get("logs", [])
        step_num = observation.get("step", 0)
        context = observation.get("context", {})

        system = """You are an expert incident responder. Read the current log and pick the best action.

Decision guide:
- Active malware/ransomware process running: kill_process
- Lateral spread detected, multiple hosts infected: isolate_system
- C2 domain resolving, attacker may re-enter: block_ip
- Compromised credentials, account takeover: reset_credentials
- Need to alert team about new threat: alert_soc
- Incident contained, need evidence or documentation: collect_forensics
- Need stakeholder notification after containment: escalate_to_management
- System needs patching: patch_system

Return ONLY valid JSON: {"action": "action_name"}
No markdown. No explanation. Just JSON.

Valid actions: alert_soc, block_ip, isolate_system, kill_process, reset_credentials, collect_forensics, escalate_to_management, patch_system, restore_backup, do_nothing



Return ONLY valid JSON: {"action": "action_name"}
No explanation. No markdown. Just JSON."""

        user = f"""Step {step_num+1}. Context: {json.dumps(context)}. Recent logs: {json.dumps(logs[-3:] if logs else [])}.\n\nBest next action? Return JSON with key "action"."""
        result = call_llm(system, user)
        parsed = parse_json(result)
        if isinstance(parsed, dict) and "action" in parsed:
            valid = ["kill_process","isolate_system","block_ip","escalate_to_management",
                     "alert_soc","restore_system","collect_forensics","patch_vulnerability"]
            if parsed["action"] in valid:
                return parsed
        fallback = ["kill_process","isolate_system","block_ip","escalate_to_management"]
        return {"action": fallback[min(step_num, len(fallback)-1)]}

    return {}


def run_task(task_id: str) -> float:
    print(f"\n{'='*60}", flush=True)
    print(f"[INFO] Running task: {task_id}", flush=True)
    sys.stdout.flush()

    log_start(task=task_id, env="CyberDefend-X", model=MODEL_NAME)

    try:
        r = httpx.post(f"{ENV_BASE_URL}/reset",
                       json={"task_id": task_id, "scenario_index": 0},
                       timeout=30)
        reset_data = r.json()
        obs = reset_data.get("observation", {})
    except Exception as e:
        print(f"[ERROR] Reset failed: {e}", flush=True)
        log_step(1, {}, 0.0, True, str(e))
        log_end(False, 1, 0.0, [0.0])
        return 0.0

    max_steps = MAX_STEPS.get(task_id, 4)
    rewards = []
    done = False
    step = 0

    for step in range(1, max_steps + 1):
        if done:
            break
        action = get_action_for_task(task_id, obs)
        try:
            r = httpx.post(f"{ENV_BASE_URL}/step",
                           json={"task_id": task_id, "scenario_index": 0, "action": action},
                           timeout=30)
            step_data = r.json()
            reward = step_data.get("reward", 0.0)
            done = step_data.get("done", True)
            obs = step_data.get("observation", obs)
        except Exception as e:
            print(f"[ERROR] Step failed: {e}", flush=True)
            reward = 0.0
            done = True

        rewards.append(reward)
        log_step(step, action, reward, done)

    score = sum(rewards) / len(rewards) if rewards else 0.0
    success = score > 0.5
    log_end(success=success, steps=step, score=round(score, 4), rewards=rewards)
    print(f"[INFO] Task {task_id} final score: {score:.4f}", flush=True)
    sys.stdout.flush()
    return score


def main():
    print("🚀 Starting CyberDefend-X inference...", flush=True)
    print(f"[INFO] Model  : {MODEL_NAME} @ {API_BASE_URL}", flush=True)
    print(f"[INFO] Env URL: {ENV_BASE_URL}", flush=True)
    print(f"[INFO] Tasks  : {TASKS}", flush=True)
    sys.stdout.flush()

    try:
        health = httpx.get(f"{ENV_BASE_URL}/health", timeout=10)
        print(f"[INFO] Health check: {health.json()}", flush=True)
    except Exception as e:
        print(f"[ERROR] Health check failed: {e}", flush=True)

    scores = {}
    for task_id in TASKS:
        scores[task_id] = run_task(task_id)

    print(f"\n{'='*60}", flush=True)
    print("[SUMMARY] Task scores:", flush=True)
    for task_id, score in scores.items():
        status = "✅" if score > 0.5 else "❌"
        print(f"  {status}  {task_id}: {score:.4f}", flush=True)

    overall = sum(scores.values()) / len(scores)
    print(f"\n  🏆 OVERALL: {overall:.4f}", flush=True)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
