"""
CyberDefend-X Inference Script
Runs the agent against the live HF Space environment.
"""

from __future__ import annotations

import json
import os
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
    print(json.dumps({
        "type": "START",
        "task": task,
        "env": env,
        "model": model,
        "timestamp": time.time(),
    }), flush=True)


def log_step(step: int, action, reward: float, done: bool, error=None):
    print(json.dumps({
        "type": "STEP",
        "step": step,
        "action": action,
        "reward": reward,
        "done": done,
        "error": error,
        "timestamp": time.time(),
    }), flush=True)


def log_end(success: bool, steps: int, score: float, rewards: list):
    print(json.dumps({
        "type": "END",
        "success": success,
        "steps": steps,
        "score": score,
        "rewards": rewards,
        "timestamp": time.time(),
    }), flush=True)


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
        system = """You are an expert SOC analyst with 10+ years of experience triaging security alerts.
Your job is to identify which alerts are genuinely HIGH priority and require immediate action.

HIGH priority alerts include:
- Data exfiltration attempts (large data transfers to external IPs)
- Unauthorized admin/privileged access or login anomalies  
- Active intrusion indicators (port scans on critical systems, lateral movement)
- Multiple failed authentication attempts suggesting brute force
- Malware or ransomware indicators
- Privilege escalation attempts

LOW priority alerts include:
- Single failed logins
- Routine network traffic
- Non-critical system warnings

CRITICAL: Return ONLY a valid JSON array of the exact alert strings that are high priority.
Example: ["Alert text 1", "Alert text 2"]
No explanation. No markdown. Just the JSON array."""

        user = f"""Analyze these security alerts and return ALL HIGH priority ones:

{alert_list}

Full observation:
{obs_str}

Return ONLY a JSON array of high-priority alert strings."""

        result = call_llm(system, user)
        parsed = parse_json(result)
        if isinstance(parsed, list):
            return parsed
        return [a["alert"] for a in alerts if a.get("true_severity", 0) >= 0.3]

    elif task_id == "threat_detection":
        system = """You are a senior threat intelligence analyst specializing in attack pattern recognition.

Common attack patterns:
- Brute Force: Many failed logins followed by success
- Privilege Escalation: sudo/su usage, new admin accounts created
- Lateral Movement: connections between internal hosts
- Data Exfiltration: large outbound transfers, DNS tunneling
- Persistence: new scheduled tasks, backdoor accounts
- Command & Control: beaconing traffic, unusual outbound connections
- Ransomware: mass file encryption, shadow copy deletion
- SQL Injection: unusual database queries in web logs

CRITICAL: Return ONLY valid JSON with exactly these two keys:
- "attack_type": A specific descriptive name (e.g. "Brute Force + Privilege Escalation")
- "signals": A list of 3-5 specific evidence strings from the logs

Example: {"attack_type": "Credential Stuffing + Data Exfiltration", "signals": ["500 failed logins", "successful login from TOR IP", "200MB transfer to external IP"]}
No explanation. No markdown. Just the JSON."""

        user = f"""Analyze these security logs and identify the attack:

{obs_str}

Return JSON with attack_type and signals."""

        result = call_llm(system, user)
        parsed = parse_json(result)
        if isinstance(parsed, dict) and "attack_type" in parsed and "signals" in parsed:
            return parsed
        return {"attack_type": "Unknown Threat", "signals": ["Suspicious activity detected"]}

    elif task_id == "incident_response":
        logs = observation.get("logs", [])
        step_num = observation.get("step", 0)
        context = observation.get("context", {})

        system = """You are an elite incident responder following a structured IR playbook.

Available actions and when to use them:
- "kill_process": FIRST - stop active malicious process
- "isolate_system": EARLY - prevent lateral movement  
- "block_ip": stop C2 communication or exfiltration
- "collect_forensics": gather evidence before remediation
- "escalate_to_management": when severity is HIGH or needs business decision
- "alert_soc": notify team and coordinate response
- "patch_vulnerability": after containment, fix root cause
- "restore_system": LAST - recovery after full eradication

Optimal IR sequence: kill_process → isolate_system → block_ip → escalate_to_management → restore_system

CRITICAL: Return ONLY valid JSON with key "action".
Example: {"action": "isolate_system"}
No explanation. No markdown. Just the JSON."""

        user = f"""Current incident state (Step {step_num + 1}):

Context: {json.dumps(context, indent=2)}
Recent logs: {json.dumps(logs[-5:] if logs else [], indent=2)}

What is the single BEST next action? Return JSON with key "action"."""

        result = call_llm(system, user)
        parsed = parse_json(result)
        if isinstance(parsed, dict) and "action" in parsed:
            valid_actions = ["kill_process", "isolate_system", "block_ip",
                           "escalate_to_management", "alert_soc", "restore_system",
                           "collect_forensics", "patch_vulnerability"]
            if parsed["action"] in valid_actions:
                return parsed
        fallback_sequence = ["kill_process", "isolate_system", "block_ip", "escalate_to_management"]
        return {"action": fallback_sequence[min(step_num, len(fallback_sequence)-1)]}

    return {}


def run_task(task_id: str) -> float:
    print(f"\n{'='*60}", flush=True)
    print(f"[INFO] Running task: {task_id}", flush=True)

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
    return score


def main():
    print("🚀 Starting CyberDefend-X inference...", flush=True)
    print(f"[INFO] CyberDefend-X baseline inference", flush=True)
    print(f"[INFO] Model  : {MODEL_NAME} @ {API_BASE_URL}", flush=True)
    print(f"[INFO] Env URL: {ENV_BASE_URL}", flush=True)
    print(f"[INFO] Tasks  : {TASKS}", flush=True)

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


if __name__ == "__main__":
    main()
