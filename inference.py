"""
CyberDefend-X — Baseline Inference Script
Uses the OpenAI-compatible API client to run an LLM agent against all 3 tasks.

Required environment variables:
  API_BASE_URL   — e.g. https://api.groq.com/openai/v1
  MODEL_NAME     — e.g. llama-3.3-70b-versatile
  HF_TOKEN       — Your API key
  ENV_BASE_URL   — e.g. "https://your-space.hf.space" or "https://devanshi86-cyberdefend-x.hf.space"

Logging format: strictly [START], [STEP], [END] as required by the spec.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI

print("🚀 Starting CyberDefend-X inference...", flush=True)

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "llama-3.3-70b-versatile")
HF_TOKEN     = os.environ.get("HF_TOKEN")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "https://devanshi86-cyberdefend-x.hf.space").rstrip("/")

SUCCESS_SCORE_THRESHOLD = float(os.environ.get("SUCCESS_SCORE_THRESHOLD", "0.5"))

TASKS: List[Dict[str, Any]] = [
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
# Logging helpers
# ------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(json.dumps({"type": "START", "task": task, "env": env,
                      "model": model, "timestamp": time.time()}), flush=True)

def log_step(step: int, action: Any, reward: float, done: bool, error: Optional[str]) -> None:
    print(json.dumps({"type": "STEP", "step": step, "action": action,
                      "reward": reward, "done": done, "error": error,
                      "timestamp": time.time()}), flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    print(json.dumps({"type": "END", "success": success, "steps": steps,
                      "score": score, "rewards": rewards,
                      "timestamp": time.time()}), flush=True)

# ------------------------------------------------------------------
# Task-specific system prompts (tuned for high scores)
# ------------------------------------------------------------------

SYSTEM_PROMPTS = {
    "alert_prioritization": """You are an elite SOC analyst with 15 years of experience.
Rank security alerts from MOST to LEAST critical using these rules:
1. CRITICAL (0.9-1.0): Active exfiltration, ransomware executing, active malware
2. HIGH (0.7-0.8): Account compromise, privilege escalation, C2 communication, lateral movement
3. MEDIUM (0.4-0.6): Port scans, suspicious access, policy violations
4. LOW (0.1-0.3): Few failed logins, outdated software, informational

You MUST respond with valid JSON only — no markdown, no preamble.
Format: {"ranking": ["most critical alert text exactly as given", "second", ...], "reason": "brief justification"}""",

    "threat_detection": """You are an elite SOC threat analyst specializing in attack pattern recognition.
Analyze correlated log entries and identify the exact attack type and key signals.

COMMON PATTERNS:
- Brute Force + Privilege Escalation: many failed logins → sudo/su → new admin backdoor account
- Command and Control (C2) with Persistence: DNS to C2 domain → unusual port (4444) → PowerShell encoded → scheduled task
- Ransomware: mass file rename/.locked → shadow copy deletion (vssadmin) → ransom note → disk spike
- Lateral Movement + Credential Theft: SMB scan → pass-the-hash → Mimikatz → RDP to DC

Match the attack type name EXACTLY to what fits (e.g. "Brute Force + Privilege Escalation").
You MUST respond with valid JSON only — no markdown, no preamble.
Format: {"attack_type": "exact attack name", "signals": ["key signal 1", "key signal 2", "key signal 3"], "reason": "explanation"}""",

    "incident_response": """You are an elite incident responder. Choose the SINGLE best containment action.

DECISION RULES (strict priority):
- "process detected" OR "files being encrypted" → kill_process
- "lateral spread" OR "SMB scan" OR "spread to other hosts" → isolate_system
- "C2 domain" OR "attacker IP" OR "may re-enter" → block_ip
- "credentials leaked" OR "compromised account" OR "active session" → reset_credentials
- "need evidence" OR "forensics" OR "audit trail" OR "document" → collect_forensics
- "stakeholders" OR "legal" OR "compliance" OR "notify" → escalate_to_management
- "suspicious activity" OR "initial detection" OR "investigate" → alert_soc

Your reason MUST mention keywords from the log (this gives an explainability bonus).
You MUST respond with valid JSON only — no markdown, no preamble.
Valid actions: alert_soc, block_ip, isolate_system, kill_process, reset_credentials, collect_forensics, escalate_to_management, patch_system, restore_backup, do_nothing
Format: {"action": "chosen_action", "reason": "quote keywords from the log that justify this action"}""",
}

# ------------------------------------------------------------------
# LLM call
# ------------------------------------------------------------------

def get_llm_action(instructions: str, history: List[Dict], task_id: str) -> Dict[str, Any]:
    system_prompt = SYSTEM_PROMPTS.get(task_id, SYSTEM_PROMPTS["incident_response"])

    history_text = ""
    if history:
        history_text = "\n\n--- PREVIOUS STEPS (learn from rewards/feedback) ---\n"
        for h in history:
            history_text += (
                f"Step {h['step']}: action={h['action']}, "
                f"reward={h['reward']:.3f}, feedback={h.get('feedback', '')}\n"
            )
        history_text += "--- End of history ---"

    FORMAT_REMINDERS = {
        "alert_prioritization": '\n\nReturn ONLY: {"ranking": ["alert 1", "alert 2", ...], "reason": "..."}',
        "threat_detection":     '\n\nReturn ONLY: {"attack_type": "name", "signals": ["s1","s2","s3"], "reason": "..."}',
        "incident_response":    '\n\nReturn ONLY: {"action": "one_valid_action", "reason": "use keywords from the log"}',
    }

    user_msg = instructions + history_text + FORMAT_REMINDERS.get(task_id, "")

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_msg},
            ],
            max_tokens=512,
            temperature=0.1,
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1] if len(parts) > 1 else raw
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())

    except json.JSONDecodeError as e:
        print(f"[DEBUG] JSON parse error: {e} | raw={raw!r}", flush=True)
    except Exception as e:
        print(f"[DEBUG] LLM call error: {e}", flush=True)

    fallbacks: Dict[str, Dict[str, Any]] = {
        "alert_prioritization": {"ranking": [], "reason": "LLM unavailable."},
        "threat_detection":     {"attack_type": "Unknown", "signals": [], "reason": "LLM unavailable."},
        "incident_response":    {"action": "alert_soc", "reason": "LLM unavailable."},
    }
    return fallbacks.get(task_id, {"action": "alert_soc", "reason": "LLM unavailable."})

# ------------------------------------------------------------------
# HTTP helpers
# ------------------------------------------------------------------

def api_reset(task_id: str, scenario_index: int) -> Dict:
    r = httpx.post(f"{ENV_BASE_URL}/reset",
                   json={"task_id": task_id, "scenario_index": scenario_index}, timeout=30)
    r.raise_for_status()
    return r.json()

def api_step(task_id: str, scenario_index: int, action: Any, reason: str) -> Dict:
    r = httpx.post(f"{ENV_BASE_URL}/step",
                   json={"task_id": task_id, "scenario_index": scenario_index,
                         "action": action, "reason": reason}, timeout=30)
    r.raise_for_status()
    return r.json()

def build_action_payload(task_id: str, llm_response: Dict[str, Any]) -> Any:
    if task_id == "alert_prioritization":
        return llm_response.get("ranking", [])
    if task_id == "threat_detection":
        return {"attack_type": llm_response.get("attack_type", ""),
                "signals": llm_response.get("signals", [])}
    return {"action": llm_response.get("action", "alert_soc")}

# ------------------------------------------------------------------
# Run one full task episode
# ------------------------------------------------------------------

def run_task(task_cfg: Dict[str, Any]) -> float:
    task_id        = task_cfg["task_id"]
    scenario_index = task_cfg["scenario_index"]
    max_steps      = task_cfg["max_steps"]

    log_start(task=task_id, env="CyberDefend-X", model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    history: List[Dict] = []
    score   = 0.0
    success = False

    try:
        reset_result = api_reset(task_id, scenario_index)
        obs  = reset_result.get("observation", {})
        done = obs.get("done", False)

        for step_num in range(1, max_steps + 1):
            if done:
                break

            instructions = obs.get("instructions", "")
            llm_response = get_llm_action(instructions, history, task_id)
            reason       = llm_response.pop("reason", "No reason provided.")
            action_payload = build_action_payload(task_id, llm_response)

            step_result = api_step(task_id, scenario_index, action_payload, reason)
            reward  = float(step_result.get("reward", 0.0))
            done    = bool(step_result.get("done", True))
            obs     = step_result.get("observation", obs)
            info    = step_result.get("info", {})

            rewards.append(reward)
            steps_taken = step_num
            history.append({"step": step_num, "action": action_payload,
                             "reward": reward, "feedback": info.get("feedback", "")})

            log_step(step=step_num, action=action_payload, reward=reward, done=done, error=None)

        score   = round(sum(rewards) / max(max_steps, 1), 4)
        score   = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Task {task_id} failed: {e}", flush=True)
        log_step(step=steps_taken, action=None, reward=0.0, done=True, error=str(e))

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    print(f"[INFO] CyberDefend-X baseline inference", flush=True)
    print(f"[INFO] Model  : {MODEL_NAME} @ {API_BASE_URL}", flush=True)
    print(f"[INFO] Env URL: {ENV_BASE_URL}", flush=True)
    print(f"[INFO] Tasks  : {[t['task_id'] for t in TASKS]}", flush=True)

    try:
        health = httpx.get(f"{ENV_BASE_URL}/health", timeout=10)
        print(f"[INFO] Health check: {health.json()}", flush=True)
    except Exception as e:
        print(f"[WARN] Could not reach environment: {e}", flush=True)

    all_scores: Dict[str, float] = {}

    for task_cfg in TASKS:
        task_id = task_cfg["task_id"]
        print(f"\n{'='*60}", flush=True)
        print(f"[INFO] Running task: {task_id}", flush=True)
        score = run_task(task_cfg)
        all_scores[task_id] = score
        print(f"[INFO] Task {task_id} final score: {score:.4f}", flush=True)

    print(f"\n{'='*60}", flush=True)
    print("[SUMMARY] Task scores:", flush=True)
    for tid, s in all_scores.items():
        status = "✅" if s >= SUCCESS_SCORE_THRESHOLD else "❌"
        print(f"  {status}  {tid}: {s:.4f}", flush=True)
    overall = round(sum(all_scores.values()) / max(len(all_scores), 1), 4)
    print(f"\n  🏆 OVERALL: {overall:.4f}", flush=True)


if __name__ == "__main__":
    main()
