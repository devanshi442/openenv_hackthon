"""
Task 3: Sequential Incident Response (Hard)
Objective: Take optimal containment actions step-by-step against an evolving intrusion.
Grader: Per-step weighted scoring with penalty for wrong/destructive actions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


VALID_ACTIONS = [
    "alert_soc",
    "block_ip",
    "isolate_system",
    "kill_process",
    "reset_credentials",
    "collect_forensics",
    "escalate_to_management",
    "patch_system",
    "restore_backup",
    "do_nothing",
]

# Penalized when taken during an active attack
DESTRUCTIVE_ACTIONS = {"restore_backup", "do_nothing"}

SCENARIOS: List[Dict[str, Any]] = [
    {
        "id": "T3-001",
        "title": "Active Ransomware Spread",
        "steps": [
            {
                "step": 1,
                "log": "Ransomware process detected on WORKSTATION-07. Files being encrypted.",
                "optimal_action": "kill_process",
                "acceptable_actions": ["kill_process", "isolate_system"],
                "weight": 0.25,
                "hint": "Stop the encryption process immediately.",
            },
            {
                "step": 2,
                "log": "Ransomware process killed. But lateral spread via SMB detected to 3 other hosts.",
                "optimal_action": "isolate_system",
                "acceptable_actions": ["isolate_system", "block_ip"],
                "weight": 0.30,
                "hint": "Contain the spread by isolating the infected subnet.",
            },
            {
                "step": 3,
                "log": "Spread contained. C2 domain still resolving. Attacker may re-enter.",
                "optimal_action": "block_ip",
                "acceptable_actions": ["block_ip", "alert_soc"],
                "weight": 0.25,
                "hint": "Block the C2 IP to prevent re-infection.",
            },
            {
                "step": 4,
                "log": "Network secured. Need to document incident and notify stakeholders.",
                "optimal_action": "collect_forensics",
                "acceptable_actions": ["collect_forensics", "escalate_to_management"],
                "weight": 0.20,
                "hint": "Collect forensic evidence before remediation.",
            },
        ],
    },
    {
        "id": "T3-002",
        "title": "Insider Threat Credential Abuse",
        "steps": [
            {
                "step": 1,
                "log": "Admin account 'jsmith' logged in from 3 different countries in 2 hours.",
                "optimal_action": "alert_soc",
                "acceptable_actions": ["alert_soc", "reset_credentials"],
                "weight": 0.20,
                "hint": "Alert the team — could be account takeover.",
            },
            {
                "step": 2,
                "log": "Confirmed: jsmith's credentials leaked on dark web. Active session ongoing.",
                "optimal_action": "reset_credentials",
                "acceptable_actions": ["reset_credentials", "isolate_system"],
                "weight": 0.35,
                "hint": "Invalidate the compromised credentials immediately.",
            },
            {
                "step": 3,
                "log": "Attacker pivoted to DB server. Sensitive data being accessed.",
                "optimal_action": "isolate_system",
                "acceptable_actions": ["isolate_system", "block_ip"],
                "weight": 0.30,
                "hint": "Isolate the DB server to stop data exfiltration.",
            },
            {
                "step": 4,
                "log": "Access stopped. Audit trail needed for legal/compliance.",
                "optimal_action": "collect_forensics",
                "acceptable_actions": ["collect_forensics", "escalate_to_management"],
                "weight": 0.15,
                "hint": "Preserve evidence for legal proceedings.",
            },
        ],
    },
]


def compute_step_reward(
    action: str,
    reason: str,
    step_data: Dict[str, Any],
) -> Tuple[float, Dict[str, float], str]:
    """
    Per-step reward:
      +weight * 1.0  → optimal action
      +weight * 0.5  → acceptable (non-optimal) action
      -0.3           → destructive/wrong action during active attack
      +0.1           → explainability bonus (reason matches hint keywords)
    """
    breakdown: Dict[str, float] = {}
    feedback_parts: List[str] = []
    weight = step_data["weight"]

    if action == step_data["optimal_action"]:
        action_score = weight * 1.0
        feedback_parts.append(f"✅ Optimal action '{action}'!")
    elif action in step_data["acceptable_actions"]:
        action_score = weight * 0.5
        feedback_parts.append(f"⚠️ Acceptable but not optimal. Best: '{step_data['optimal_action']}'.")
    elif action in DESTRUCTIVE_ACTIONS:
        action_score = -0.3
        feedback_parts.append(f"💥 Destructive/passive action '{action}' during active attack!")
    else:
        action_score = 0.0
        feedback_parts.append(f"❌ Wrong action '{action}'. Hint: {step_data['hint']}")

    breakdown["action_score"] = round(action_score, 4)

    # Explainability bonus: reward well-reasoned responses
    hint_keywords = [kw for kw in step_data["hint"].lower().split() if len(kw) > 4]
    reason_norm = reason.lower()
    keyword_hits = sum(1 for kw in hint_keywords if kw in reason_norm)
    explainability_bonus = 0.1 if keyword_hits >= 2 else 0.0
    breakdown["explainability_bonus"] = explainability_bonus
    if explainability_bonus:
        feedback_parts.append("🧠 Good reasoning!")

    # Cap total at 0.0 minimum; explainability cannot compensate a destructive penalty below 0
    total = round(min(max(action_score + explainability_bonus, 0.001), 0.999), 4)
    return total, breakdown, " ".join(feedback_parts)


def get_scenario(index: int = 0) -> Dict[str, Any]:
    if not SCENARIOS:
        raise ValueError("No scenarios defined.")
    return SCENARIOS[index % len(SCENARIOS)]
