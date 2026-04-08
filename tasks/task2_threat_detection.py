"""
Task 2: Multi-Signal Threat Detection (Medium)
Objective: Classify the attack type from multiple correlated log signals.
Grader: Exact/partial match with signal coverage scoring.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


SCENARIOS: List[Dict[str, Any]] = [
    {
        "id": "T2-001",
        "logs": [
            "Failed login x20 on SSH port 22 from 192.168.4.11",
            "Access from foreign IP: 45.33.32.156 (Russia)",
            "Privilege escalation attempt: sudo su root",
            "New admin account created: backdoor_user",
        ],
        "true_attack_type": "Brute Force + Privilege Escalation",
        "accepted_partials": ["brute force", "privilege escalation", "credential attack"],
        "key_signals": ["failed login", "privilege escalation", "new admin"],
    },
    {
        "id": "T2-002",
        "logs": [
            "DNS query to known C2 domain: update.malware-c2.net",
            "Outbound connection on port 4444 (Metasploit default)",
            "PowerShell encoded command execution detected",
            "Scheduled task created: SystemUpdate at boot",
        ],
        "true_attack_type": "Command and Control (C2) with Persistence",
        "accepted_partials": ["c2", "command and control", "persistence", "malware"],
        "key_signals": ["dns c2", "port 4444", "powershell", "scheduled task"],
    },
    {
        "id": "T2-003",
        "logs": [
            "Mass file rename with .locked extension (3,412 files)",
            "Shadow copy deletion: vssadmin delete shadows",
            "Ransom note dropped: README_DECRYPT.txt",
            "Unusual spike in disk write operations",
        ],
        "true_attack_type": "Ransomware",
        "accepted_partials": ["ransomware", "crypto", "encryption attack"],
        "key_signals": ["file encryption", "shadow copy deletion", "ransom note"],
    },
    {
        "id": "T2-004",
        "logs": [
            "SMB scan across internal subnets 10.0.0.0/16",
            "Pass-the-hash authentication from workstation WS-042",
            "RDP connection from internal host to domain controller",
            "Mimikatz artifact detected in memory dump",
        ],
        "true_attack_type": "Lateral Movement + Credential Theft",
        "accepted_partials": ["lateral movement", "credential theft", "pass the hash", "mimikatz"],
        "key_signals": ["smb scan", "pass-the-hash", "mimikatz", "rdp lateral"],
    },
]


def _normalize(text: str) -> str:
    return text.lower().strip()


def compute_reward(
    predicted_attack_type: str,
    predicted_signals: List[str],
    scenario: Dict[str, Any],
) -> Tuple[float, Dict[str, float], str]:
    """
    Reward shaping:
      +0.4  correct signals identified (coverage)
      +0.6  correct attack classification (exact=0.6, partial=0.3)

    Total is capped at 1.0.
    """
    breakdown: Dict[str, float] = {}
    feedback_parts: List[str] = []

    # --- Signal coverage score ---
    key_signals = scenario["key_signals"]
    matched_signals = sum(
        1 for sig in key_signals
        if any(
            _normalize(sig) in _normalize(ps) or _normalize(ps) in _normalize(sig)
            for ps in predicted_signals
        )
    )
    signal_score = round((matched_signals / len(key_signals)) * 0.4, 4) if key_signals else 0.0
    breakdown["signal_coverage"] = signal_score
    feedback_parts.append(f"📡 Signal coverage: {matched_signals}/{len(key_signals)} key signals identified.")

    # --- Classification score ---
    pred_norm = _normalize(predicted_attack_type)
    true_norm = _normalize(scenario["true_attack_type"])

    if pred_norm == true_norm:
        class_score = 0.6
        feedback_parts.append("✅ Exact classification match!")
    elif any(_normalize(p) in pred_norm for p in scenario["accepted_partials"]):
        class_score = 0.3
        feedback_parts.append(f"⚠️ Partial match. Expected: '{scenario['true_attack_type']}'.")
    else:
        class_score = 0.0
        feedback_parts.append(f"❌ Wrong classification. Expected: '{scenario['true_attack_type']}'.")

    breakdown["classification"] = class_score

    total = round(min(max(sum(breakdown.values()), 0.001), 0.999), 4)
    return total, breakdown, " ".join(feedback_parts)


def get_scenario(index: int = 0) -> Dict[str, Any]:
    if not SCENARIOS:
        raise ValueError("No scenarios defined.")
    return SCENARIOS[index % len(SCENARIOS)]
