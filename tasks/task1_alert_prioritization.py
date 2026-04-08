"""
Task 1: Alert Prioritization (Easy)
Objective: Rank incoming alerts by true severity score.
Grader: Kendall Tau correlation — deterministic, 0.0–1.0.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


# ---------------------------------------------------------------------------
# Dataset — static scenarios for reproducibility
# ---------------------------------------------------------------------------

SCENARIOS: List[Dict[str, Any]] = [
    {
        "id": "T1-001",
        "alerts": [
            {"alert": "5 failed SSH logins in 10 minutes", "true_severity": 0.3},
            {"alert": "Data exfiltration to unknown external IP (200MB)", "true_severity": 0.95},
            {"alert": "Admin login from unusual geographic location", "true_severity": 0.7},
            {"alert": "Port scan detected on DMZ server", "true_severity": 0.5},
        ],
        "true_ranking": [
            "Data exfiltration to unknown external IP (200MB)",
            "Admin login from unusual geographic location",
            "Port scan detected on DMZ server",
            "5 failed SSH logins in 10 minutes",
        ],
    },
    {
        "id": "T1-002",
        "alerts": [
            {"alert": "Malware signature detected on endpoint", "true_severity": 0.85},
            {"alert": "User accessed HR system outside business hours", "true_severity": 0.4},
            {"alert": "Ransomware behavior: mass file encryption started", "true_severity": 1.0},
            {"alert": "Outdated antivirus definitions on 3 hosts", "true_severity": 0.2},
        ],
        "true_ranking": [
            "Ransomware behavior: mass file encryption started",
            "Malware signature detected on endpoint",
            "User accessed HR system outside business hours",
            "Outdated antivirus definitions on 3 hosts",
        ],
    },
    {
        "id": "T1-003",
        "alerts": [
            {"alert": "Suspicious PowerShell execution with encoded command", "true_severity": 0.8},
            {"alert": "DNS query to known C2 domain", "true_severity": 0.9},
            {"alert": "Failed VPN login x3", "true_severity": 0.25},
            {"alert": "Lateral movement: SMB scan across internal subnets", "true_severity": 0.75},
        ],
        "true_ranking": [
            "DNS query to known C2 domain",
            "Suspicious PowerShell execution with encoded command",
            "Lateral movement: SMB scan across internal subnets",
            "Failed VPN login x3",
        ],
    },
]


def kendall_tau_score(predicted: List[str], true_ranking: List[str]) -> float:
    """
    Normalized Kendall Tau distance — 1.0 is perfect, 0.0 is fully reversed.
    Only scores alerts that appear in both lists.
    """
    common = [a for a in true_ranking if a in predicted]
    if len(common) < 2:
        return 0.0

    # Map each alert in common to its position in the predicted list
    pred_order = [predicted.index(a) for a in common]
    n = len(common)
    concordant = 0
    discordant = 0
    pairs = n * (n - 1) // 2

    for i in range(n):
        for j in range(i + 1, n):
            # true_order is implicitly i < j (common is ordered by true_ranking)
            if pred_order[i] < pred_order[j]:
                concordant += 1
            else:
                discordant += 1

    if pairs == 0:
        return 1.0
    tau = (concordant - discordant) / pairs
    return round((tau + 1) / 2, 4)  # normalize [-1, 1] → [0, 1]


def compute_reward(
    predicted_ranking: List[str],
    scenario: Dict[str, Any],
) -> Tuple[float, Dict[str, float], str]:
    """
    Reward shaping:
      +0.3  correct top alert identified
      +0.3  good ordering (Kendall Tau >= 0.8), else proportional partial credit
      +0.4  perfect ranking (Kendall Tau == 1.0)
    """
    true_ranking = scenario["true_ranking"]
    tau = kendall_tau_score(predicted_ranking, true_ranking)

    breakdown: Dict[str, float] = {}
    feedback_parts: List[str] = []

    # +0.3 correct top alert
    top_correct = bool(predicted_ranking) and predicted_ranking[0] == true_ranking[0]
    breakdown["correct_top_alert"] = 0.3 if top_correct else 0.0
    if top_correct:
        feedback_parts.append("✅ Correctly identified most critical alert.")
    else:
        top_expected = true_ranking[0] if true_ranking else "N/A"
        feedback_parts.append(f"❌ Top alert wrong. Expected: '{top_expected}'.")

    # +0.3 good ordering (tau >= 0.8), else proportional partial credit
    breakdown["good_ordering"] = 0.3 if tau >= 0.8 else round(tau * 0.3, 4)
    feedback_parts.append(f"📊 Kendall Tau score: {tau:.2f}.")

    # +0.4 perfect ranking
    breakdown["perfect_ranking"] = 0.4 if tau == 1.0 else 0.0
    if tau == 1.0:
        feedback_parts.append("🏆 Perfect ranking!")

    total = round(min(max(sum(breakdown.values()), 0.001), 0.999), 4)
    return total, breakdown, " ".join(feedback_parts)


def get_scenario(index: int = 0) -> Dict[str, Any]:
    if not SCENARIOS:
        raise ValueError("No scenarios defined.")
    return SCENARIOS[index % len(SCENARIOS)]
