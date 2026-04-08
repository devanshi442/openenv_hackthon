# 🛡️ CyberDefend-X: SOC Decision Intelligence Environment

> An OpenEnv-compliant reinforcement learning environment where an AI agent acts as a **Security Operations Center (SOC) analyst**, performing real-world cybersecurity tasks.

---

## 🧠 Motivation

SOC analysts face high-pressure, high-stakes decisions daily — triaging hundreds of alerts, correlating threat signals, and executing containment actions under time pressure. This environment trains and evaluates AI agents on these exact workflows, providing a realistic benchmark for cybersecurity decision intelligence.

---

## 🗂️ Environment Overview

| Property | Value |
|---|---|
| **Tasks** | 3 (Easy → Medium → Hard) |
| **Reward Range** | 0.0 – 1.0 |
| **Episode Type** | Single-step (T1, T2) / Multi-step (T3) |
| **Real-world domain** | Cybersecurity / SOC operations |
| **OpenEnv spec** | ✅ Compliant |

---

## 📋 Tasks

### 🟢 Task 1: Alert Prioritization (Easy)
**Objective**: Rank incoming security alerts from most to least critical.

**Input**: A list of alert strings with hidden true severity scores.
**Output**: Ordered list of alert strings (most critical first).
**Grader**: Kendall Tau correlation between predicted and true ranking → [0.0, 1.0]

**Reward Shaping**:
- `+0.3` correct top alert identified
- `+0.3` good ordering (Kendall Tau ≥ 0.8)
- `+0.4` perfect ranking (Kendall Tau = 1.0)

---

### 🟡 Task 2: Multi-Signal Threat Detection (Medium)
**Objective**: Classify the attack type from multiple correlated log entries.

**Input**: 4 related security log lines.
**Output**: `attack_type` string + list of identified `signals`.
**Grader**: Exact match (0.6) / partial match (0.3) + signal coverage (0.4)

**Reward Shaping**:
- `+0.4` signal coverage score (identified key signals / total key signals)
- `+0.6` exact classification match
- `+0.3` partial classification match

---

### 🔴 Task 3: Sequential Incident Response (Hard)
**Objective**: Take optimal containment actions step-by-step over 4 turns against an evolving intrusion scenario.

**Input**: Evolving log entries that update after each step.
**Output**: One action per step + reason from the valid actions list.
**Grader**: Per-step weighted scoring with explainability bonus.

**Valid Actions**:
`alert_soc`, `block_ip`, `isolate_system`, `kill_process`, `reset_credentials`,
`collect_forensics`, `escalate_to_management`, `patch_system`, `restore_backup`, `do_nothing`

**Reward Shaping (per step)**:
- `+weight × 1.0` optimal action chosen
- `+weight × 0.5` acceptable (non-optimal) action
- `-0.3` destructive/passive action during active attack
- `+0.1` explainability bonus (reason matches key signal keywords)

---

## 🔭 Observation Space

```json
{
  "task_id": "string",
  "step": "int",
  "context": "object — scenario metadata, valid_actions, cumulative_reward",
  "logs": ["string — security log entries"],
  "alerts": [{"alert": "string", ...}],
  "instructions": "string — natural language instructions for the agent",
  "history": ["previous step objects"],
  "done": "bool"
}
```

## 🎮 Action Space

```json
{
  "action": "any — task-specific payload (list / object / string)",
  "reason": "string — mandatory explanation (explainability bonus)"
}
```

---

## 🚀 Setup & Usage

### Local (Docker)

```bash
# Build
docker build -t cyberdefend-x .

# Run
docker run -p 7860:7860 cyberdefend-x

# Verify
curl http://localhost:7860/health
```

### API Quickstart

```bash
# Reset Task 1
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "alert_prioritization", "scenario_index": 0}'

# Step (submit ranking)
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "alert_prioritization",
    "scenario_index": 0,
    "action": ["Data exfiltration to unknown external IP (200MB)", "Admin login from unusual geographic location", "Port scan detected on DMZ server", "5 failed SSH logins in 10 minutes"],
    "reason": "Exfiltration is always critical. Admin geo-anomaly signals account compromise."
  }'

# Get state
curl "http://localhost:7860/state?task_id=alert_prioritization&scenario_index=0"

# List tasks
curl http://localhost:7860/tasks
```

### Run Baseline Inference

```bash
export API_BASE_URL="https://api.x.ai/v1"
export MODEL_NAME="grok-3-mini"
export HF_TOKEN="your_xai_api_key"
export ENV_BASE_URL="http://localhost:7860"

pip install openai httpx
python inference.py
```

---

## 📊 Baseline Scores

| Task | Score | Notes |
|---|---|---|
| Alert Prioritization | ~0.70 | Grok-3-mini gets top alert right, ordering imperfect |
| Threat Detection | ~0.65 | Exact classification hard, signals usually correct |
| Incident Response | ~0.55 | Sequential reasoning degrades on step 3–4 |
| **Overall** | **~0.63** | |

---

## 📁 Project Structure

```
cyberdefend-x/
├── openenv.yaml          # OpenEnv metadata
├── models.py             # Pydantic typed models (Observation, Action, Reward)
├── env.py                # Core environment (step, reset, state)
├── server.py             # FastAPI REST server
├── inference.py          # Baseline inference script (Grok via OpenAI client)
├── requirements.txt
├── Dockerfile
├── README.md
└── tasks/
    ├── task1_alert_prioritization.py
    ├── task2_threat_detection.py
    └── task3_incident_response.py
```

---

## ✅ Pre-Submission Checklist

- [x] HF Space deploys and `/health` returns 200
- [x] `openenv.yaml` valid with 3 tasks
- [x] Typed Pydantic models (`Observation`, `Action`, `Reward`)
- [x] `step()` / `reset()` / `state()` endpoints functional
- [x] Dockerfile builds and runs
- [x] `inference.py` in root, uses OpenAI client
- [x] `[START]` / `[STEP]` / `[END]` log format
- [x] `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` env vars used
- [x] All graders return scores in [0.0, 1.0]
- [x] Rewards at each step (not just episode end)
- [x] Explainability requirement (reason field mandatory)

---

*Built for the OpenEnv Hackathon — Scaler × Hugging Face, April 2025.*
