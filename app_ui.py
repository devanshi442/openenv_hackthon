"""
CyberDefend-X — Gradio Demo UI
Wraps the existing FastAPI backend via HTTP requests.

Usage:
  pip install gradio requests
  python app_ui.py

The UI calls /reset then /step on the running FastAPI server.
Default backend: http://localhost:7860
Override with ENV_BASE_URL environment variable.
"""

from __future__ import annotations

import json
import os

import gradio as gr
import requests

ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860").rstrip("/")

# ------------------------------------------------------------------
# Task metadata (mirrors server.py TASKS list)
# ------------------------------------------------------------------
TASK_META = {
    "alert_prioritization": {
        "label": "🟢 Task 1 — Alert Prioritization (Easy)",
        "max_scenarios": 3,
        "action_hint": (
            "Enter a JSON list of alert strings ordered most-to-least critical.\n\n"
            'Example:\n["Data exfiltration to unknown external IP (200MB)", '
            '"Admin login from unusual geographic location", '
            '"Port scan detected on DMZ server", '
            '"5 failed SSH logins in 10 minutes"]'
        ),
        "action_placeholder": '["most critical alert", "second", "third", "least critical"]',
    },
    "threat_detection": {
        "label": "🟡 Task 2 — Threat Detection (Medium)",
        "max_scenarios": 4,
        "action_hint": (
            'Enter a JSON object with "attack_type" and "signals" keys.\n\n'
            'Example:\n{"attack_type": "Ransomware", '
            '"signals": ["file encryption", "shadow copy deletion", "ransom note"]}'
        ),
        "action_placeholder": '{"attack_type": "...", "signals": ["signal1", "signal2"]}',
    },
    "incident_response": {
        "label": "🔴 Task 3 — Incident Response (Hard)",
        "max_scenarios": 2,
        "action_hint": (
            'Enter a JSON object with an "action" key (choose from valid actions).\n\n'
            "Valid actions: alert_soc, block_ip, isolate_system, kill_process,\n"
            "reset_credentials, collect_forensics, escalate_to_management,\n"
            "patch_system, restore_backup, do_nothing\n\n"
            'Example:\n{"action": "kill_process"}'
        ),
        "action_placeholder": '{"action": "isolate_system"}',
    },
}

TASK_IDS = list(TASK_META.keys())
TASK_LABELS = [v["label"] for v in TASK_META.values()]
LABEL_TO_ID = {v["label"]: k for k, v in TASK_META.items()}


# ------------------------------------------------------------------
# API helpers
# ------------------------------------------------------------------

def _post(path: str, payload: dict) -> dict:
    try:
        r = requests.post(f"{ENV_BASE_URL}{path}", json=payload, timeout=20)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        return {"error": f"Cannot connect to backend at {ENV_BASE_URL}. Is the server running?"}
    except Exception as e:
        return {"error": str(e)}


def _get(path: str, params: dict | None = None) -> dict:
    try:
        r = requests.get(f"{ENV_BASE_URL}{path}", params=params, timeout=20)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        return {"error": f"Cannot connect to backend at {ENV_BASE_URL}. Is the server running?"}
    except Exception as e:
        return {"error": str(e)}


def _fmt(data: dict | list) -> str:
    return json.dumps(data, indent=2)


# ------------------------------------------------------------------
# Core logic
# ------------------------------------------------------------------

def check_health() -> str:
    result = _get("/health")
    if "error" in result:
        return f"❌  {result['error']}"
    return f"✅  Backend healthy — {result}"


def update_hints(task_label: str) -> tuple[str, str, str]:
    """Return (scenario max label, action hint, action placeholder) for the task."""
    tid = LABEL_TO_ID[task_label]
    meta = TASK_META[tid]
    max_s = meta["max_scenarios"]
    return (
        f"Scenario index (0 – {max_s - 1})",
        meta["action_hint"],
        meta["action_placeholder"],
    )


def run_simulation(
    task_label: str,
    scenario_index: int,
    action_json_str: str,
    reason_text: str,
) -> tuple[str, str, str, str]:
    """
    1. POST /reset
    2. Parse action input
    3. POST /step
    Returns (reset_output, step_output, reward_display, feedback_display)
    """
    tid = LABEL_TO_ID[task_label]
    si = int(scenario_index)

    # --- Step 1: Reset ---
    reset_result = _post("/reset", {"task_id": tid, "scenario_index": si})
    if "error" in reset_result:
        return _fmt(reset_result), "—", "—", reset_result["error"]

    reset_display = _fmt(reset_result)

    # --- Parse action input ---
    action_str = action_json_str.strip()
    if not action_str:
        return reset_display, "—", "—", "⚠️  Please enter an action before clicking Run."

    try:
        action_payload = json.loads(action_str)
    except json.JSONDecodeError as e:
        return reset_display, "—", "—", f"❌  Invalid JSON in action field: {e}"

    if not reason_text.strip():
        return reset_display, "—", "—", "⚠️  Please provide a reason (required for explainability bonus)."

    # --- Step 2: Step ---
    step_result = _post("/step", {
        "task_id": tid,
        "scenario_index": si,
        "action": action_payload,
        "reason": reason_text.strip(),
    })

    if "error" in step_result:
        return reset_display, _fmt(step_result), "—", step_result["error"]

    reward = step_result.get("reward", 0.0)
    info = step_result.get("info", {})
    feedback = info.get("feedback", "No feedback.")
    breakdown = info.get("breakdown", {})

    reward_str = f"**{reward:.4f}** / 1.0\n\nBreakdown:\n"
    for k, v in breakdown.items():
        reward_str += f"- `{k}`: {v}\n"

    return reset_display, _fmt(step_result), reward_str, feedback


def get_current_state(task_label: str, scenario_index: int) -> str:
    tid = LABEL_TO_ID[task_label]
    result = _get("/state", {"task_id": tid, "scenario_index": int(scenario_index)})
    return _fmt(result)


def list_all_tasks() -> str:
    result = _get("/tasks")
    return _fmt(result)


# ------------------------------------------------------------------
# Gradio UI layout
# ------------------------------------------------------------------

DESCRIPTION = """
# 🛡️ CyberDefend-X: SOC Decision Intelligence Environment

An **OpenEnv-compliant** reinforcement learning environment where an AI agent acts as a
**Security Operations Center (SOC) analyst**. Select a task, enter your action as JSON,
and see how the grader scores your decision.

> **Backend URL:** `{env_url}`
""".format(env_url=ENV_BASE_URL)

with gr.Blocks(
    title="CyberDefend-X",
    theme=gr.themes.Base(
        primary_hue="red",
        secondary_hue="gray",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
    ),
    css="""
    .score-box { font-size: 1.1em; }
    .feedback-box { font-size: 1.0em; }
    footer { display: none !important; }
    """,
) as demo:

    gr.Markdown(DESCRIPTION)

    with gr.Row():
        health_btn = gr.Button("🔍 Check Backend Health", variant="secondary", scale=1)
        health_out = gr.Textbox(label="Health Status", interactive=False, scale=3)
    health_btn.click(fn=check_health, outputs=health_out)

    gr.Markdown("---")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## ⚙️ Configuration")

            task_dropdown = gr.Dropdown(
                choices=TASK_LABELS,
                value=TASK_LABELS[0],
                label="Select Task",
                interactive=True,
            )

            scenario_slider = gr.Slider(
                minimum=0, maximum=2, step=1, value=0,
                label="Scenario Index (0 – 2)",
                interactive=True,
            )

            gr.Markdown("---")
            gr.Markdown("## 📝 Your Action")

            action_hint_box = gr.Textbox(
                value=TASK_META["alert_prioritization"]["action_hint"],
                label="Action Format Guide",
                lines=5,
                interactive=False,
            )

            action_input = gr.Textbox(
                value=TASK_META["alert_prioritization"]["action_placeholder"],
                label="Action (JSON)",
                lines=5,
                placeholder='Enter JSON action here...',
                interactive=True,
            )

            reason_input = gr.Textbox(
                label="Reason / Explanation (required)",
                lines=2,
                placeholder="Explain why you chose this action...",
                interactive=True,
            )

            run_btn = gr.Button("🚀 Run Simulation", variant="primary")
            state_btn = gr.Button("🔭 View Current State", variant="secondary")
            tasks_btn = gr.Button("📋 List All Tasks", variant="secondary")

        with gr.Column(scale=2):
            gr.Markdown("## 📊 Results")

            with gr.Row():
                reward_display = gr.Markdown(
                    value="*Run a simulation to see your score.*",
                    label="Reward",
                    elem_classes=["score-box"],
                )

            feedback_display = gr.Textbox(
                label="📣 Grader Feedback",
                lines=3,
                interactive=False,
                elem_classes=["feedback-box"],
            )

            gr.Markdown("### 🔄 Reset Response (Observation)")
            reset_output = gr.Code(
                label="POST /reset → initial observation",
                language="json",
                lines=12,
            )

            gr.Markdown("### 🎯 Step Response (Full Result)")
            step_output = gr.Code(
                label="POST /step → observation + reward + info",
                language="json",
                lines=20,
            )

            gr.Markdown("### 🔭 State / Tasks")
            state_output = gr.Code(
                label="GET /state or /tasks",
                language="json",
                lines=10,
            )

    # ------------------------------------------------------------------
    # Event wiring
    # ------------------------------------------------------------------

    def on_task_change(task_label):
        label, hint, placeholder = update_hints(task_label)
        tid = LABEL_TO_ID[task_label]
        max_s = TASK_META[tid]["max_scenarios"] - 1
        return (
            gr.update(label=label, maximum=max_s, value=0),
            hint,
            placeholder,
        )

    task_dropdown.change(
        fn=on_task_change,
        inputs=task_dropdown,
        outputs=[scenario_slider, action_hint_box, action_input],
    )

    run_btn.click(
        fn=run_simulation,
        inputs=[task_dropdown, scenario_slider, action_input, reason_input],
        outputs=[reset_output, step_output, reward_display, feedback_display],
    )

    state_btn.click(
        fn=get_current_state,
        inputs=[task_dropdown, scenario_slider],
        outputs=state_output,
    )

    tasks_btn.click(
        fn=list_all_tasks,
        inputs=[],
        outputs=state_output,
    )

    gr.Markdown(
        """
---
*Built for the OpenEnv Hackathon — Scaler × Hugging Face.*
*Backend: FastAPI + Uvicorn | Graders: Kendall Tau, signal coverage, weighted step scoring*
        """
    )

# ------------------------------------------------------------------
# Launch
# ------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("UI_PORT", 7861))
    print(f"🚀 Launching Gradio UI on http://0.0.0.0:{port}")
    print(f"   Backend: {ENV_BASE_URL}")
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)