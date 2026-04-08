content = open('inference.py', 'r', encoding='utf-8').read()

content = content.replace(
    'You are an incident responder. Choose the best next IR action.',
    'You are an expert incident responder. Read the current log and pick the best action.\n\nDecision guide:\n- Active malware/ransomware process running: kill_process\n- Lateral spread detected, multiple hosts infected: isolate_system\n- C2 domain resolving, attacker may re-enter: block_ip\n- Compromised credentials, account takeover: reset_credentials\n- Need to alert team about new threat: alert_soc\n- Incident contained, need evidence or documentation: collect_forensics\n- Need stakeholder notification after containment: escalate_to_management\n- System needs patching: patch_system\n\nReturn ONLY valid JSON: {"action": "action_name"}\nNo markdown. No explanation. Just JSON.'
)

content = content.replace(
    'Actions: kill_process, isolate_system, block_ip, collect_forensics, escalate_to_management, alert_soc, patch_vulnerability, restore_system',
    'Valid actions: alert_soc, block_ip, isolate_system, kill_process, reset_credentials, collect_forensics, escalate_to_management, patch_system, restore_backup, do_nothing'
)

content = content.replace(
    'Optimal sequence: kill_process \u2192 isolate_system \u2192 block_ip \u2192 escalate_to_management \u2192 restore_system',
    ''
)

content = content.replace(
    'user = f"""Step {step_num+1}. Context: {json.dumps(context)}. Recent logs: {json.dumps(logs[-3:] if logs else [])}.\\n\\nBest next action? Return JSON with key \\"action\\"."""',
    'current_log = logs[-1] if logs else "No log"\n        user = f"Step {step_num+1}. Log: {current_log}. Pick best action. Return JSON with key action."'
)

open('inference.py', 'w', encoding='utf-8').write(content)
print('DONE')
