content = open('inference.py', 'r', encoding='utf-8').read()

# Fix: add reason to the step request
content = content.replace(
    'r = httpx.post(f"{ENV_BASE_URL}/step",\n                           json={"task_id": task_id, "scenario_index": 0, "action": action},',
    'r = httpx.post(f"{ENV_BASE_URL}/step",\n                           json={"task_id": task_id, "scenario_index": 0, "action": action, "reason": "Based on current log analysis, this action addresses the immediate threat by containing and stopping the attack process"},',
)

open('inference.py', 'w', encoding='utf-8').write(content)
print('DONE')
