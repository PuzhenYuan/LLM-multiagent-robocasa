import re

text = "Plan: [step0, step1, step2, ...],  Execution: step0."

# 提取 plan
plan_match = re.search(r"Plan: \[(.*?)\]", text)
plan = plan_match.group(1).split(", ") if plan_match else []

# 提取 execution
execution_match = re.search(r"Execution: (\w+)", text)
execution = execution_match.group(1) if execution_match else ""

print("Plan:", plan)
print("Execution:", execution)

m = """
what
"""

print(', '.join(plan))

m += ("qwq")
print(type(m))
print(m)