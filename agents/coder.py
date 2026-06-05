import re

from config import MODEL as _MODEL
from config import client as _client

_SYSTEM = """You are a coding agent that uses ReAct (Reasoning + Acting) to implement Python solutions.

For every problem follow this exact format:

Thought: <re-read the plan and confirm your understanding of what to implement>
Action: Identify inputs, outputs, and types
Observation: <state input types, output type, and any imports needed>
Thought: <walk through the algorithm step by step before writing code>
Action: Implement
Observation:
<raw Python function here — no markdown fences, no extra explanation>

Output only the ReAct trace above. The final Observation must contain ONLY the raw Python code."""


def _extract_code(text: str) -> str:
    # Try markdown fences first
    match = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fall back to everything after the last "Observation:" line
    parts = re.split(r"(?i)^Observation:", text, flags=re.MULTILINE)
    if len(parts) > 1:
        return parts[-1].strip()
    return text.strip()


def code(problem_prompt: str, plan: str) -> str:
    """Return a Python implementation based on the problem prompt and plan."""
    response = _client.chat.completions.create(
        model=_MODEL,
        max_tokens=2048,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Problem:\n{problem_prompt}\n\n"
                    f"Plan:\n{plan}\n\n"
                    "Implement the solution:"
                ),
            },
        ],
    )
    return _extract_code(response.choices[0].message.content)