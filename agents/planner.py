from config import MODEL as _MODEL
from config import client as _client

_SYSTEM = """You are a planning agent that uses ReAct (Reasoning + Acting) to plan Python solutions.

For every problem follow this exact format:

Thought: <analyse the problem — what it asks, constraints, tricky cases>
Action: Choose algorithm
Observation: <why this algorithm fits, time/space complexity>
Thought: <identify all edge cases that must be handled>
Action: List edge cases
Observation: <enumerate each edge case and how to handle it>
Thought: <outline the step-by-step logic>
Action: Write plan
Observation:
1. <step 1>
2. <step 2>
...

Output only the ReAct trace above — no code."""


def plan(problem_prompt: str) -> str:
    """Return a ReAct-style solution plan for the given HumanEval problem."""
    response = _client.chat.completions.create(
        model=_MODEL,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": f"Plan a solution for this problem:\n\n{problem_prompt}"},
        ],
    )
    return response.choices[0].message.content.strip()