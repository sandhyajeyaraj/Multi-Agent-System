from config import PLAN_MAX_TOKENS as _MAX_TOKENS
from config import PLANNER_MODEL as _MODEL
from config import SEED as _SEED
from config import TEMPERATURE as _TEMPERATURE
from config import TOP_P as _TOP_P
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


def plan(problem_prompt: str, error_context: str = "") -> str:
    """Return a ReAct-style solution plan for the given HumanEval problem."""
    user_content = f"Plan a solution for this problem:\n\n{problem_prompt}"
    if error_context:
        user_content += (
            f"\n\nA previous plan was ineffective. Here's why it failed:\n{error_context}\n"
            "Create a better plan:"
        )
    response = _client.chat.completions.create(
        model=_MODEL,
        max_tokens=_MAX_TOKENS,
        temperature=_TEMPERATURE,
        top_p=_TOP_P,
        seed=_SEED,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user_content},
        ],
    )
    return response.choices[0].message.content.strip()