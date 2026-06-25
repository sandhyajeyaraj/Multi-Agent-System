import ast
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
```python
<your complete solution here>
```

Output only the ReAct trace above. The final Observation must contain exactly one ```python fenced block with the complete solution and nothing else."""

_FENCE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)
_CODE_START = re.compile(r"^\s*(def |class |import |from |@)", re.MULTILINE)


def _try_parse(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def _trim_to_valid(code: str) -> str | None:
    """Drop trailing lines until the snippet parses; return None if never valid."""
    code = code.rstrip()
    if _try_parse(code):
        return code
    lines = code.splitlines()
    for end in range(len(lines) - 1, 0, -1):
        snippet = "\n".join(lines[:end]).rstrip()
        if snippet and _try_parse(snippet):
            return snippet
    return None


def _extract_code(text: str) -> str:
    # 1. Prefer fenced blocks; return the first that parses
    for m in _FENCE.finditer(text):
        result = _trim_to_valid(m.group(1).strip())
        if result:
            return result

    # 2. Scan for code-like regions; retry on each until one parses
    for m in _CODE_START.finditer(text):
        result = _trim_to_valid(text[m.start():])
        if result:
            return result

    return text.strip()


def code(problem_prompt: str, plan: str, error_context: str = "") -> str:
    """Return a Python implementation based on the problem prompt and plan."""
    user_content = (
        f"Problem:\n{problem_prompt}\n\n"
        f"Plan:\n{plan}\n\n"
        "Implement the solution:"
    )
    if error_context:
        user_content += (
            f"\n\nA previous attempt produced this test error — fix the bug:\n{error_context}"
        )
    response = _client.chat.completions.create(
        model=_MODEL,
        max_tokens=2048,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user_content},
        ],
    )
    return _extract_code(response.choices[0].message.content)