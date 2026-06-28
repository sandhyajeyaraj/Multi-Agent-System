import ast
import re

import config
from config import CODE_MAX_TOKENS as _MAX_TOKENS
from config import SEED as _SEED
from config import TEMPERATURE as _TEMPERATURE
from config import TOP_P as _TOP_P
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

_RECOVERY_SYSTEM = """You are a debugging agent. Your previous solution FAILED a test.
Use ReAct to diagnose and fix it. Follow this exact format:

Thought: <restate what the code was supposed to do>
Observation: The failing test was: <FAILING_INPUT> → expected <EXPECTED>, got <ACTUAL>
Thought: <trace your previous code line by line ON THIS SPECIFIC INPUT, find where it diverges>
Action: Locate the exact line/branch causing the wrong output
Observation: <name the bug: e.g. "the loop skips the last element">
Thought: <state the minimal fix>
Action: Implement
Observation:
```python
<complete corrected solution>
```

You will be given: the problem, your previous (failing) code, and the failing test case.
Trace the failing input through your old code BEFORE rewriting. Output only the ReAct trace."""

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
    if error_context:
        system = _RECOVERY_SYSTEM
        user_content = (
            f"Problem:\n{problem_prompt}\n\n"
            f"Previous (failing) code:\n{plan}\n\n"
            f"Failing test output:\n{error_context}"
        )
    else:
        system = _SYSTEM
        user_content = (
            f"Problem:\n{problem_prompt}\n\n"
            f"Plan:\n{plan}\n\n"
            "Implement the solution:"
        )
    response = _client.chat.completions.create(
        model=config.CODER_MODEL,
        max_tokens=_MAX_TOKENS,
        temperature=_TEMPERATURE,
        top_p=_TOP_P,
        seed=_SEED,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
    )
    return _extract_code(response.choices[0].message.content)