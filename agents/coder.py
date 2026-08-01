import ast
import re

import config
from agents.instrument import instrument_source
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
Action: List the edge cases from the plan and the examples from the problem docstring
Observation: <each edge case / example input with its expected output>
Action: Implement
Observation:
```python
<your complete solution here>
```
Thought: <trace your solution by hand against ONE docstring example: state
the input, walk the key steps, state the output your code produces. If it
does not match the expected output, fix the code and show the corrected
block.>

Rules:
- Keep the exact function name and signature from the problem prompt.
- Include all imports your code needs at the top of the code block.
- The code block must contain only the solution — no test calls, no
  example usage, no prints.

Output only the ReAct trace above, ending with exactly one final ```python
fenced block containing the complete solution."""


_RECOVERY_SYSTEM = """You are a debugging agent. Your previous solution FAILED a test.
Use ReAct to diagnose and fix it. Follow this exact format:

Thought: <restate what the code was supposed to do>
Observation: <quote the failing evidence you were ACTUALLY given: the
diagnosis if present, otherwise the raw error/test output. Do NOT invent
inputs or values that were not given to you.>
Thought: <pinpoint the exact line/branch the diagnosis points to. If the
diagnosis names a fix (e.g. "change range(n) to range(n+1)"), that is
your fix — do not second-guess it.>
Action: State the minimal edit — which line(s) change and to what
Observation: <the specific edit, e.g. "line 4: range(n) → range(n+1)">
Thought: <verify the edit against the failing case mentally>
Action: Implement
Observation:
```python
<complete corrected solution>
```

Rules:
- Make ONLY the minimal targeted change. Copy every other line of the
  previous solution unchanged. Do NOT refactor, rename, or restructure.
- If the previous code contains debug artifacts (print(...) lines or
  variables named __dbg_ret_N), these were auto-injected for tracing.
  They are NOT part of the solution — strip them all from your output
  and do not treat them as the bug.
- When a diagnosis is given, trust its concrete values — they reflect
  what the code actually did. If it prescribes a specific fix, apply
  exactly that fix.
- When NO diagnosis is given, reason from the code and error alone, and
  say so in your first Observation instead of inventing test values.

Output only the ReAct trace."""

_FENCE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)
_CODE_START = re.compile(r"^\s*(def |class |import |from |@)", re.MULTILINE)
_DEF_LINE = re.compile(r"^\s*def\s+\w+\s*\(", re.MULTILINE)


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


def _ensure_imports(solution_code: str, problem_prompt: str) -> str:
    """Prepend any import/setup lines HumanEval puts before the function
    signature (e.g. `from typing import List`) if the model's generated
    code dropped them."""
    match = _DEF_LINE.search(problem_prompt)
    if not match:
        return solution_code
    header_lines = [
        line for line in problem_prompt[: match.start()].splitlines() if line.strip()
    ]
    missing = [line for line in header_lines if line.strip() not in solution_code]
    if not missing:
        return solution_code
    return "\n".join(missing) + "\n\n" + solution_code


def code(problem_prompt: str, plan: str, error_context: str = "") -> str:
    """Return a Python implementation based on the problem prompt and plan."""
    if error_context:
        system = _RECOVERY_SYSTEM
        user_content = (
            f"Problem:\n{problem_prompt}\n\n"
            f"Previous (failing) code:\n{plan}\n\n"
            f"Diagnosis / failing test output:\n{error_context}"
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
    extracted = _extract_code(response.choices[0].message.content)
    with_imports = _ensure_imports(extracted, problem_prompt)
    return instrument_source(with_imports)