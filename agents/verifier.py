import os
import subprocess
import sys
import tempfile

from config import REVIEW_MAX_TOKENS as _MAX_TOKENS
from config import SEED as _SEED
from config import TEMPERATURE as _TEMPERATURE
from config import TOP_P as _TOP_P
from config import VERIFIER_MODEL as _MODEL
from config import VERIFIER_TIMEOUT as _TIMEOUT
from config import client as _client

_SYSTEM = (
    "You are a code review agent. Given a Python function and its failing test output, "
    "identify the bug and explain concisely what went wrong and how to fix it."
)


def verify(solution_code: str, test_code: str, entry_point: str) -> dict:
    """Execute the solution against HumanEval test cases.

    Returns a dict with keys:
      passed  – bool
      stdout  – captured stdout
      stderr  – captured stderr / error message
      review  – Claude's diagnosis (only populated on failure)
    """
    full_source = f"{solution_code}\n\n{test_code}\n\ncheck({entry_point})\n"

    "creates a throwaway temp .py file with the solution + test code, runs it, captures output and errors, and deletes the file"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full_source)
        tmp_path = f.name

    try:
        proc = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT,
        )
        passed = proc.returncode == 0
        result = {
            "passed": passed,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "review": "",
        }

        if not passed:
            result["review"] = _review(solution_code, proc.stderr)

        return result

    except subprocess.TimeoutExpired:
        return {
            "passed": False,
            "stdout": "",
            "stderr": f"Execution timed out after {_TIMEOUT} s",
            "review": "",
        }
    finally:
        "deletes the temp file to avoid clutter"
        os.unlink(tmp_path)


def _review(solution_code: str, error_output: str) -> str:
    """Ask the model to diagnose why the solution failed."""
    response = _client.chat.completions.create(
        model=_MODEL,
        max_tokens=_MAX_TOKENS,
        temperature=_TEMPERATURE,
        top_p=_TOP_P,
        seed=_SEED,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Code:\n```python\n{solution_code}\n```\n\n"
                    f"Test error:\n{error_output}\n\n"
                    "What is wrong and how should it be fixed?"
                ),
            },
        ],
    )
    return response.choices[0].message.content.strip()