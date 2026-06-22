"""
Failure classifier — determines which pipeline step is the root cause.

Returns (failing_step, reason):
  1  →  Planner produced a wrong algorithm or missed key constraints.
  2  →  Plan was sound but coder introduced implementation bugs.
  3  →  Code logic appears correct; failure is execution-level only
       (timeout, environment, test-harness issue).
"""

from __future__ import annotations

import json
import re

from config import ANALYZER_MODEL as _MODEL
from config import client as _client

_SYSTEM = """You are a diagnostic agent for a multi-agent coding pipeline (Planner → Coder → Verifier).

Given a failed pipeline run, classify which step's output is the ROOT CAUSE of the failure.

Step 1 — Planner: Chose the algorithm and outlined the approach.
Step 2 — Coder:   Wrote Python code from the plan.
Step 3 — Verifier: Ran tests; only classify here for execution-level failures
                   (timeout, import error, test-harness bug) when the code logic itself is correct.

RULES:
- Prefer the earliest failing step.
- Return step 1 when the plan chose a wrong algorithm or ignored constraints.
- Return step 2 when the plan is correct but the code has logic/implementation bugs.
- Return step 3 ONLY when code logic looks right but the test failed for environmental reasons.

Respond with ONLY valid JSON, nothing else:
{"failing_step": <1|2|3>, "reason": "<one sentence>"}"""


def classify_failure(
    problem_prompt: str,
    plan_text: str,
    solution_code: str,
    error: str,
    review: str,
) -> tuple[int, str]:
    """Return (failing_step, reason). Defaults to step 2 on parse failure."""
    user_content = (
        f"PROBLEM:\n{problem_prompt}\n\n"
        f"PLAN (Planner output):\n{plan_text}\n\n"
        f"CODE (Coder output):\n{solution_code}\n\n"
        f"TEST ERROR:\n{error}\n\n"
        f"VERIFIER REVIEW:\n{review or '(none)'}"
    )
    response = _client.chat.completions.create(
        model=_MODEL,
        max_tokens=256,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user_content},
        ],
    )
    raw = response.choices[0].message.content.strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            step = int(data.get("failing_step", 2))
            if step not in (1, 2, 3):
                step = 2
            return step, data.get("reason", "")
        except (json.JSONDecodeError, ValueError):
            pass
    return 2, "classifier parse error — defaulting to coder"
