"""
Trace Analyzer — Phase 2 repair path.

The Coder always instruments its solutions with debug prints (after every
loop variable update and before every return), so a failing run leaves a
captured stdout trace behind for free — no re-instrumentation or second run
needed. When the classifier attributes a failure to the Coder (step 2),
this reads that trace against the failing test and produces a concrete
diagnosis of exactly where the code's runtime values diverged from what was
expected, so the Recovery Coder can make a targeted fix instead of a blind
rewrite.
"""

from __future__ import annotations

from config import ANALYZER_MODEL as _MODEL
from config import SEED as _SEED
from config import TEMPERATURE as _TEMPERATURE
from config import TOP_P as _TOP_P
from config import TRACE_MAX_TOKENS as _MAX_TOKENS
from config import client as _client

_SYSTEM = """You are a trace analysis agent. A Python solution failed a test.

You are given the captured runtime trace — stdout produced by debug prints
the code already emits after every loop variable update and before every
return — from the failing run, alongside the code, the test, and the error.

Diagnose the ROOT CAUSE of the failure, not just the symptom. The repair
agent needs to know not only WHAT went wrong but WHY and WHERE to fix it.

Follow these steps:
1. Find where the trace diverged from expected (which loop iteration,
   which return point).
2. State the actual value (quote it from the trace) vs. the expected value.
3. Trace BACKWARD through the code to the SOURCE of that divergence —
   the specific line, condition, loop bound, or expression that produced
   the wrong value.
4. State the minimal fix concretely.

The difference between a bad and a good diagnosis:
  BAD:  "At return, result was [0,1,2,3] but should have been [0,1,2,3,4]
         because the last element is missing."
  GOOD: "At return, result was [0,1,2,3] but should have been [0,1,2,3,4]
         because the loop uses range(n), which stops at n-1.
         Fix: change range(n) to range(n+1)."

The bad one restates the symptom. The good one names the guilty line and
the exact change. Always produce the second kind.

Respond in 3-5 sentences. Do not suggest a full rewrite.

If the trace contains no useful signal (e.g. the failure happened before
any prints ran, or the code wasn't instrumented), say so explicitly and
diagnose from the code, test, and error instead — still tracing to the
root cause and naming the fix."""


def analyze_trace(
    problem_prompt: str,
    solution_code: str,
    test_code: str,
    trace: str,
    error: str,
) -> str:
    """Return a concrete diagnosis of where the Coder's logic diverged."""
    user_content = (
        f"Problem:\n{problem_prompt}\n\n"
        f"Code (with debug prints):\n{solution_code}\n\n"
        f"Test:\n{test_code}\n\n"
        f"Captured trace (stdout from the failing run):\n"
        f"{trace.strip() if trace and trace.strip() else '(empty — no prints captured before failure)'}\n\n"
        f"Test error:\n{error}"
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
