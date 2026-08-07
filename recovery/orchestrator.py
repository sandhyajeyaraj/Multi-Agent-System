"""
Recovery Orchestrator — re-runs the pipeline from the classified failing step.

Routing rules:
  step 1  →  rerun Planner → Coder → Verifier  (plan was bad)
  step 2  →  analyze the captured debug-print trace, then
             rerun Coder(diagnosis) → Verifier  (code was bad)
  step 3  →  reuse plan + code,  rerun Verifier only  (execution glitch)
"""

from __future__ import annotations

import config
from agents.coder import code
from agents.planner import plan
from agents.verifier import verify
from recovery.classifier import classify_failure
from recovery.trace_analyzer import analyze_trace


def _code_with_escalation(problem_prompt: str, plan_text: str, error_context: str = "") -> str:
    """Run the coder with RECOVERY_CODER_MODEL, restoring the original model afterwards."""
    original = config.CODER_MODEL
    config.CODER_MODEL = config.RECOVERY_CODER_MODEL
    try:
        return code(problem_prompt, plan_text, error_context=error_context)
    finally:
        config.CODER_MODEL = original


class RecoveryOrchestrator:
    def recover(
        self,
        problem: dict,
        plan_text: str,
        solution_code: str,
        error: str,
        review: str,
        trace: str = "",
    ) -> dict:
        failing_step, reason = classify_failure(
            problem["prompt"], plan_text, solution_code, error, review
        )
        print(f"\n[RECOVERY] Classifier → step {failing_step} failed: {reason}")

        diagnosis = None

        if failing_step == 1:
            print(f"[RECOVERY] Rerunning from step 1: Planner → Coder({config.RECOVERY_CODER_MODEL}) → Verifier")
            plan_text = plan(problem["prompt"], error_context=reason)
            print(f"\n{plan_text}\n")
            solution_code = _code_with_escalation(problem["prompt"], plan_text)
            print(f"\n{solution_code}\n")

        elif failing_step == 2:
            if config.SKIP_TRACE_ANALYZER:
                print("[RECOVERY] Coder is root cause — skipping analyzer, sending raw trace + assertion directly to Coder")
                diagnosis = (
                    f"Problem:\n{problem['prompt']}\n\n"
                    f"Code (with debug prints):\n{solution_code}\n\n"
                    f"Test:\n{problem['test']}\n\n"
                    f"Captured trace (stdout from the failing run):\n"
                    f"{trace.strip() if trace and trace.strip() else '(empty — no prints captured before failure)'}\n\n"
                    f"Test error:\n{error}"
                )
            else:
                print("[RECOVERY] Coder is root cause — analyzing captured debug-print trace...")
                diagnosis = analyze_trace(
                    problem["prompt"], solution_code, problem["test"], trace, error
                )
                print(f"\n[RECOVERY] Diagnosis: {diagnosis}\n")
            print(f"[RECOVERY] Rerunning from step 2: Coder({config.RECOVERY_CODER_MODEL}) → Verifier")
            solution_code = _code_with_escalation(
                problem["prompt"], solution_code,
                error_context=diagnosis,
            )
            print(f"\n{solution_code}\n")

        else:
            print("[RECOVERY] Rerunning step 3: Verifier only")

        result = verify(solution_code, problem["test"], problem["entry_point"])
        status = "PASS ✓" if result["passed"] else "FAIL ✗"
        print(f"[RECOVERY] Result: {status}")
        if not result["passed"]:
            print(f"\n--- Recovery Error ---\n{result['stderr']}")

        return {
            "failing_step": failing_step,
            "reason": reason,
            "diagnosis": diagnosis,
            "plan": plan_text,
            "code": solution_code,
            "passed": result["passed"],
            "error": result["stderr"],
            "review": result["review"],
        }
