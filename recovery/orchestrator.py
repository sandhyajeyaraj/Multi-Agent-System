"""
Recovery Orchestrator — re-runs the pipeline from the classified failing step.

Routing rules:
  step 1  →  rerun Planner → Coder → Verifier  (plan was bad)
  step 2  →  reuse plan,  rerun Coder → Verifier  (code was bad)
  step 3  →  reuse plan + code,  rerun Verifier only  (execution glitch)
"""

from __future__ import annotations

from agents.coder import code
from agents.planner import plan
from agents.verifier import verify
from recovery.classifier import classify_failure


class RecoveryOrchestrator:
    def recover(
        self,
        problem: dict,
        plan_text: str,
        solution_code: str,
        error: str,
        review: str,
    ) -> dict:
        failing_step, reason = classify_failure(
            problem["prompt"], plan_text, solution_code, error, review
        )
        print(f"\n[RECOVERY] Classifier → step {failing_step} failed: {reason}")

        if failing_step == 1:
            print("[RECOVERY] Rerunning from step 1: Planner → Coder → Verifier")
            plan_text = plan(problem["prompt"], error_context=reason)
            print(f"\n{plan_text}\n")
            solution_code = code(problem["prompt"], plan_text)
            print(f"\n{solution_code}\n")

        elif failing_step == 2:
            print("[RECOVERY] Rerunning from step 2: Coder → Verifier")
            solution_code = code(problem["prompt"], plan_text, error_context=error)
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
            "plan": plan_text,
            "code": solution_code,
            "passed": result["passed"],
            "error": result["stderr"],
            "review": result["review"],
        }
