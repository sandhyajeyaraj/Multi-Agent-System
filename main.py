"""
Multi-Agent System: Planner → Coder → Verifier on HumanEval.

Usage:
    python main.py              # runs first 5 problems
    python main.py --n 20       # runs first 20 problems
    python main.py --id 42      # runs HumanEval/42 only
    python main.py --debug      # enable AgentDebug root cause analysis on failures
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
SUMMARY_DIR = BASE_DIR / "summary"

from agents.coder import code
from agents.planner import plan
from agents.verifier import verify
from data.humaneval_loader import load_humaneval

# Lazy-import debugger so the system still works without it being invoked
_debugger = None


def _get_debugger():
    global _debugger
    if _debugger is None:
        from debugger import AgentDebugger
        _debugger = AgentDebugger(verbose=False)
    return _debugger


def run_pipeline(problem: dict, debug: bool = False) -> dict:
    task_id = problem["task_id"]
    print(f"\n{'='*60}")
    print(f"Task:               {task_id}")
    print(f"Entry point:        {problem['entry_point']}")
    print(f"\nPrompt:\n{problem['prompt']}")
    print(f"\nCanonical solution:\n{problem['canonical_solution']}")
    print(f"\nTest harness:\n{problem['test']}")
    print(f"{'='*60}")

    # --- Planner ---
    print("[PLANNER] Generating plan...")
    planner_input = f"Plan a solution for this problem:\n\n{problem['prompt']}"
    plan_text = plan(problem["prompt"])
    print(f"\n{plan_text}\n")

    # --- Coder ---
    print("[CODER] Generating solution...")
    coder_input = (
        f"Problem:\n{problem['prompt']}\n\n"
        f"Plan:\n{plan_text}\n\n"
        "Implement the solution:"
    )
    solution_code = code(problem["prompt"], plan_text)
    print(f"\n{solution_code}\n")

    # --- Verifier ---
    print("[VERIFIER] Running tests...")
    result = verify(solution_code, problem["test"], problem["entry_point"])
    status = "PASS ✓" if result["passed"] else "FAIL ✗"
    print(f"  Result: {status}")
    if not result["passed"]:
        print(f"\n--- Error ---\n{result['stderr']}")
        if result["review"]:
            print(f"\n--- Review ---\n{result['review']}")

    pipeline_result = {
        "task_id": task_id,
        "passed": result["passed"],
        "plan": plan_text,
        "code": solution_code,
        "error": result["stderr"],
        "review": result["review"],
        "debug": None,
    }

    # --- AgentDebug root cause analysis on failure ---
    if debug and not result["passed"]:
        from debugger import AgentStep, Trajectory

        trajectory = Trajectory(
            task_id=task_id,
            task_description=problem["prompt"],
            steps=[
                AgentStep(
                    step_num=1,
                    agent_name="planner",
                    agent_input=planner_input,
                    agent_output=plan_text,
                ),
                AgentStep(
                    step_num=2,
                    agent_name="coder",
                    agent_input=coder_input,
                    agent_output=solution_code,
                ),
                AgentStep(
                    step_num=3,
                    agent_name="verifier",
                    agent_input=(
                        f"Code:\n{solution_code}\n\n"
                        f"Test error:\n{result['stderr']}\n\n"
                        "What is wrong and how should it be fixed?"
                    ),
                    agent_output=result["review"] or "(no review generated)",
                    metadata={"passed": result["passed"], "error": result["stderr"]},
                ),
            ],
            final_passed=result["passed"],
            final_error=result["stderr"],
        )

        debug_result = _get_debugger().debug(trajectory)
        pipeline_result["debug"] = debug_result.to_dict()

    return pipeline_result


def _build_summary_report(results: list, pass_at_1: float) -> str:
    W = 72
    SEP = "=" * W
    THIN = "-" * W
    lines = []

    def _trunc(text: str, n: int) -> str:
        text = (text or "").replace("\n", " ").strip()
        return text[:n] + "…" if len(text) > n else text

    lines.append(f"\n{SEP}")
    lines.append("  FINAL SUMMARY REPORT")
    lines.append(f"  {len(results)} tasks  |  Pass@1: {pass_at_1:.2%}  "
                 f"({sum(1 for r in results if r['passed'])}/{len(results)} passed)")
    lines.append(SEP)

    for r in results:
        status = "PASS ✓" if r["passed"] else "FAIL ✗"
        lines.append(f"\n  Task       : {r['task_id']}")
        lines.append(f"  Result     : {status}")

        if not r["passed"]:
            lines.append(f"  Error      : {_trunc(r.get('error', ''), 80)}")

            rc = (r.get("debug") or {}).get("root_cause")
            if rc:
                agent = rc.get("critical_agent", "unknown").upper()
                step  = rc.get("critical_step", "?")
                etype = rc.get("error_type", "unknown")
                mod   = rc.get("module", "unknown")
                desc  = _trunc(rc.get("description", ""), 90)
                fix   = _trunc(rc.get("fix_suggestion", ""), 90)
                conf  = rc.get("confidence", 0.0)
                lines.append(f"  Root cause : [{mod.upper()}] {etype}  (confidence {conf:.0%})")
                lines.append(f"  Caused by  : Step {step} — {agent}")
                lines.append(f"  Description: {desc}")
                lines.append(f"  Fix hint   : {fix}")
            else:
                lines.append("  Root cause : (run with --debug to enable analysis)")

        lines.append(f"  {THIN}")

    lines.append(SEP)
    return "\n".join(lines)


def print_summary_report(results: list, pass_at_1: float) -> None:
    print(_build_summary_report(results, pass_at_1))


def save_summary(results: list, pass_at_1: float, timestamp: str) -> str:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    path = SUMMARY_DIR / f"summary_{timestamp}.txt"
    path.write_text(_build_summary_report(results, pass_at_1) + "\n")
    return str(path)


def save_results(results: list, pass_at_1: float, timestamp=None) -> str:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"run_{timestamp}.json"
    payload = {
        "timestamp": timestamp,
        "total": len(results),
        "passed": sum(1 for r in results if r["passed"]),
        "pass_at_1": round(pass_at_1, 4),
        "results": results,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5, help="Number of problems to run")
    parser.add_argument("--id", type=int, default=None, help="Run a single problem by index")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run AgentDebug root cause analysis on failed problems",
    )
    args = parser.parse_args()

    print("Loading HumanEval dataset...")
    problems = load_humaneval()
    print(f"Loaded {len(problems)} problems.\n")

    subset = [problems[args.id]] if args.id is not None else problems[: args.n]

    results = [run_pipeline(p, debug=args.debug) for p in subset]

    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    pass_at_1 = passed / total if total > 0 else 0.0

    print_summary_report(results, pass_at_1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = save_results(results, pass_at_1, timestamp)
    summary_path = save_summary(results, pass_at_1, timestamp)
    print(f"\nResults saved to  {out_path}")
    print(f"Summary saved to  {summary_path}")


if __name__ == "__main__":
    main()
