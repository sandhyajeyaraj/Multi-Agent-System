"""
Multi-Agent System: Planner → Coder → Verifier on HumanEval.

Usage:
    python main.py              # runs first 5 problems
    python main.py --n 20       # runs first 20 problems
    python main.py --id 42      # runs HumanEval/42 only
"""

import argparse
import json
import os
from datetime import datetime

from agents.coder import code
from agents.planner import plan
from agents.verifier import verify
from data.humaneval_loader import load_humaneval


def run_pipeline(problem: dict) -> dict:
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
    plan_text = plan(problem["prompt"])
    print(f"\n{plan_text}\n")

    # --- Coder ---
    print("[CODER] Generating solution...")
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

    return {
        "task_id": task_id,
        "passed": result["passed"],
        "plan": plan_text,
        "code": solution_code,
        "error": result["stderr"],
        "review": result["review"],
    }


def save_results(results: list, pass_at_1: float) -> str:
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"results/run_{timestamp}.json"
    payload = {
        "timestamp": timestamp,
        "total": len(results),
        "passed": sum(1 for r in results if r["passed"]),
        "pass_at_1": round(pass_at_1, 4),
        "results": results,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5, help="Number of problems to run")
    parser.add_argument("--id", type=int, default=None, help="Run a single problem by index")
    args = parser.parse_args()

    print("Loading HumanEval dataset...")
    problems = load_humaneval()
    print(f"Loaded {len(problems)} problems.\n")

    subset = [problems[args.id]] if args.id is not None else problems[: args.n]

    results = [run_pipeline(p) for p in subset]

    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    pass_at_1 = passed / total if total > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"Pass@1: {pass_at_1:.2%}  ({passed}/{total} passed)")
    print(f"{'='*60}")
    for r in results:
        icon = "✓" if r["passed"] else "✗"
        print(f"  {icon}  {r['task_id']}")

    out_path = save_results(results, pass_at_1)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
