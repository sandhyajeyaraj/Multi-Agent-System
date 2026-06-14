"""
AgentDebugger — root cause analysis for the Planner→Coder→Verifier pipeline.

Typical usage:

    from debugger import AgentDebugger, Trajectory, AgentStep

    traj = Trajectory(
        task_id="HumanEval/42",
        task_description=problem["prompt"],
        steps=[
            AgentStep(1, "planner", planner_input, plan_text),
            AgentStep(2, "coder",   coder_input,   solution_code),
            AgentStep(3, "verifier", verifier_input, review_text,
                      metadata={"error": stderr}),
        ],
        final_passed=False,
        final_error=stderr,
    )

    debugger = AgentDebugger()
    report   = debugger.debug(traj)      # runs Phase 1 + Phase 2, prints report
"""

from debugger.debugger import AgentDebugger
from debugger.report import build_report, print_report
from debugger.root_cause import RootCause, RootCauseAnalyzer
from debugger.step_analyzer import ModuleError, StepAnalysis, StepAnalyzer
from debugger.taxonomy import AGENT_MODULES, TAXONOMY
from debugger.trajectory import AgentStep, Trajectory

__all__ = [
    "AgentDebugger",
    "AgentStep",
    "Trajectory",
    "StepAnalyzer",
    "StepAnalysis",
    "ModuleError",
    "RootCauseAnalyzer",
    "RootCause",
    "build_report",
    "print_report",
    "TAXONOMY",
    "AGENT_MODULES",
]
