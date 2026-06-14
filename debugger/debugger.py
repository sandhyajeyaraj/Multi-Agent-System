"""
AgentDebugger — top-level orchestrator combining Phase 1 + Phase 2.
"""

from __future__ import annotations

from dataclasses import dataclass

from config import ANALYZER_MODEL as _MODEL
from debugger.report import build_report, print_report
from debugger.root_cause import RootCause, RootCauseAnalyzer
from debugger.step_analyzer import StepAnalysis, StepAnalyzer
from debugger.trajectory import Trajectory


@dataclass
class DebugResult:
    trajectory: Trajectory
    step_analyses: list[StepAnalysis]
    root_cause: RootCause | None
    report: str

    def to_dict(self) -> dict:
        rc = self.root_cause
        return {
            "phase1": [
                {
                    "step": sa.step_num,
                    "agent": sa.agent_name,
                    "summary": sa.summary,
                    "errors": [
                        {
                            "module": e.module,
                            "error_type": e.error_type,
                            "detected": e.detected,
                            "evidence": e.evidence,
                            "reasoning": e.reasoning,
                        }
                        for e in sa.module_errors
                    ],
                }
                for sa in self.step_analyses
            ],
            "root_cause": {
                "critical_step": rc.critical_step,
                "critical_agent": rc.critical_agent,
                "module": rc.module,
                "error_type": rc.error_type,
                "description": rc.description,
                "evidence": rc.evidence,
                "cascading_effects": rc.cascading_effects,
                "fix_suggestion": rc.fix_suggestion,
                "confidence": rc.confidence,
            } if rc else None,
        }


class AgentDebugger:
    """Run the full two-phase debugging pipeline on a failed trajectory."""

    def __init__(self, model: str = _MODEL, verbose: bool = True):
        self.step_analyzer = StepAnalyzer(model=model)
        self.root_cause_analyzer = RootCauseAnalyzer(model=model)
        self.verbose = verbose

    def debug(self, trajectory: Trajectory) -> DebugResult:
        if trajectory.final_passed:
            report = build_report(trajectory, [], None)
            if self.verbose:
                print(report)
            return DebugResult(
                trajectory=trajectory,
                step_analyses=[],
                root_cause=None,
                report=report,
            )

        if self.verbose:
            print(f"\n[DEBUGGER] Running Phase 1 — step-level analysis for {trajectory.task_id}...")

        step_analyses = self.step_analyzer.analyze_trajectory(trajectory)

        if self.verbose:
            print(f"[DEBUGGER] Running Phase 2 — root cause identification...")

        root_cause = self.root_cause_analyzer.identify_root_cause(step_analyses, trajectory)

        report = print_report(trajectory, step_analyses, root_cause) if self.verbose \
            else build_report(trajectory, step_analyses, root_cause)

        return DebugResult(
            trajectory=trajectory,
            step_analyses=step_analyses,
            root_cause=root_cause,
            report=report,
        )
