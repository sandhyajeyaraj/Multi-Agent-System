"""
Renders a human-readable diagnostic report from Phase 1 + Phase 2 results.
Prints to stdout and optionally returns the report as a string.
"""

from __future__ import annotations

from debugger.root_cause import RootCause
from debugger.step_analyzer import StepAnalysis
from debugger.trajectory import Trajectory

_WIDTH = 70
_SEP = "=" * _WIDTH
_THIN = "-" * _WIDTH

AGENT_LABEL = {
    "planner": "Step 1 · PLANNER",
    "coder":   "Step 2 · CODER",
    "verifier":"Step 3 · VERIFIER",
}

MODULE_EMOJI = {
    "memory":     "[MEM]",
    "reflection": "[REF]",
    "planning":   "[PLN]",
    "action":     "[ACT]",
    "system":     "[SYS]",
}


def _wrap(text: str, indent: int = 4) -> str:
    pad = " " * indent
    words = text.split()
    lines, current = [], ""
    for w in words:
        if len(current) + len(w) + 1 > _WIDTH - indent:
            if current:
                lines.append(pad + current)
            current = w
        else:
            current = (current + " " + w).strip()
    if current:
        lines.append(pad + current)
    return "\n".join(lines)


def build_report(
    trajectory: Trajectory,
    step_analyses: list[StepAnalysis],
    root_cause: RootCause | None,
) -> str:
    lines: list[str] = []

    lines += [
        _SEP,
        f"  AGENT DEBUGGER REPORT",
        f"  Task: {trajectory.task_id}",
        f"  Result: {'PASS' if trajectory.final_passed else 'FAIL'}",
        _SEP,
    ]

    if trajectory.final_passed:
        lines.append("  No errors to report — task passed.")
        return "\n".join(lines)

    # Phase 1
    lines += ["", "PHASE 1 — STEP-LEVEL ERROR ANALYSIS", _THIN]
    for sa in step_analyses:
        label = AGENT_LABEL.get(sa.agent_name, sa.agent_name.upper())
        lines.append(f"\n{label}")
        lines.append(f"  Summary: {sa.summary}")
        detected = sa.detected_errors()
        if not detected:
            lines.append("  No errors detected.")
        else:
            for e in detected:
                tag = MODULE_EMOJI.get(e.module, f"[{e.module[:3].upper()}]")
                lines.append(f"  {tag} {e.error_type}")
                lines.append(_wrap(f"Evidence: {e.evidence}"))
                lines.append(_wrap(f"Reasoning: {e.reasoning}"))

    # Phase 2
    lines += ["", _THIN, "PHASE 2 — ROOT CAUSE ANALYSIS", _THIN]
    if root_cause is None:
        lines.append("  Root cause could not be determined.")
    else:
        tag = MODULE_EMOJI.get(root_cause.module, f"[{root_cause.module[:3].upper()}]")
        agent_label = AGENT_LABEL.get(root_cause.critical_agent, root_cause.critical_agent)
        lines += [
            f"  Root cause at: {agent_label}",
            f"  Module:        {tag} {root_cause.module}",
            f"  Error type:    {root_cause.error_type}",
            f"  Confidence:    {root_cause.confidence:.0%}",
            "",
            "  DESCRIPTION:",
            _wrap(root_cause.description),
            "",
            "  EVIDENCE:",
            _wrap(root_cause.evidence),
            "",
            "  CASCADING EFFECTS:",
            _wrap(root_cause.cascading_effects),
            "",
            "  FIX SUGGESTION:",
            _wrap(root_cause.fix_suggestion),
        ]

    lines += ["", _SEP]
    return "\n".join(lines)


def print_report(
    trajectory: Trajectory,
    step_analyses: list[StepAnalysis],
    root_cause: RootCause | None,
) -> str:
    report = build_report(trajectory, step_analyses, root_cause)
    print(report)
    return report
