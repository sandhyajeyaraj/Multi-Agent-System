"""
Phase 2 — Root cause identification.

Takes the Phase 1 step analyses and the original trajectory, then asks
the LLM to identify the single earliest failure that caused the cascade.
Mirrors AgentDebug's CriticalErrorAnalyzer logic.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from config import ANALYZER_MODEL as _MODEL
from config import client as _client
from debugger.step_analyzer import StepAnalysis
from debugger.taxonomy import TAXONOMY
from debugger.trajectory import Trajectory

MAX_RETRIES = 2


@dataclass
class RootCause:
    critical_step: int          # step number of the root cause (1=Planner, 2=Coder, 3=Verifier)
    critical_agent: str         # agent name
    module: str                 # which module the root cause lives in
    error_type: str             # specific error type from the taxonomy
    description: str            # plain-English root cause description
    evidence: str               # quote / observation from the agent output
    cascading_effects: str      # how this error propagated into later steps
    fix_suggestion: str         # concrete actionable fix
    confidence: float           # 0.0 – 1.0


def _format_step_analyses(analyses: list[StepAnalysis]) -> str:
    parts = []
    for sa in analyses:
        detected = sa.detected_errors()
        if not detected:
            error_lines = "  No errors detected."
        else:
            error_lines = "\n".join(
                f"  [{e.module.upper()}] {e.error_type}: {e.evidence}"
                for e in detected
            )
        parts.append(
            f"Step {sa.step_num} ({sa.agent_name.upper()}):\n"
            f"  Summary: {sa.summary}\n"
            f"  Errors:\n{error_lines}"
        )
    return "\n\n".join(parts)


def _valid_error_types_for_module(module: str) -> list[str]:
    return list(TAXONOMY.get(module, {}).keys())


def _build_root_cause_prompt(
    analyses: list[StepAnalysis],
    trajectory: Trajectory,
    is_retry: bool = False,
) -> str:
    step_summary = _format_step_analyses(analyses)
    full_trajectory = trajectory.to_chat_history()

    retry_note = ""
    if is_retry:
        retry_note = (
            "\nNOTE: A previous attempt incorrectly flagged Step 1 (Planner) with a "
            "memory or reflection error. Step 1 has no prior history, so those modules "
            "cannot apply. Re-evaluate starting from Step 2 if needed.\n"
        )

    return f"""You are an expert AI agent debugger performing root cause analysis.

A multi-agent pipeline (Planner → Coder → Verifier) failed to solve a coding task.
Your job: identify the SINGLE earliest decision or error that set the pipeline on an
irreversible path to failure. Think holistically — consider error propagation.
{retry_note}
TASK:
{trajectory.task_description}

FULL TRAJECTORY:
{full_trajectory}

FINAL ERROR:
{trajectory.final_error}

PHASE 1 STEP-LEVEL ERROR ANALYSES:
{step_summary}

INSTRUCTIONS:
1. Identify the EARLIEST step where a decision or error made task failure inevitable.
2. If multiple errors exist, prefer the one in the earliest step.
3. Step 1 (Planner) CANNOT have memory or reflection errors — no prior history exists.
4. System errors (step_limit, tool_execution_error) are valid root causes.
5. Trace how the root cause cascaded into later steps.
6. Suggest a concrete fix.

Respond with ONLY valid JSON in this exact format:
{{
  "critical_step": <1|2|3>,
  "critical_agent": "<planner|coder|verifier>",
  "module": "<memory|reflection|planning|action|system>",
  "error_type": "<exact error type from taxonomy>",
  "description": "<1-2 sentence plain-English root cause description>",
  "evidence": "<direct quote or observation from the agent output>",
  "cascading_effects": "<how this error led to later failures>",
  "fix_suggestion": "<concrete actionable fix for the agent or pipeline>",
  "confidence": <0.0 to 1.0>
}}"""


def _parse_root_cause(raw: str) -> RootCause | None:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group())
        return RootCause(
            critical_step=int(data.get("critical_step", 1)),
            critical_agent=data.get("critical_agent", "unknown"),
            module=data.get("module", "unknown"),
            error_type=data.get("error_type", "unknown"),
            description=data.get("description", ""),
            evidence=data.get("evidence", ""),
            cascading_effects=data.get("cascading_effects", ""),
            fix_suggestion=data.get("fix_suggestion", ""),
            confidence=float(data.get("confidence", 0.5)),
        )
    except (json.JSONDecodeError, ValueError):
        return None


def _validate_and_fix(rc: RootCause) -> RootCause:
    """Auto-correct impossible assignments (e.g. memory error on step 1)."""
    if rc.critical_step == 1 and rc.module in ("memory", "reflection"):
        rc.module = "planning"
        rc.error_type = "constraint_ignorance"
    valid_types = _valid_error_types_for_module(rc.module)
    if rc.error_type not in valid_types and valid_types:
        rc.error_type = valid_types[0]
    return rc


class RootCauseAnalyzer:
    """Phase 2: identify the single root cause from Phase 1 analyses."""

    def __init__(self, model: str = _MODEL):
        self.model = model

    def _call_llm(self, prompt: str) -> str:
        response = _client.chat.completions.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()

    def identify_root_cause(
        self,
        analyses: list[StepAnalysis],
        trajectory: Trajectory,
    ) -> RootCause | None:
        if trajectory.final_passed:
            return None

        for attempt in range(MAX_RETRIES + 1):
            is_retry = attempt > 0
            prompt = _build_root_cause_prompt(analyses, trajectory, is_retry)
            raw = self._call_llm(prompt)
            rc = _parse_root_cause(raw)
            if rc is None:
                continue

            # If step 1 is flagged with memory/reflection, retry once
            if rc.critical_step == 1 and rc.module in ("memory", "reflection") and not is_retry:
                continue

            return _validate_and_fix(rc)

        return None
