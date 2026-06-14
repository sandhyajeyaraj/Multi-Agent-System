"""
Phase 1 — Step-level error detection.

For each agent step, ask the LLM to check the applicable modules
(from taxonomy.AGENT_MODULES) and return a structured JSON report.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from config import ANALYZER_MODEL as _MODEL
from config import client as _client
from debugger.taxonomy import (
    format_module_definitions,
    modules_for_agent,
)
from debugger.trajectory import AgentStep, Trajectory


@dataclass
class ModuleError:
    module: str
    error_type: str        # e.g. "constraint_ignorance" or "none"
    detected: bool
    evidence: str          # quote from the agent output that shows the error
    reasoning: str         # why this counts as that error type


@dataclass
class StepAnalysis:
    step_num: int
    agent_name: str
    module_errors: list[ModuleError] = field(default_factory=list)
    summary: str = ""

    def has_errors(self) -> bool:
        return any(e.detected for e in self.module_errors)

    def detected_errors(self) -> list[ModuleError]:
        return [e for e in self.module_errors if e.detected]


def _build_step_prompt(
    step: AgentStep,
    trajectory: Trajectory,
    module: str,
    prior_module_outputs: dict[str, str],
) -> str:
    prior_steps = trajectory.steps_before(step.step_num)

    prior_context = ""
    if prior_steps:
        prior_context = "PRIOR AGENT OUTPUTS:\n"
        for ps in prior_steps:
            prior_context += f"  [{ps.agent_name.upper()}]: {ps.agent_output[:600]}\n"
    else:
        prior_context = "PRIOR AGENT OUTPUTS: None (this is the first step).\n"

    same_step_context = ""
    if prior_module_outputs:
        same_step_context = "EARLIER MODULE ANALYSES THIS STEP:\n"
        for mod, out in prior_module_outputs.items():
            same_step_context += f"  [{mod.upper()}]: {out}\n"

    module_defs = format_module_definitions(module)

    return f"""You are an expert AI agent debugger. Analyze the output of a single agent step for errors in one specific module.

TASK DESCRIPTION:
{trajectory.task_description}

{prior_context}

CURRENT STEP: {step.agent_name.upper()} (step {step.step_num})
AGENT INPUT:
{step.agent_input}

AGENT OUTPUT:
{step.agent_output}

{same_step_context}

MODULE TO ANALYZE: {module.upper()}
{module_defs}

INSTRUCTIONS:
Determine whether this agent step contains an error in the {module.upper()} module.
If an error is detected, specify the exact error type from the list above.
If no error, set error_type to "none" and detected to false.

Respond with ONLY valid JSON in this exact format:
{{
  "module": "{module}",
  "error_type": "<error_type_name_or_none>",
  "detected": <true|false>,
  "evidence": "<direct quote or observation from the agent output>",
  "reasoning": "<explanation of why this is or is not an error>"
}}"""


def _parse_module_error(raw: str, module: str) -> ModuleError:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return ModuleError(
            module=module,
            error_type="none",
            detected=False,
            evidence="",
            reasoning=f"Could not parse LLM response: {raw[:200]}",
        )
    try:
        data = json.loads(match.group())
        return ModuleError(
            module=module,
            error_type=data.get("error_type", "none"),
            detected=bool(data.get("detected", False)),
            evidence=data.get("evidence", ""),
            reasoning=data.get("reasoning", ""),
        )
    except json.JSONDecodeError:
        return ModuleError(
            module=module,
            error_type="none",
            detected=False,
            evidence="",
            reasoning=f"JSON parse failed: {raw[:200]}",
        )


def _build_summary_prompt(step: AgentStep, module_errors: list[ModuleError]) -> str:
    detected = [e for e in module_errors if e.detected]
    if not detected:
        error_list = "No errors detected in any module."
    else:
        lines = [f"  - [{e.module.upper()}] {e.error_type}: {e.evidence}" for e in detected]
        error_list = "\n".join(lines)

    return f"""Summarize the error analysis for this agent step in 1-2 sentences.

Agent: {step.agent_name.upper()} (step {step.step_num})

Detected errors:
{error_list}

Write a concise plain-English summary. If there are no errors, say so briefly."""


class StepAnalyzer:
    """Phase 1: analyze each agent step for module-level errors."""

    def __init__(self, model: str = _MODEL):
        self.model = model

    def _call_llm(self, prompt: str, max_tokens: int = 512) -> str:
        response = _client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()

    def analyze_step(self, step: AgentStep, trajectory: Trajectory) -> StepAnalysis:
        modules = modules_for_agent(step.agent_name)
        module_errors: list[ModuleError] = []
        prior_module_outputs: dict[str, str] = {}

        for module in modules:
            prompt = _build_step_prompt(step, trajectory, module, prior_module_outputs)
            raw = self._call_llm(prompt)
            error = _parse_module_error(raw, module)
            module_errors.append(error)
            prior_module_outputs[module] = f"{error.error_type} | {error.reasoning[:200]}"

        summary_prompt = _build_summary_prompt(step, module_errors)
        summary = self._call_llm(summary_prompt, max_tokens=256)

        return StepAnalysis(
            step_num=step.step_num,
            agent_name=step.agent_name,
            module_errors=module_errors,
            summary=summary,
        )

    def analyze_trajectory(self, trajectory: Trajectory) -> list[StepAnalysis]:
        """Run Phase 1 over all steps. Skips successful trajectories."""
        if trajectory.final_passed:
            return []
        return [self.analyze_step(step, trajectory) for step in trajectory.steps]
