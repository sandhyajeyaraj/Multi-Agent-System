"""
Error taxonomy adapted from AgentDebug (ulab-uiuc/AgentDebug).

17 error types across 5 modules. Each agent in the Planner→Coder→Verifier
pipeline is only checked against the modules that apply to it.
"""

# fmt: off
TAXONOMY: dict[str, dict[str, str]] = {
    "memory": {
        "over_simplification": (
            "The agent oversimplifies information from prior steps, discarding "
            "details that are needed for the current step."
        ),
        "memory_retrieval_failure": (
            "Relevant information produced in a prior step exists but the agent "
            "fails to use or reference it when it is needed."
        ),
        "hallucination": (
            "The agent recalls or asserts facts about prior steps or context that "
            "never actually occurred or were never produced."
        ),
    },
    "reflection": {
        "progress_misjudge": (
            "The agent incorrectly evaluates how far along the task is — either "
            "believing it is done when it is not, or failing to recognize progress."
        ),
        "outcome_misinterpretation": (
            "The agent executes a step correctly but misreads the result or "
            "feedback from that step, leading to a wrong conclusion."
        ),
        "causal_misattribution": (
            "The agent identifies that something went wrong but attributes it to "
            "the wrong cause, obscuring the real problem."
        ),
    },
    "planning": {
        "constraint_ignorance": (
            "The agent's plan ignores a known constraint, requirement, or edge "
            "case stated in the task description."
        ),
        "impossible_action": (
            "The agent plans an action that is fundamentally impossible given the "
            "current environment or available tools."
        ),
        "inefficient_plan": (
            "The plan is theoretically valid but extremely inefficient — it takes "
            "far more steps or uses a much slower algorithm than necessary."
        ),
    },
    "action": {
        "misalignment": (
            "The agent's executed action contradicts or diverges from its stated "
            "plan or the task requirements."
        ),
        "invalid_action": (
            "The agent attempts an operation that is not available, not defined, "
            "or syntactically/semantically incorrect in the current context."
        ),
        "format_error": (
            "The agent's output has an invalid format that prevents it from being "
            "parsed or used by the next step."
        ),
        "parameter_error": (
            "The action is conceptually correct but uses wrong, unreasonable, or "
            "missing parameter values."
        ),
    },
    "system": {
        "step_limit": (
            "The agent reached a maximum step or token limit before completing "
            "the task, despite making reasonable progress."
        ),
        "tool_execution_error": (
            "An external tool (subprocess, test runner, API) returned an error or "
            "produced unexpected behaviour outside the agent's control."
        ),
        "llm_limit": (
            "The LLM encountered an API timeout, context-length overflow, or "
            "other provider-level constraint."
        ),
        "environment_error": (
            "The execution environment itself had a bug, race condition, or "
            "violated an undocumented rule that caused the failure."
        ),
    },
}

# Which modules to check for each agent (step 1 has no memory/reflection history)
AGENT_MODULES: dict[str, list[str]] = {
    "planner":  ["planning", "action", "system"],
    "coder":    ["memory", "reflection", "planning", "action", "system"],
    "verifier": ["memory", "reflection", "action", "system"],
}

ALL_MODULES = list(TAXONOMY.keys())


def error_types_for_module(module: str) -> dict[str, str]:
    return TAXONOMY.get(module, {})


def modules_for_agent(agent_name: str) -> list[str]:
    return AGENT_MODULES.get(agent_name, ALL_MODULES)


def format_module_definitions(module: str) -> str:
    errors = TAXONOMY.get(module, {})
    lines = [f"Module: {module.upper()}"]
    for name, desc in errors.items():
        lines.append(f"  - {name}: {desc}")
    return "\n".join(lines)
