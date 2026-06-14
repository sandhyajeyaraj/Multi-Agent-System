"""
Data classes representing a captured agent run.

A Trajectory contains one AgentStep per agent in the pipeline.
Steps are numbered 1-based (1=Planner, 2=Coder, 3=Verifier).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AgentStep:
    step_num: int           # 1=Planner, 2=Coder, 3=Verifier
    agent_name: str         # "planner" | "coder" | "verifier"
    agent_input: str        # the prompt / context fed to this agent
    agent_output: str       # the raw text the agent returned
    metadata: dict = field(default_factory=dict)  # e.g. {"error": "...", "passed": False}


@dataclass
class Trajectory:
    task_id: str
    task_description: str   # the full problem prompt
    steps: list[AgentStep]
    final_passed: bool
    final_error: str = ""   # stderr / error message on failure

    def get_step(self, agent_name: str) -> AgentStep | None:
        for s in self.steps:
            if s.agent_name == agent_name:
                return s
        return None

    def steps_before(self, step_num: int) -> list[AgentStep]:
        return [s for s in self.steps if s.step_num < step_num]

    def to_chat_history(self) -> str:
        """Render the full trajectory as a readable conversation for LLM prompts."""
        parts = []
        for step in self.steps:
            parts.append(
                f"=== Step {step.step_num}: {step.agent_name.upper()} ===\n"
                f"[INPUT]\n{step.agent_input}\n\n"
                f"[OUTPUT]\n{step.agent_output}"
            )
        return "\n\n".join(parts)
