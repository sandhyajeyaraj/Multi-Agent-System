from openai import OpenAI

# Model used by the pipeline agents (Planner, Coder, Verifier)
MODEL = "qwen2.5-coder:7b"

# Separate model used by the debugger to analyze agent mistakes.
# Must be different from MODEL so it's not judging its own output.
ANALYZER_MODEL = "llama3.2:latest"

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)
