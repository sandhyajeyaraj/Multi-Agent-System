from openai import OpenAI

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
# Per-agent model split: lets you test "small planner + large coder" combos.
# Set all three to the same value to replicate the original single-model setup.
PLANNER_MODEL  = "qwen2.5-coder:7b"
CODER_MODEL    = "qwen2.5-coder:7b"
VERIFIER_MODEL = "qwen2.5-coder:7b"

# Kept for backwards-compatibility — agents now import their specific model.
MODEL = CODER_MODEL

# Separate model used by the classifier/debugger to avoid judging its own output.
ANALYZER_MODEL = "llama3.2:latest"

# 32b variants for comparison runs:
# PLANNER_MODEL = CODER_MODEL = VERIFIER_MODEL = MODEL = "qwen2.5-coder:32b"
# ANALYZER_MODEL = "llama2:13b"

# ---------------------------------------------------------------------------
# Sampling — tune these for comparison runs
# ---------------------------------------------------------------------------
# temperature=0 + fixed seed → fully deterministic (best for ablations).
# Raise temperature (0.2–0.5) to let the model escape bad local solutions on retry.
TEMPERATURE = 0.0
TOP_P       = 0.95
SEED        = 42

# ---------------------------------------------------------------------------
# Token budgets — per agent
# ---------------------------------------------------------------------------
PLAN_MAX_TOKENS       = 1024
CODE_MAX_TOKENS       = 2048
REVIEW_MAX_TOKENS     = 512
CLASSIFIER_MAX_TOKENS = 256

# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
VERIFIER_TIMEOUT = 10  # seconds before a test run is killed

# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)
