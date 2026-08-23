from openai import OpenAI

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
# Pipeline agents — smaller model for planner/verifier, bigger model for the coder.
PLANNER_MODEL  = "codellama:7b"
CODER_MODEL    = "codellama:7b"
VERIFIER_MODEL = "codellama:7b"

# Escalation — only the coder is upgraded during recovery, to an even bigger model.
RECOVERY_CODER_MODEL = "codellama:13b"

# Kept for backwards-compatibility — agents now import their specific model.
MODEL = CODER_MODEL

# Separate model used by the classifier/debugger to avoid judging its own output.
ANALYZER_MODEL = "llama3.2:latest"

# Bigger model dedicated to the Phase 2 trace analyzer — diagnosing which
# variable diverged from a captured runtime trace needs more careful
# reasoning than ANALYZER_MODEL reliably gives.
TRACE_ANALYZER_MODEL = "llama3.2:latest"

# ---------------------------------------------------------------------------
# Ablation switches
# ---------------------------------------------------------------------------
# Experiment 1: skip the trace-analyzer agent on step-2 (Coder) failures and
# feed the raw assertion error + captured debug-print trace directly to the
# Recovery Coder as error_context, instead of a synthesized diagnosis.
SKIP_TRACE_ANALYZER = True

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
TRACE_MAX_TOKENS      = 512

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
