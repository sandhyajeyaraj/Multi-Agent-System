# Multi-Agent System with Recovery

A **Planner → Coder → Verifier** multi-agent pipeline that solves [HumanEval](https://huggingface.co/datasets/openai_humaneval) coding problems, with an optional failure-handling layer:

- **Recovery Orchestrator** (`--recover`) — classifies the failing step and reruns the pipeline from that point

---

## How it works

```
Problem Prompt
     │
     ▼
┌─────────────┐
│   PLANNER   │  ReAct-style plan (algorithm + edge cases + steps)
└──────┬──────┘
       │ plan_text
       ▼
┌─────────────┐
│    CODER    │  Python implementation based on the plan
└──────┬──────┘
       │ solution_code
       ▼
┌─────────────┐
│  VERIFIER   │  Runs tests; on failure asks LLM to diagnose the bug
└──────┬──────┘
       │ pass / fail + stderr + review
       │
       ├─── PASS ──────────────────────────────────── Final Summary
       │
       └─── FAIL ──── (--recover) ────┐
                                       ▼
                       ┌───────────────────────┐
                       │  FAILURE CLASSIFIER   │
                       │  llama3.2 classifies  │
                       │  which step failed    │
                       └──────────┬────────────┘
                                  │ failing_step
                                  ▼
                       ┌───────────────────────┐
                       │  RECOVERY ORCHESTRAT. │
                       │  step 1 → full rerun  │
                       │  step 2 → Coder+Verf  │
                       │  step 3 → Verif only  │
                       └───────────┬───────────┘
                                   │
                                   ▼
                             Final Summary Report
```

### Why a separate analyzer model?

The pipeline agents run on `qwen2.5-coder:32b`. Asking the same model to judge its own mistakes is unreliable. The failure classifier uses `llama3.2` as an independent judge.

| Component | Model | Config key |
|---|---|---|
| Planner | `qwen2.5-coder:32b` | `PLANNER_MODEL` |
| Coder | `qwen2.5-coder:32b` | `CODER_MODEL` |
| Verifier | `qwen2.5-coder:32b` | `VERIFIER_MODEL` |
| Failure classifier (recovery) | `llama3.2` | `ANALYZER_MODEL` |

> All three pipeline model keys are independent — you can test a "small planner + large coder" combo by changing only `PLANNER_MODEL` in `config.py`.

---

## Project structure

```
├── main.py                        # Entry point
├── config.py                      # Models, sampling params, token budgets, Ollama client
├── requirements.txt
│
├── agents/
│   ├── planner.py                 # ReAct planner agent
│   ├── coder.py                   # ReAct coder agent
│   └── verifier.py                # Test runner + LLM reviewer
│
├── data/
│   └── humaneval_loader.py        # Loads HumanEval from HuggingFace
│
├── recovery/                      # Targeted recovery on failure
│   ├── classifier.py              # LLM classifies which step (1/2/3) caused the failure
│   ├── orchestrator.py            # Reruns the pipeline from the classified failing step
│   └── __init__.py
│
├── results/                       # JSON output from each run
└── summary/                       # Plain-text summary reports
```

---

## Recovery Orchestrator

When `--recover` is passed, failed problems go through a two-stage recovery:

**1. Failure Classifier** (`recovery/classifier.py`)

Uses `llama3.2` to read the problem prompt, plan, code, test error, and verifier review, then returns the step most likely responsible:

| Classified step | Meaning | Recovery action |
|---|---|---|
| 1 — Planner | Wrong algorithm or missed constraints | Rerun Planner → Coder → Verifier with error context |
| 2 — Coder | Sound plan but buggy implementation | Reuse plan, rerun Coder → Verifier with error context |
| 3 — Verifier | Correct logic, execution-level glitch | Rerun Verifier only |

**2. Recovery Orchestrator** (`recovery/orchestrator.py`)

Executes the routing decision above and returns the recovery result, which appears in the final summary alongside the initial result.

---

## Setup

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.ai) installed and running

### Install

```bash
git clone https://github.com/YOUR_USERNAME/Multi-Agent-System.git
cd Multi-Agent-System
pip install -r requirements.txt
```

### Pull models

```bash
ollama pull qwen2.5-coder:32b   # pipeline agents (Planner, Coder, Verifier)
ollama pull llama3.2             # failure classifier
```

---

## Usage

```bash
# Run first 5 problems (default)
python main.py

# Run first N problems
python main.py --n 20

# Run a single problem by index (0-based, HumanEval has 164 problems)
python main.py --id 42

# Enable recovery: classify failing step and rerun from there
python main.py --recover
python main.py --n 10 --recover
python main.py --id 0 --recover
```

---

## Output

After all tasks complete, a **Final Summary Report** is printed and saved to `summary/`:

```
========================================================================
  FINAL SUMMARY REPORT
  5 tasks  |  Pass@1: 60.00%  (3/5 passed)
========================================================================

  Task       : HumanEval/0
  Result     : PASS ✓  (12.3s)
  ------------------------------------------------------------------------

  Task       : HumanEval/1
  Result     : FAIL ✗  (9.8s)
  Error      : AssertionError: assert has_close_elements([1.0, 2.0], 0.5) == False
  Recovery   : classified → step 2 (Coder)
  Rec reason : Plan was correct but coder used strict inequality instead of abs difference.
  Rec result : PASS ✓  (8.1s)
  Total time : 17.9s  (9.8s initial + 8.1s recovery)
  ------------------------------------------------------------------------
```

Results are also saved as JSON to `results/run_TIMESTAMP.json`.

---

## Configuration

Key knobs in `config.py`:

```python
# Swap any agent to a different model independently
PLANNER_MODEL  = "qwen2.5-coder:32b"
CODER_MODEL    = "qwen2.5-coder:32b"
VERIFIER_MODEL = "qwen2.5-coder:32b"
ANALYZER_MODEL = "llama3.2:latest"   # used by the recovery classifier

# Sampling — temperature=0 + fixed seed gives fully deterministic results
TEMPERATURE = 0.0
TOP_P       = 0.95
SEED        = 42

# Per-agent token budgets
PLAN_MAX_TOKENS       = 1024
CODE_MAX_TOKENS       = 2048
REVIEW_MAX_TOKENS     = 512
CLASSIFIER_MAX_TOKENS = 256

# Kill a test run after this many seconds
VERIFIER_TIMEOUT = 10
```

---

## Running on Kaggle

### 1. Create a notebook
New Notebook → enable **Internet** in settings.

### 2. Upload your code
Upload as a zip via **Add Data**, then extract:
```python
import zipfile, os
with zipfile.ZipFile("/kaggle/input/YOUR_DATASET/multi-agent-system.zip") as z:
    z.extractall("/kaggle/working/")
os.chdir("/kaggle/working/Multi-Agent-System")
```

### 3. Install Ollama
```python
!curl -fsSL https://ollama.ai/install.sh | sh
```

### 4. Start Ollama in the background
```python
import subprocess, time
subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(4)
```

### 5. Pull models
```python
!ollama pull qwen2.5-coder:32b
!ollama pull llama3.2
```

### 6. Install dependencies and run
```python
!pip install -q openai datasets python-dotenv
!python main.py --n 5 --recover
```

> **Notes:** The `32b` model requires ~20 GB of VRAM/RAM. Use the `7b` variant (`qwen2.5-coder:7b`) on CPU-only Kaggle notebooks. Both `qwen2.5-coder:7b` + `llama3.2` together are ~6.7 GB (Kaggle gives 20 GB). For large runs (`--n 20+`) use a **committed** notebook to avoid timeout.

---

## References

- [HumanEval dataset — OpenAI](https://huggingface.co/datasets/openai_humaneval)
- [Ollama](https://ollama.ai)
