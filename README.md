# Multi-Agent System with Root Cause Analysis

A **Planner → Coder → Verifier** multi-agent pipeline that solves [HumanEval](https://huggingface.co/datasets/openai_humaneval) coding problems, with a built-in two-phase root cause debugger inspired by [AgentDebug](https://github.com/ulab-uiuc/AgentDebug).

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
│  VERIFIER   │  Runs tests, on failure asks LLM to diagnose the bug
└──────┬──────┘
       │ pass / fail + stderr
       ▼
┌─────────────────────┐   (only on failure, with --debug)
│  AGENT DEBUGGER     │
│  Phase 1: per-step  │  llama3.2 classifies errors in each agent's output
│  Phase 2: root cause│  llama3.2 identifies the earliest failure point
└─────────────────────┘
       │
       ▼
  Final Summary Report
```

### Why a separate analyzer model?

The pipeline agents run on `qwen2.5-coder:7b`. Asking the same model to analyze its own mistakes is unreliable. The debugger uses `llama3.2` as an independent judge so there is no conflict of interest.

| Component | Model |
|---|---|
| Planner, Coder, Verifier | `qwen2.5-coder:7b` |
| Phase 1 — step error classifier | `llama3.2` |
| Phase 2 — root cause identifier | `llama3.2` |

---

## Project structure

```
├── main.py                        # Entry point
├── config.py                      # Model names + Ollama client
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
├── debugger/
│   ├── taxonomy.py                # 17 error types across 5 modules
│   ├── trajectory.py              # Trajectory + AgentStep dataclasses
│   ├── step_analyzer.py           # Phase 1: per-agent error detection
│   ├── root_cause.py              # Phase 2: root cause identification
│   ├── report.py                  # Terminal report renderer
│   ├── debugger.py                # AgentDebugger orchestrator
│   └── __init__.py
│
└── results/                       # JSON output from each run
```

---

## Error taxonomy (from AgentDebug)

The debugger classifies failures across 5 modules — each agent is only checked against the modules that apply to it.

| Module | Error types | Agents checked |
|---|---|---|
| **Memory** | over_simplification, memory_retrieval_failure, hallucination | Coder, Verifier |
| **Reflection** | progress_misjudge, outcome_misinterpretation, causal_misattribution | Coder, Verifier |
| **Planning** | constraint_ignorance, impossible_action, inefficient_plan | Planner, Coder |
| **Action** | misalignment, invalid_action, format_error, parameter_error | All |
| **System** | step_limit, tool_execution_error, llm_limit, environment_error | All |

> Planner (step 1) is never checked for memory or reflection — there is no prior history at that point.

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
ollama pull qwen2.5-coder:7b   # pipeline agents
ollama pull llama3.2            # debugger analyzer
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

# Enable root cause analysis on failures
python main.py --debug

# Combine flags
python main.py --n 10 --debug
python main.py --id 0 --debug
```

---

## Output

After all tasks complete, a **Final Summary Report** is printed:

```
========================================================================
  FINAL SUMMARY REPORT
  5 tasks  |  Pass@1: 60.00%  (3/5 passed)
========================================================================

  Task       : HumanEval/0
  Result     : PASS ✓
  ------------------------------------------------------------------------

  Task       : HumanEval/1
  Result     : FAIL ✗
  Error      : AssertionError: assert has_close_elements([1.0, 2.0], 0.5) == False
  Root cause : [ACTION] misalignment  (confidence 88%)
  Caused by  : Step 2 — CODER
  Description: Coder used strict inequality instead of checking absolute difference.
  Fix hint   : Use abs(a - b) < threshold instead of a != b.
  ------------------------------------------------------------------------
```

Results are also saved as JSON to `results/run_TIMESTAMP.json`.

To swap the analyzer for a stronger model, change one line in `config.py`:

```python
ANALYZER_MODEL = "llama3.1:8b"   # ollama pull llama3.1:8b
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
!ollama pull qwen2.5-coder:7b
!ollama pull llama3.2
```

### 6. Install dependencies and run
```python
!pip install -q openai datasets python-dotenv
!python main.py --n 5 --debug
```

> **Notes:** GPU is not required — Ollama runs on CPU. Both models together are ~6.7 GB (Kaggle gives 20 GB). For large runs (`--n 20+`) use a **committed** notebook to avoid timeout.

---

## References

- [AgentDebug — ulab-uiuc](https://github.com/ulab-uiuc/AgentDebug)
- [HumanEval dataset — OpenAI](https://huggingface.co/datasets/openai_humaneval)
- [Ollama](https://ollama.ai)
