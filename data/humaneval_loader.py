from datasets import load_dataset


def load_humaneval(split: str = "test") -> list[dict]:
    """Load the HumanEval benchmark dataset.

    Each problem dict contains:
      task_id            - e.g. "HumanEval/0"
      prompt             - function signature + docstring
      canonical_solution - reference implementation
      test               - test harness code (defines check())
      entry_point        - name of the function to test
    """
    dataset = load_dataset("openai_humaneval", split=split)
    return list(dataset)
