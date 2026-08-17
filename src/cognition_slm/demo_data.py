"""Project-authored synthetic examples used for the smoke project."""

from __future__ import annotations

import json
from pathlib import Path


TRAIN_ROWS = [
    {
        "id": "train-factorial-loop",
        "prompt": "Write a Python function that returns the factorial of a non-negative integer n.",
        "answer": "def factorial(n):\n    result = 1\n    for value in range(2, n + 1):\n        result *= value\n    return result",
        "task_type": "code_generation",
        "confidence": 0.92,
        "error_category": "none",
        "source": "project-synthetic",
        "license": "CC0-1.0",
    },
    {
        "id": "train-palindrome",
        "prompt": "Write a Python function that returns True when text reads the same backward after lowercasing.",
        "answer": "def is_palindrome(text):\n    normalized = text.lower()\n    return normalized == normalized[::-1]",
        "task_type": "code_generation",
        "confidence": 0.9,
        "error_category": "none",
        "source": "project-synthetic",
        "license": "CC0-1.0",
    },
    {
        "id": "train-debug-off-by-one",
        "prompt": "Fix this function so it returns the sum of every item: def total(items): return sum(items[:-1])",
        "answer": "def total(items):\n    return sum(items)",
        "task_type": "code_debugging",
        "confidence": 0.86,
        "error_category": "logic",
        "source": "project-synthetic",
        "license": "CC0-1.0",
    },
    {
        "id": "train-debug-colon",
        "prompt": "Fix the syntax error in: for item in items print(item)",
        "answer": "for item in items:\n    print(item)",
        "task_type": "code_debugging",
        "confidence": 0.97,
        "error_category": "syntax",
        "source": "project-synthetic",
        "license": "CC0-1.0",
    },
    {
        "id": "train-explain-comprehension",
        "prompt": "Explain this Python expression in one concise paragraph: [x * 2 for x in values if x > 0]",
        "answer": "It builds a new list containing twice each positive value from values, while skipping zero and negative values.",
        "task_type": "code_explanation",
        "confidence": 0.88,
        "error_category": "none",
        "source": "project-synthetic",
        "license": "CC0-1.0",
    },
    {
        "id": "train-binary-search",
        "prompt": "Describe the invariant that makes binary search correct on a sorted list.",
        "answer": "At every step, target can only remain inside the current inclusive search interval; comparing with the midpoint removes a half that cannot contain target.",
        "task_type": "algorithm_reasoning",
        "confidence": 0.79,
        "error_category": "none",
        "source": "project-synthetic",
        "license": "CC0-1.0",
    },
    {
        "id": "train-safe-file",
        "prompt": "Give a safe rule for opening a user-provided file path in a coding assistant example.",
        "answer": "Validate the path against an allowed directory, reject traversal outside it, and avoid executing file contents as code.",
        "task_type": "metacognitive_review",
        "confidence": 0.73,
        "error_category": "unsafe",
        "source": "project-synthetic",
        "license": "CC0-1.0",
    },
    {
        "id": "train-uncertain-api",
        "prompt": "Review an answer that names an unfamiliar library API without a version or test. What should the assistant say?",
        "answer": "State that the API is unverified, check the installed version or official documentation, and provide a small test before presenting it as working.",
        "task_type": "metacognitive_review",
        "confidence": 0.42,
        "error_category": "hallucination",
        "source": "project-synthetic",
        "license": "CC0-1.0",
    },
]

EVAL_ROWS = [
    {
        "id": "eval-factorial-recursion",
        "prompt": "Write a Python function that returns factorial recursively for n >= 0.",
        "answer": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)",
        "task_type": "code_generation",
        "confidence": 0.78,
        "error_category": "none",
        "source": "project-synthetic",
        "license": "CC0-1.0",
    },
    {
        "id": "eval-debug-index",
        "prompt": "Fix this function so it returns the first item: def first(items): return items[1]",
        "answer": "def first(items):\n    return items[0]",
        "task_type": "code_debugging",
        "confidence": 0.88,
        "error_category": "logic",
        "source": "project-synthetic",
        "license": "CC0-1.0",
    },
    {
        "id": "eval-explain-map",
        "prompt": "Explain what list(map(str, numbers)) returns.",
        "answer": "It returns a list containing the string form of each item in numbers.",
        "task_type": "code_explanation",
        "confidence": 0.82,
        "error_category": "none",
        "source": "project-synthetic",
        "license": "CC0-1.0",
    },
    {
        "id": "eval-review-uncertainty",
        "prompt": "What should a coding assistant do when its proposed fix has not been run against a test?",
        "answer": "Label the fix as unverified, state the expected behavior, and run or provide a focused test before claiming it works.",
        "task_type": "metacognitive_review",
        "confidence": 0.5,
        "error_category": "incomplete",
        "source": "project-synthetic",
        "license": "CC0-1.0",
    },
]


def write_jsonl(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
