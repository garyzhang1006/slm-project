"""Static quality checks for generated Python without executing it."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass


CODE_TASK_TYPES = frozenset({"code_generation", "code_debugging"})
_FENCED_BLOCK = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL)


@dataclass(frozen=True)
class PythonQuality:
    syntax_valid: bool
    required_symbol_recall: float
    static_score: float
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "syntax_valid": self.syntax_valid,
            "required_symbol_recall": self.required_symbol_recall,
            "static_score": self.static_score,
            "error": self.error,
        }


def extract_python(text: str) -> str:
    """Return the largest fenced Python block, or raw text when no fence exists."""
    matches = [body for language, body in _FENCED_BLOCK.findall(text)
               if language.strip().casefold() in {"", "python", "py"}]
    return max(matches, key=len).strip() if matches else text.strip()


def python_syntax_valid(text: str) -> bool:
    try:
        tree = ast.parse(extract_python(text))
        compile(tree, "<generated-python>", "exec")
    except SyntaxError:
        return False
    return True


def _function_names(tree: ast.AST) -> set[str]:
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def assess_python(generated: str, expected: str) -> PythonQuality:
    """Measure syntax and expected function-name coverage; never execute code."""
    generated_source = extract_python(generated)
    expected_source = extract_python(expected)
    try:
        generated_tree = ast.parse(generated_source)
        expected_tree = ast.parse(expected_source)
        compile(generated_tree, "<generated-python>", "exec")
        compile(expected_tree, "<expected-python>", "exec")
    except SyntaxError as exc:
        return PythonQuality(False, 0.0, 0.0, f"SyntaxError: {exc.msg}")

    expected_names = _function_names(expected_tree)
    generated_names = _function_names(generated_tree)
    if expected_names:
        recall = len(expected_names.intersection(generated_names)) / len(expected_names)
    else:
        recall = 1.0
    score = 0.5 * recall + 0.5
    return PythonQuality(True, recall, score)


def assess_code(generated: str, expected: str, task_type: str) -> PythonQuality | None:
    """Assess Python only for tasks whose contract asks for executable code."""
    if task_type not in CODE_TASK_TYPES:
        return None
    return assess_python(generated, expected)
