"""Static quality checks for generated Python without executing it."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass


CODE_TASK_TYPES = frozenset({"code_generation", "code_debugging"})
# Null bytes raise ValueError before Python 3.12, and deep nesting overflows the parser (MemoryError)
# or the compiler (RecursionError); all of them mean the text is not usable Python.
_UNPARSEABLE = (SyntaxError, ValueError, RecursionError, MemoryError)
# A generation cut off at max_new_tokens can leave its last fence unclosed.
_FENCED_BLOCK = re.compile(r"```([^\n`]*)\n(.*?)(?:```|\Z)", re.DOTALL)


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


def _is_trivial(tree: ast.Module) -> bool:
    """Empty output or prose that parses as bare names or literals is not code."""
    return all(
        isinstance(node, ast.Expr) and isinstance(node.value, (ast.Constant, ast.Name))
        for node in tree.body
    )


def python_syntax_valid(text: str) -> bool:
    try:
        tree = ast.parse(extract_python(text))
        compile(tree, "<generated-python>", "exec")
    except _UNPARSEABLE:
        return False
    return not _is_trivial(tree)


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
        compile(generated_tree, "<generated-python>", "exec")
    except _UNPARSEABLE as exc:
        return PythonQuality(False, 0.0, 0.0, f"{type(exc).__name__}: {exc.msg if isinstance(exc, SyntaxError) else exc}")
    if _is_trivial(generated_tree):
        return PythonQuality(False, 0.0, 0.0, "no Python statements in generation")
    # A prose reference answer names no functions to recall; it says nothing about the generation's syntax.
    try:
        expected_names = _function_names(ast.parse(expected_source))
    except _UNPARSEABLE:
        expected_names = set()

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
