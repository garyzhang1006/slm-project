"""Inspectable mini SLM and context-care tools for coding experiments."""

from typing import Any

__version__ = "0.4.0"

_CONTEXT_EXPORTS = frozenset(
    {"ContextAssessment", "ContextHandoff", "ContextMessage", "ContextTherapist"}
)

__all__ = [
    "ContextAssessment",
    "ContextHandoff",
    "ContextMessage",
    "ContextTherapist",
    "__version__",
]


def __getattr__(name: str) -> Any:
    if name in _CONTEXT_EXPORTS:
        from . import context_therapy

        value = getattr(context_therapy, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
