"""Configuration objects shared by training, generation, and evaluation."""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, fields
from typing import Any


LEGACY_TASK_TYPES = (
    "code_generation",
    "code_debugging",
    "code_explanation",
    "algorithm_reasoning",
    "metacognitive_review",
)

TASK_TYPES = (
    *LEGACY_TASK_TYPES,
    "language_generation",
)

ERROR_CATEGORIES = (
    "none",
    "syntax",
    "logic",
    "hallucination",
    "incomplete",
    "unsafe",
)

ARCHITECTURES = ("legacy", "modern")
# Every attention layer holds a block_size x block_size mask, which is 1 GiB at this size.
MAX_BLOCK_SIZE = 32_768

MODEL_PRESETS = {
    "demo": dict(block_size=2048, n_layer=2, n_head=4, n_embd=128, architecture="modern"),
    "slm-50m": dict(block_size=2048, n_layer=12, n_head=8, n_embd=512, architecture="modern"),
    # 17 x 768 x 12 (head_dim 64) yields 160,721,679 parameters, sized for a Kaggle T4.
    "slm-160m": dict(block_size=2048, n_layer=17, n_head=12, n_embd=768, architecture="modern"),
    # 24 x 1,140 x 10 keeps 2,048-byte context and yields 499,524,075 parameters.
    "slm-500m": dict(block_size=2048, n_layer=24, n_head=10, n_embd=1140, architecture="modern"),
}


@dataclass(frozen=True)
class ModelConfig:
    """Small decoder-only transformer settings."""

    vocab_size: int = 259
    block_size: int = 256
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    architecture: str = "legacy"
    rope_theta: float = 10_000.0
    scaled_residual_init: bool = True
    task_types: tuple[str, ...] = TASK_TYPES
    error_categories: tuple[str, ...] = ERROR_CATEGORIES

    def validate(self) -> None:
        # A checkpoint's JSON config can carry 2.0 or true where an int belongs, and bool is an int subclass.
        for name in ("vocab_size", "block_size", "n_layer", "n_head", "n_embd"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer, got {value!r}")
        if self.vocab_size != 259:
            raise ValueError("vocab_size must be exactly 259 for the byte tokenizer")
        if not 8 <= self.block_size <= MAX_BLOCK_SIZE:
            raise ValueError(f"block_size must be between 8 and {MAX_BLOCK_SIZE}, got {self.block_size}")
        if self.n_layer < 1 or self.n_head < 1 or self.n_embd < 1:
            raise ValueError("n_layer, n_head, and n_embd must be positive")
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        if (isinstance(self.dropout, bool) or not isinstance(self.dropout, (int, float))
                or not 0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be a number in [0, 1), got {self.dropout!r}")
        for name in ("task_types", "error_categories"):
            labels = getattr(self, name)
            # Items are checked before set(), which raises TypeError on a nested list such as [["qa"]].
            if (not isinstance(labels, (tuple, list)) or not labels
                    or not all(isinstance(label, str) and label for label in labels) or len(set(labels)) != len(labels)):
                raise ValueError(f"{name} must be a non-empty list of distinct label names, got {labels!r}")
        if self.architecture not in ARCHITECTURES:
            raise ValueError(f"architecture must be one of {ARCHITECTURES}")
        if (isinstance(self.rope_theta, bool) or not isinstance(self.rope_theta, (int, float))
                or not 0.0 < self.rope_theta <= sys.float_info.max):
            raise ValueError(f"rope_theta must be a finite positive number, got {self.rope_theta!r}")
        if self.architecture == "modern" and (self.n_embd // self.n_head) % 2:
            raise ValueError("modern architecture requires an even attention head dimension")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ModelConfig":
        values = dict(raw)
        known = {item.name for item in fields(cls)}
        unknown = sorted(str(key) for key in values if key not in known)
        if unknown:
            raise ValueError(
                f"model config has unknown fields: {', '.join(unknown)}; "
                "the checkpoint may come from a newer cognition_slm, so upgrade before loading it"
            )
        # Historical checkpoints may omit these fields; never reinterpret their weights.
        values.setdefault("architecture", "legacy")
        values.setdefault("block_size", 256)
        # Configs saved before GPT-2 residual scaling existed were initialized unscaled.
        values.setdefault("scaled_residual_init", False)
        # Checkpoints created before language_generation used five task classes.
        for name, default in (("task_types", LEGACY_TASK_TYPES), ("error_categories", ERROR_CATEGORIES)):
            labels = values.get(name, default)
            # tuple("abc") would silently become three one-letter classes.
            if not isinstance(labels, (tuple, list)):
                raise ValueError(f"{name} must be a list of label names, got {labels!r}")
            values[name] = tuple(labels)
        config = cls(**values)
        config.validate()
        return config
