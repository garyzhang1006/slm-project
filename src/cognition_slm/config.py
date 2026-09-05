"""Configuration objects shared by training, generation, and evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


TASK_TYPES = (
    "code_generation",
    "code_debugging",
    "code_explanation",
    "algorithm_reasoning",
    "metacognitive_review",
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

MODEL_PRESETS = {
    "demo": dict(block_size=2048, n_layer=2, n_head=4, n_embd=128, architecture="modern"),
    "slm-50m": dict(block_size=2048, n_layer=12, n_head=8, n_embd=512, architecture="modern"),
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
    task_types: tuple[str, ...] = TASK_TYPES
    error_categories: tuple[str, ...] = ERROR_CATEGORIES

    def validate(self) -> None:
        if self.vocab_size < 259:
            raise ValueError("vocab_size must include 256 byte tokens and 3 specials")
        if self.block_size < 8:
            raise ValueError("block_size must be at least 8")
        if self.n_layer < 1 or self.n_head < 1 or self.n_embd < 1:
            raise ValueError("n_layer, n_head, and n_embd must be positive")
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.architecture not in ARCHITECTURES:
            raise ValueError(f"architecture must be one of {ARCHITECTURES}")
        if self.rope_theta <= 0.0:
            raise ValueError("rope_theta must be positive")
        if self.architecture == "modern" and (self.n_embd // self.n_head) % 2:
            raise ValueError("modern architecture requires an even attention head dimension")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ModelConfig":
        values = dict(raw)
        # Historical checkpoints may omit these fields; never reinterpret their weights.
        values.setdefault("architecture", "legacy")
        values.setdefault("block_size", 256)
        values["task_types"] = tuple(values.get("task_types", TASK_TYPES))
        values["error_categories"] = tuple(values.get("error_categories", ERROR_CATEGORIES))
        config = cls(**values)
        config.validate()
        return config
