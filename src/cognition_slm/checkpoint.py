"""Validated loading for model checkpoint payloads."""

from __future__ import annotations

from pathlib import Path

from .config import ModelConfig


def load_checkpoint_payload(torch, path: str | Path) -> tuple[dict, ModelConfig]:
    """Load a weights-only checkpoint and validate its model configuration."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must be a dictionary")
    required = {"model_config", "model_state_dict"}
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"checkpoint missing required fields: {sorted(missing)}")
    raw_config = checkpoint["model_config"]
    if not isinstance(raw_config, dict):
        raise ValueError("checkpoint model_config must be a dictionary")
    return checkpoint, ModelConfig.from_dict(raw_config)
