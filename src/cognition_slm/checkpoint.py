"""Validated loading for model checkpoint payloads."""

from __future__ import annotations

from pathlib import Path
import zipfile

from .config import ModelConfig


def load_checkpoint_payload(torch, path: str | Path, *, inference_only: bool = False) -> tuple[dict, ModelConfig]:
    """Safely load checkpoints, lazily reading ZIP tensors during inference."""
    # Legacy non-ZIP checkpoints do not support mmap. Resume keeps eager loading.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True,
                            mmap=inference_only and zipfile.is_zipfile(path))
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must be a dictionary")
    required = {"model_config", "model_state_dict"}
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"checkpoint missing required fields: {sorted(missing)}")
    raw_config = checkpoint["model_config"]
    if not isinstance(raw_config, dict):
        raise ValueError("checkpoint model_config must be a dictionary")
    if not isinstance(checkpoint["model_state_dict"], dict):
        raise ValueError("checkpoint model_state_dict must be a dictionary")
    if inference_only:
        checkpoint = {key: checkpoint[key] for key in ("model_config", "model_state_dict", "metadata") if key in checkpoint}
    return checkpoint, ModelConfig.from_dict(raw_config)
