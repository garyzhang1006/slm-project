import tempfile
import unittest
from pathlib import Path

import torch

from cognition_slm.checkpoint import load_checkpoint_payload
from cognition_slm.config import ModelConfig


class CheckpointTests(unittest.TestCase):
    def test_inference_preserves_weights_and_drops_training_state(self):
        payload = {"model_config": ModelConfig().to_dict(),
                   "model_state_dict": {"weight": torch.arange(6)},
                   "optimizer_state_dict": {"moment": torch.ones(6)},
                   "metadata": {"step": 12}}
        with tempfile.TemporaryDirectory() as directory:
            for zip_format in (True, False):
                with self.subTest(zip_format=zip_format):
                    path = Path(directory) / "checkpoint.pt"
                    torch.save(payload, path, _use_new_zipfile_serialization=zip_format)
                    inference, _ = load_checkpoint_payload(torch, path, inference_only=True)
                    self.assertNotIn("optimizer_state_dict", inference)
                    self.assertEqual(inference["metadata"]["step"], 12)
                    torch.testing.assert_close(inference["model_state_dict"]["weight"], payload["model_state_dict"]["weight"])
                    resumed, _ = load_checkpoint_payload(torch, path)
                    self.assertIn("optimizer_state_dict", resumed)

    def test_invalid_weight_mapping_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.pt"
            torch.save({"model_config": ModelConfig().to_dict(), "model_state_dict": []}, path)
            with self.assertRaisesRegex(ValueError, "model_state_dict must be a dictionary"):
                load_checkpoint_payload(torch, path, inference_only=True)
