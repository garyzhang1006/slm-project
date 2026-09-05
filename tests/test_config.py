import unittest

from cognition_slm.config import MODEL_PRESETS, ModelConfig


class ConfigTests(unittest.TestCase):
    def test_presets_are_valid_2048_modern_models(self):
        for preset in MODEL_PRESETS.values():
            config = ModelConfig(**preset)
            config.validate()
            self.assertEqual(config.block_size, 2048)
            self.assertEqual(config.architecture, "modern")

    def test_historical_checkpoint_defaults_preserved(self):
        config = ModelConfig.from_dict({"n_layer": 1, "n_head": 2, "n_embd": 16})
        self.assertEqual(config.architecture, "legacy")
        self.assertEqual(config.block_size, 256)
