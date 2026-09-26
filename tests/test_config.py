import unittest

from cognition_slm.config import MODEL_PRESETS, ModelConfig


class ConfigTests(unittest.TestCase):
    def test_vocabulary_matches_byte_tokenizer(self):
        for size in (258, 260, 512):
            with self.subTest(vocab_size=size), self.assertRaisesRegex(ValueError, "exactly 259"):
                ModelConfig(vocab_size=size).validate()

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
        self.assertEqual(len(config.task_types), 5)

    def test_500m_preset_has_expected_shape(self):
        config = ModelConfig(**MODEL_PRESETS["slm-500m"])
        self.assertEqual((config.n_layer, config.n_embd, config.n_head), (24, 1140, 10))
        self.assertEqual(len(config.task_types), 6)

    def test_160m_preset_has_expected_shape_and_head_dim(self):
        config = ModelConfig(**MODEL_PRESETS["slm-160m"])
        config.validate()
        self.assertEqual((config.n_layer, config.n_embd, config.n_head), (17, 768, 12))
        self.assertEqual(config.n_embd // config.n_head, 64)
        self.assertEqual(config.n_embd % 64, 0)

    def test_scaled_residual_init_defaults(self):
        self.assertTrue(ModelConfig().scaled_residual_init)
        self.assertFalse(ModelConfig.from_dict({"n_layer": 1, "n_head": 2, "n_embd": 16}).scaled_residual_init)
        self.assertTrue(ModelConfig.from_dict(ModelConfig().to_dict()).scaled_residual_init)
