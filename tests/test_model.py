import unittest


try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None, "PyTorch not installed")
class ModelTests(unittest.TestCase):
    def test_forward_shapes_and_loss(self):
        from cognition_slm.config import ModelConfig
        from cognition_slm.model import CognitionSLM

        config = ModelConfig(block_size=32, n_layer=1, n_head=2, n_embd=16)
        model = CognitionSLM(config)
        input_ids = torch.randint(3, config.vocab_size, (2, 12))
        mask = torch.ones_like(input_ids)
        output = model(
            input_ids,
            attention_mask=mask,
            task_labels=torch.tensor([0, 1]),
            error_labels=torch.tensor([0, 2]),
            confidence_labels=torch.tensor([1, 2]),
        )
        self.assertEqual(tuple(output.logits.shape), (2, 12, config.vocab_size))
        self.assertEqual(tuple(output.task_logits.shape), (2, len(config.task_types)))
        self.assertIsNotNone(output.loss)
        self.assertTrue(torch.isfinite(output.loss))


if __name__ == "__main__":
    unittest.main()
