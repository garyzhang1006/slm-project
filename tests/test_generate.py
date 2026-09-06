import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from cognition_slm.generate import generate_ids
from cognition_slm.tokenizer import ByteTokenizer


class BatchedGenerationTests(unittest.TestCase):
    def test_invalid_temperature_rejected_before_model_execution(self):
        model = Mock()
        for temperature in (-1, float("nan"), float("inf"), float("-inf"), True, "warm", 10 ** 500):
            with self.subTest(temperature=temperature), self.assertRaisesRegex(ValueError, "finite, non-negative"):
                generate_ids(model, None, ByteTokenizer(), temperature=temperature)
        model.eval.assert_not_called()
        model.assert_not_called()

    def test_finished_rows_stay_at_eos_until_all_rows_finish(self):
        tokenizer = ByteTokenizer()

        class ScriptedModel:
            config = SimpleNamespace(block_size=32)

            def __init__(self):
                self.calls = 0

            def eval(self):
                return self

            def __call__(self, ids, attention_mask=None):
                self.calls += 1
                logits = torch.full((*ids.shape, tokenizer.vocab_size), float("-inf"))
                tokens = [tokenizer.eos_id, 100] if self.calls == 1 else [101, tokenizer.eos_id]
                for row, token in enumerate(tokens):
                    logits[row, -1, token] = 0.0
                return SimpleNamespace(logits=logits)

        for temperature in (0, 1):
            with self.subTest(temperature=temperature):
                model = ScriptedModel()
                result = generate_ids(
                    model, torch.tensor([[tokenizer.bos_id], [tokenizer.bos_id]]),
                    tokenizer, max_new_tokens=5, temperature=temperature, top_k=0,
                )
                self.assertEqual(model.calls, 2)
                self.assertEqual(result.tolist(), [
                    [tokenizer.bos_id, tokenizer.eos_id, tokenizer.eos_id],
                    [tokenizer.bos_id, 100, tokenizer.eos_id],
                ])


if __name__ == "__main__":
    unittest.main()
