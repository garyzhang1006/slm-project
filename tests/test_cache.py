import unittest
from unittest.mock import patch

import torch

from cognition_slm.config import ModelConfig
from cognition_slm.generate import generate_ids, score_generated_ids
from cognition_slm.model import CognitionSLM
from cognition_slm.tokenizer import ByteTokenizer


class CacheTests(unittest.TestCase):
    def model(self, architecture="modern"):
        torch.manual_seed(19)
        return CognitionSLM(ModelConfig(block_size=8, n_layer=2, n_head=2,
                                       n_embd=16, architecture=architecture)).eval()

    def test_chunk_and_single_token_logits_match_full_forward(self):
        for architecture in ("legacy", "modern"):
            with self.subTest(architecture=architecture):
                model = self.model(architecture)
                ids = torch.randint(3, 259, (2, 8))
                expected = model(ids).logits
                first, cache = model.forward_inference(ids[:, :3])
                middle, cache = model.forward_inference(ids[:, 3:7], cache)
                last, cache = model.forward_inference(ids[:, 7:], cache, last_only=True)
                actual = torch.cat((first, middle, last), dim=1)
                torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
                self.assertEqual(cache[0][0].shape, (2, 2, 8, 8))
                self.assertFalse(actual.requires_grad)

    def test_generation_matches_uncached_before_and_after_overflow(self):
        tokenizer = ByteTokenizer()
        for architecture in ("legacy", "modern"):
            model = self.model(architecture)
            # Suppress EOS so the test must cross the rolling-window boundary.
            with torch.no_grad():
                model.lm_head.weight[tokenizer.eos_id].zero_()
            ids = torch.tensor([[1, 45, 77], [1, 31, 42]])
            expected = generate_ids(model, ids, tokenizer, max_new_tokens=12,
                                    temperature=0, use_cache=False)
            actual = generate_ids(model, ids, tokenizer, max_new_tokens=12,
                                  temperature=0, use_cache=True)
            torch.testing.assert_close(actual, expected)
            self.assertGreater(actual.size(1), model.config.block_size)

    def test_inference_skips_losses_heads_and_does_not_change_checkpoint(self):
        model = self.model()
        keys = set(model.state_dict())
        ids = torch.tensor([[1, 20, 30]])
        with patch.object(model.task_head, "forward", side_effect=AssertionError), \
             patch.object(model.error_head, "forward", side_effect=AssertionError), \
             patch.object(model.confidence_head, "forward", side_effect=AssertionError), \
             patch("torch.nn.functional.cross_entropy", side_effect=AssertionError):
            logits, cache = model.forward_inference(ids, last_only=True)
            model.forward_inference(ids[:, :1], cache)
        self.assertEqual(logits.shape, (1, 1, 259))
        self.assertEqual(set(model.state_dict()), keys)
        self.model().load_state_dict(model.state_dict(), strict=True)

    def test_cache_rejects_invalid_shape_length_dtype_and_training(self):
        model = self.model()
        ids = torch.tensor([[1, 20, 30]])
        _, cache = model.forward_inference(ids)
        invalid = [(), ((cache[0][0],),) * 2,
                   ((cache[0][0][:, :, :0], cache[0][1][:, :, :0]),) * 2,
                   (cache[0], (cache[1][0][:, :, :2], cache[1][1][:, :, :2])),
                   tuple((key.double(), value.double()) for key, value in cache)]
        for item in invalid:
            with self.assertRaises(ValueError):
                model.forward_inference(ids[:, :1], item)
        with self.assertRaisesRegex(ValueError, "exceeds block_size"):
            model.forward_inference(ids.repeat(1, 2), cache)
        with self.assertRaisesRegex(ValueError, "non-empty"):
            model.forward_inference(ids[:, :0])
        with self.assertRaisesRegex(ValueError, "model.eval"):
            model.train().forward_inference(ids)

    def test_request_caches_are_independent_and_scores_match(self):
        model = self.model()
        ids = torch.tensor([[1, 20, 30, 40, 50, 60, 70, 80, 90, 100]])
        _, cache = model.forward_inference(ids[:, :3])
        saved = cache[0][0].clone()
        model.forward_inference(ids[:, 4:7])
        model.forward_inference(ids[:, 3:4], cache)
        torch.testing.assert_close(cache[0][0], saved)
        total = 0.0
        for position in range(3, ids.size(1)):
            logits = model(ids[:, max(0, position - 8):position]).logits[:, -1]
            total += float(logits.log_softmax(-1)[0, ids[0, position]])
        self.assertAlmostEqual(score_generated_ids(model, ids, 3), total / 7, places=5)

    def test_cached_batch_finished_rows_remain_at_eos(self):
        model = self.model()
        tokenizer = ByteTokenizer()
        original = model.forward_inference
        calls = []

        def scripted(ids, cache=None, **kwargs):
            logits, updated = original(ids, cache, **kwargs)
            calls.append(ids.size(1))
            logits.fill_(float("-inf"))
            tokens = [2, 100] if len(calls) == 1 else [101, 2]
            for row, token in enumerate(tokens):
                logits[row, -1, token] = 0
            return logits, updated

        with patch.object(model, "forward_inference", side_effect=scripted):
            actual = generate_ids(model, torch.tensor([[1], [1]]), tokenizer,
                                  max_new_tokens=5, temperature=0)
        self.assertEqual(actual.tolist(), [[1, 2, 2], [1, 100, 2]])
        self.assertEqual(calls, [1, 1])


if __name__ == "__main__":
    unittest.main()
