import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from cognition_slm.config import ModelConfig
from cognition_slm.generate import generate_ids, strip_stop_sequence
from cognition_slm.model import CognitionSLM
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

    def test_tiny_positive_temperature_decodes_greedily_instead_of_nan(self):
        tokenizer = ByteTokenizer()
        torch.manual_seed(19)
        model = CognitionSLM(ModelConfig(block_size=16, n_layer=1, n_head=2, n_embd=16)).eval()
        with torch.no_grad():
            model.lm_head.weight.mul_(1000)
        ids = torch.tensor([[tokenizer.bos_id, 45, 77]])
        expected = generate_ids(model, ids, tokenizer, max_new_tokens=4, temperature=0)
        for temperature in (1e-300, 1e-30, 1e-6):
            with self.subTest(temperature=temperature):
                actual = generate_ids(model, ids, tokenizer, max_new_tokens=4,
                                      temperature=temperature, top_k=0)
                self.assertEqual(actual.tolist(), expected.tolist())


class _FixedLogitsModel:
    """Return the same final-position logits on every call, via the uncached path."""

    def __init__(self, values, block_size=64):
        self.config = SimpleNamespace(block_size=block_size)
        self.values = values
        self.calls = 0

    def eval(self):
        return self

    def __call__(self, ids, attention_mask=None):
        self.calls += 1
        logits = torch.full((*ids.shape, ByteTokenizer().vocab_size), -1e4)
        for token, value in self.values.items():
            logits[:, -1, token] = value
        return SimpleNamespace(logits=logits)


class _ScriptedTokensModel(_FixedLogitsModel):
    """Emit one scripted token per call for every row."""

    def __init__(self, tokens, block_size=64):
        super().__init__({}, block_size)
        self.tokens = tokens

    def __call__(self, ids, attention_mask=None):
        self.values = {self.tokens[min(self.calls, len(self.tokens) - 1)]: 0.0}
        return super().__call__(ids, attention_mask)


class DecodingOptionTests(unittest.TestCase):
    def test_new_options_default_to_previous_sampling(self):
        tokenizer = ByteTokenizer()
        torch.manual_seed(3)
        model = CognitionSLM(ModelConfig(block_size=16, n_layer=1, n_head=2, n_embd=16)).eval()
        ids = torch.tensor([[tokenizer.bos_id, 45, 77], [tokenizer.bos_id, 31, 42]])
        for use_cache in (False, True):
            with self.subTest(use_cache=use_cache):
                torch.manual_seed(11)
                expected = generate_ids(model, ids, tokenizer, max_new_tokens=10, use_cache=use_cache)
                torch.manual_seed(11)
                actual = generate_ids(model, ids, tokenizer, max_new_tokens=10, use_cache=use_cache,
                                      top_p=1.0, repetition_penalty=1.0, stop_sequences=None)
                self.assertEqual(actual.tolist(), expected.tolist())

    def test_invalid_new_options_rejected_before_model_execution(self):
        model = Mock()
        for key, value in (
            ("top_p", 0), ("top_p", 1.5), ("top_p", float("nan")), ("top_p", True), ("top_p", "0.9"),
            ("repetition_penalty", 0), ("repetition_penalty", -1.0),
            ("repetition_penalty", float("inf")), ("repetition_penalty", 10 ** 500),
            ("stop_sequences", "\n"), ("stop_sequences", [""]), ("stop_sequences", [1]),
        ):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                generate_ids(model, None, ByteTokenizer(), **{key: value})
        model.eval.assert_not_called()

    def test_top_p_keeps_only_the_nucleus(self):
        tokenizer = ByteTokenizer()
        # Probabilities are about 0.6, 0.3 and 0.1 for tokens 100, 101 and 102.
        model = _FixedLogitsModel({100: 0.0, 101: -0.6931, 102: -1.7918})
        ids = torch.full((64, 1), tokenizer.bos_id)
        torch.manual_seed(5)
        unrestricted = generate_ids(model, ids, tokenizer, max_new_tokens=1, temperature=1, top_k=0)
        self.assertIn(102, unrestricted[:, 1].tolist())
        for top_p, allowed in ((0.5, {100}), (0.8, {100, 101})):
            with self.subTest(top_p=top_p):
                torch.manual_seed(5)
                result = generate_ids(model, ids, tokenizer, max_new_tokens=1, temperature=1,
                                      top_k=0, top_p=top_p)
                self.assertEqual(set(result[:, 1].tolist()), allowed)

    def test_repetition_penalty_applies_to_generated_tokens_only(self):
        tokenizer = ByteTokenizer()
        model = _FixedLogitsModel({100: 2.0, 101: 1.5, 102: -1.0})
        # Token 100 in the prompt must not be penalized.
        ids = torch.tensor([[tokenizer.bos_id, 100, 100]])
        plain = generate_ids(model, ids, tokenizer, max_new_tokens=3, temperature=0)
        self.assertEqual(plain[0, 3:].tolist(), [100, 100, 100])
        penalized = generate_ids(model, ids, tokenizer, max_new_tokens=3, temperature=0,
                                 repetition_penalty=2.0)
        # 100 drops to 1.0 < 1.5, then 101 drops to 0.75 < 1.0.
        self.assertEqual(penalized[0, 3:].tolist(), [100, 101, 100])

    def test_repetition_penalty_pushes_negative_logits_further_down(self):
        tokenizer = ByteTokenizer()
        model = _FixedLogitsModel({100: -1.0, 101: -1.5})
        ids = torch.tensor([[tokenizer.bos_id]])
        result = generate_ids(model, ids, tokenizer, max_new_tokens=2, temperature=0,
                              repetition_penalty=2.0)
        self.assertEqual(result[0, 1:].tolist(), [100, 101])

    def test_stop_sequence_ends_generation_after_its_last_byte(self):
        tokenizer = ByteTokenizer()
        script = tokenizer.encode("Paris.\nMore", add_bos=False, add_eos=False)
        model = _ScriptedTokensModel(script)
        ids = torch.tensor([[tokenizer.bos_id]])
        result = generate_ids(model, ids, tokenizer, max_new_tokens=20, temperature=0,
                              stop_sequences=["\n\n", ".\n"])
        self.assertEqual(tokenizer.decode(result[0, 1:].tolist()), "Paris.\n")
        self.assertEqual(model.calls, 7)

    def test_strip_removes_the_longest_matching_stop(self):
        self.assertEqual(strip_stop_sequence("xab", ["b", "ab"]), "x")
        self.assertEqual(strip_stop_sequence("xab", None), "xab")

    def test_stop_sequence_ignores_prompt_text(self):
        tokenizer = ByteTokenizer()
        model = _ScriptedTokensModel(tokenizer.encode("ab", add_bos=False, add_eos=False))
        ids = torch.tensor([tokenizer.encode("x\n", add_eos=False)])
        result = generate_ids(model, ids, tokenizer, max_new_tokens=2, temperature=0,
                              stop_sequences=["\n"])
        self.assertEqual(tokenizer.decode(result[0, ids.size(1):].tolist()), "ab")

    def test_stopped_row_pads_with_eos_while_other_rows_continue(self):
        tokenizer = ByteTokenizer()

        class RowModel(_FixedLogitsModel):
            def __call__(self, ids, attention_mask=None):
                self.calls += 1
                logits = torch.full((*ids.shape, tokenizer.vocab_size), -1e4)
                logits[0, -1, tokenizer.encode(";", add_bos=False, add_eos=False)[0]] = 0.0
                logits[1, -1, 100 if self.calls < 3 else tokenizer.eos_id] = 0.0
                return SimpleNamespace(logits=logits)

        semicolon = tokenizer.encode(";", add_bos=False, add_eos=False)[0]
        result = generate_ids(RowModel({}), torch.tensor([[1], [1]]), tokenizer,
                              max_new_tokens=5, temperature=0, stop_sequences=[";"])
        self.assertEqual(result.tolist(), [
            [1, semicolon, tokenizer.eos_id, tokenizer.eos_id],
            [1, 100, 100, tokenizer.eos_id],
        ])


if __name__ == "__main__":
    unittest.main()
