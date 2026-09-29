import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from cognition_slm.data import format_prompt, validate_record
from cognition_slm.generate import generate_text, main, rank_candidate_indices
from cognition_slm.tokenizer import ByteTokenizer


def _prompt_length(prompt, task_type="code_generation"):
    record = validate_record({
        "id": "generation", "prompt": prompt, "answer": "placeholder", "task_type": task_type,
        "confidence": 0.5, "error_category": "none", "source": "runtime", "license": "runtime",
    })
    return len(ByteTokenizer().encode(format_prompt(record), add_eos=False))


class GenerationTests(unittest.TestCase):
    def test_oversized_prompt_rejected_before_model_execution(self):
        model = SimpleNamespace(config=SimpleNamespace(block_size=2048))
        with self.assertRaisesRegex(ValueError, "shorten the prompt"):
            generate_text(model, ByteTokenizer(), "x" * 2048)

    def test_prompt_filling_block_size_rejected(self):
        block_size = _prompt_length("hello")
        model = SimpleNamespace(config=SimpleNamespace(block_size=block_size))
        with self.assertRaisesRegex(ValueError, "shorten the prompt"):
            generate_text(model, ByteTokenizer(), "hello")

    def test_max_new_tokens_clamped_to_remaining_context(self):
        block_size = _prompt_length("hello") + 3
        parameter = torch.nn.Parameter(torch.zeros(1))
        model = SimpleNamespace(config=SimpleNamespace(block_size=block_size),
                                parameters=lambda: iter([parameter]))
        with patch("cognition_slm.generate.generate_ids",
                   side_effect=lambda _m, ids, _t, **kw: ids) as generate:
            generate_text(model, ByteTokenizer(), "hello", max_new_tokens=96)
        self.assertEqual(generate.call_args.kwargs["max_new_tokens"], 3)

    def test_cli_defaults_to_language_generation(self):
        argv = ["cognition-slm-generate", "--checkpoint", "x.pt", "--prompt", "hi"]
        with patch("sys.argv", argv), \
             patch("cognition_slm.generate.load_checkpoint", return_value=(None, None)), \
             patch("cognition_slm.generate.generate_text", return_value="") as generate, \
             patch("builtins.print"):
            main()
        self.assertEqual(generate.call_args.kwargs["task_type"], "language_generation")

    def test_stop_sequence_is_removed_from_returned_text(self):
        parameter = torch.nn.Parameter(torch.zeros(1))
        model = SimpleNamespace(config=SimpleNamespace(block_size=2048),
                                parameters=lambda: iter([parameter]))
        answer = ByteTokenizer().encode(" Paris.\n", add_bos=False, add_eos=False)

        def output(_m, ids, _t, **kwargs):
            return torch.cat([ids, torch.tensor([answer])], dim=1)

        with patch("cognition_slm.generate.generate_ids", side_effect=output) as generate:
            text = generate_text(model, ByteTokenizer(), "Capital of France?", top_p=0.9,
                                 repetition_penalty=1.2, stop_sequences=["\n"])
        self.assertEqual(text, "Paris.")
        self.assertEqual(generate.call_args.kwargs["top_p"], 0.9)
        self.assertEqual(generate.call_args.kwargs["repetition_penalty"], 1.2)
        self.assertEqual(generate.call_args.kwargs["stop_sequences"], ["\n"])

    def test_cli_passes_decoding_flags(self):
        argv = ["cognition-slm-generate", "--checkpoint", "x.pt", "--prompt", "hi", "--top-p", "0.9",
                "--repetition-penalty", "1.1", "--stop", "\n", "--stop", "###"]
        with patch("sys.argv", argv), \
             patch("cognition_slm.generate.load_checkpoint", return_value=(None, None)), \
             patch("cognition_slm.generate.generate_text", return_value="") as generate, \
             patch("builtins.print"):
            main()
        kwargs = generate.call_args.kwargs
        self.assertEqual((kwargs["top_p"], kwargs["repetition_penalty"], kwargs["stop_sequences"]),
                         (0.9, 1.1, ["\n", "###"]))

    def test_cli_defaults_leave_new_decoding_options_off(self):
        argv = ["cognition-slm-generate", "--checkpoint", "x.pt", "--prompt", "hi"]
        with patch("sys.argv", argv), \
             patch("cognition_slm.generate.load_checkpoint", return_value=(None, None)), \
             patch("cognition_slm.generate.generate_text", return_value="") as generate, \
             patch("builtins.print"):
            main()
        kwargs = generate.call_args.kwargs
        self.assertEqual((kwargs["top_p"], kwargs["repetition_penalty"], kwargs["stop_sequences"]),
                         (1.0, 1.0, None))

    def test_cli_rejects_invalid_decoding_flags(self):
        for extra in (["--top-p", "0"], ["--top-p", "1.5"], ["--repetition-penalty", "0"],
                      ["--repetition-penalty", "inf"], ["--repetition-penalty", "nan"], ["--stop", ""],
                      ["--temperature", "-1"], ["--temperature", "nan"], ["--temperature", "inf"], ["--top-k", "-1"],
                      ["--max-new-tokens", "0"], ["--task-type", "poetry"]):
            argv = ["cognition-slm-generate", "--checkpoint", "x.pt", "--prompt", "hi", *extra]
            with self.subTest(extra=extra), patch("sys.argv", argv), \
                 patch("cognition_slm.generate.load_checkpoint") as load, \
                 patch("sys.stderr"), self.assertRaises(SystemExit):
                main()
            load.assert_not_called()

    def test_syntax_bonus_can_prefer_valid_code(self):
        texts = ["def add(a, b)\n    return a + b", "def add(a, b):\n    return a + b"]
        self.assertEqual(
            rank_candidate_indices(texts, [-0.1, -0.3], "code_generation", 0.5),
            1,
        )

    def test_prose_ranking_uses_model_score_only(self):
        texts = ["first", "second"]
        self.assertEqual(rank_candidate_indices(texts, [-0.3, -0.1], "code_explanation", 0.5), 1)


if __name__ == "__main__":
    unittest.main()
