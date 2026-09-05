import unittest

from cognition_slm.generate import generate_text, rank_candidate_indices


class GenerationTests(unittest.TestCase):
    def test_oversized_prompt_rejected_before_model_execution(self):
        from types import SimpleNamespace
        from cognition_slm.tokenizer import ByteTokenizer

        model = SimpleNamespace(config=SimpleNamespace(block_size=2048))
        with self.assertRaisesRegex(ValueError, "shorten the prompt"):
            generate_text(model, ByteTokenizer(), "x" * 2048)

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
