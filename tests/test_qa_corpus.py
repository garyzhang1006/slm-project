import unittest

from cognition_slm.data import format_prompt, validate_record
from cognition_slm.tokenizer import ByteTokenizer
from scripts.qa_corpus import answer_scores, normalized_answer, pilot_improved, squad_record


def fixture(context="Lena placed her book on a shelf.", answer="a shelf"):
    return {
        "id": "fixture", "title": "A book", "context": context,
        "question": "Where did Lena put her book?",
        "answers": {"text": [answer], "answer_start": [context.index(answer)]},
    }


class QACorpusTests(unittest.TestCase):
    def test_squad_preserves_full_passage_and_reference(self):
        context = "Before lunch, Lena read.\nLena placed her book on a shelf. Then she left."
        row = squad_record(fixture(context))
        self.assertEqual(row["prompt"],
                         "Answer the question using the passage. Reply with only the answer."
                         f"\n\nPassage: {context}\n\nQuestion: Where did Lena put her book?")
        self.assertEqual(row["answer"], "a shelf")
        self.assertEqual(row["id"], "squad-fixture")
        self.assertEqual(row["source"], "rajpurkar/squad")
        self.assertEqual(row["license"], "CC-BY-SA-4.0")
        self.assertEqual(validate_record(row).task_type, "language_generation")

    def test_malformed_rows_are_rejected(self):
        raw = fixture()
        invalid = [None, {}, dict(raw, id=" "), dict(raw, context=None),
                   dict(raw, question=""), dict(raw, answers=None)]
        for answers in ({}, {"text": [], "answer_start": []},
                        {"text": "a shelf", "answer_start": [24]},
                        {"text": ["a shelf"], "answer_start": "24"},
                        {"text": ["a shelf"], "answer_start": []},
                        {"text": [None], "answer_start": [0]},
                        {"text": [""], "answer_start": [0]},
                        {"text": ["a shelf"], "answer_start": [False]},
                        {"text": ["a shelf"], "answer_start": [-1]},
                        {"text": ["a shelf"], "answer_start": [1.5]}):
            invalid.append(dict(raw, answers=answers))
        for item in invalid:
            with self.subTest(raw=item):
                self.assertIsNone(squad_record(item))

    def test_reference_offset_must_match_original_characters(self):
        raw = fixture("Léna placed her book on a shelf.")
        self.assertIsNotNone(squad_record(raw))
        raw["answers"]["answer_start"][0] += 1
        self.assertIsNone(squad_record(raw))
        raw["answers"]["answer_start"][0] = 10000
        self.assertIsNone(squad_record(raw))

    def test_long_context_is_rejected_instead_of_cropped_near_answer(self):
        raw = fixture("Background detail. " * 100 + "Lena placed her book on a shelf.")
        self.assertIsNone(squad_record(raw))

    def test_budget_includes_utf8_prompt_formatting_and_special_tokens(self):
        raw = fixture("Léna placed her book on a shelf.")
        row = squad_record(raw)
        example = validate_record(row)
        size = len(ByteTokenizer().encode(format_prompt(example) + example.answer.strip()))
        self.assertEqual(squad_record(raw, max_bytes=size), row)
        self.assertIsNone(squad_record(raw, max_bytes=size - 1))
        for budget in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                squad_record(raw, max_bytes=budget)

    def test_answer_length_limit_counts_utf8_bytes(self):
        self.assertIsNotNone(squad_record(fixture("é" * 48, "é" * 48)))
        self.assertIsNone(squad_record(fixture("é" * 49, "é" * 49)))

    def test_normalization_uses_standard_squad_rules(self):
        self.assertEqual(normalized_answer("  The CAT, an apple; A dog!\n"), "cat apple dog")
        self.assertEqual(normalized_answer("theater can't co-op"), "theater cant coop")
        self.assertEqual(normalized_answer("A café—THE café"), "café— café")

    def test_exact_match_uses_best_reference(self):
        self.assertEqual(answer_scores("The shelf!", ["table", "a shelf"]),
                         {"exact_match": 1.0, "token_f1": 1.0})
        self.assertEqual(answer_scores("table", ["shelf"]),
                         {"exact_match": 0.0, "token_f1": 0.0})

    def test_f1_counts_repeated_tokens_and_extra_words(self):
        result = answer_scores("red red blue", ["red blue", "green"])
        self.assertEqual(result["exact_match"], 0.0)
        self.assertAlmostEqual(result["token_f1"], 0.8)
        self.assertAlmostEqual(answer_scores("red", ["red red"])["token_f1"], 2 / 3)

    def test_empty_predictions_and_references_have_explicit_scores(self):
        self.assertEqual(answer_scores("", ["the"]), {"exact_match": 1.0, "token_f1": 1.0})
        self.assertEqual(answer_scores("", ["shelf"]), {"exact_match": 0.0, "token_f1": 0.0})
        self.assertEqual(answer_scores("shelf", []), {"exact_match": 0.0, "token_f1": 0.0})

    def test_pilot_requires_meaningful_gain(self):
        baseline = {"exact_match": 0.0, "token_f1": 0.0}
        self.assertFalse(pilot_improved(baseline, baseline))
        self.assertFalse(pilot_improved(baseline, {"exact_match": 0.019, "token_f1": 0.049}))
        self.assertTrue(pilot_improved(baseline, {"exact_match": 0.02, "token_f1": 0.02}))
        self.assertTrue(pilot_improved(baseline, {"exact_match": 0.0, "token_f1": 0.05}))

    def test_f1_gain_does_not_override_exact_match_regression(self):
        baseline = {"exact_match": 0.2, "token_f1": 0.3}
        self.assertFalse(pilot_improved(baseline, {"exact_match": 0.19, "token_f1": 0.9}))
        self.assertTrue(pilot_improved(baseline, {"exact_match": 0.2, "token_f1": 0.4}))
        self.assertTrue(pilot_improved(baseline, {"exact_match": 0.23, "token_f1": 0.25}))

    def test_pilot_rejects_invalid_or_missing_metrics_on_either_side(self):
        valid = {"exact_match": 0.0, "token_f1": 0.0}
        invalid = [None, {}, {"exact_match": 0.0}]
        for key in ("exact_match", "token_f1"):
            for value in (float("nan"), float("inf"), -0.1, 1.1, True, "0.5", None):
                invalid.append(dict(valid, **{key: value}))
        for metrics in invalid:
            with self.subTest(metrics=metrics):
                with self.assertRaises(ValueError):
                    pilot_improved(metrics, valid)
                with self.assertRaises(ValueError):
                    pilot_improved(valid, metrics)


if __name__ == "__main__":
    unittest.main()
