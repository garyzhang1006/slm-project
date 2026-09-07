import unittest

from cognition_slm.data import format_prompt, validate_record
from cognition_slm.tokenizer import ByteTokenizer
from scripts.english_corpus import deterministic_split, dolly_record, story_record


class EnglishCorpusTests(unittest.TestCase):
    def test_story_preserves_complete_opening_and_remainder(self):
        opening = "One day a little rabbit found a red ball."
        remainder = "She took it home.\nHer friends came over to play."
        row = story_record(opening + "\n" + remainder, "fixture")
        self.assertEqual(row["prompt"], "Continue this story in English:\n" + opening)
        self.assertEqual(row["answer"], remainder)
        self.assertEqual(row["license"], "cdla-sharing-1.0")
        self.assertEqual(validate_record(row).task_type, "language_generation")

    def test_quoted_opening_keeps_closing_quote(self):
        row = story_record('"Shall we play outside today?" The children ran outside.', "quote")
        self.assertTrue(row["prompt"].endswith('today?"'))
        self.assertEqual(row["answer"], "The children ran outside.")

    def test_incomplete_or_invalid_stories_are_rejected(self):
        for text in ("No sentence boundary here", "Hi. A longer story follows.",
                     "A whole sentence with no remaining story.", "x" * 181 + ". More."):
            with self.subTest(text=text):
                self.assertIsNone(story_record(text, "invalid"))

    def test_budget_counts_formatting_utf8_and_special_tokens_without_truncation(self):
        raw = {"instruction": "Describe this word: café", "response": "A café serves coffee."}
        row = dolly_record(raw, 1)
        example = validate_record(row)
        size = len(ByteTokenizer().encode(format_prompt(example) + example.answer.strip()))
        self.assertEqual(dolly_record(raw, 1, max_bytes=size), row)
        self.assertIsNone(dolly_record(raw, 1, max_bytes=size - 1))
        self.assertIsNone(story_record(
            "A small dog went to the garden. " + "He played happily. " * 100, "long"
        ))

    def test_dolly_context_and_response_are_preserved(self):
        raw = {
            "instruction": "What did Sam buy?",
            "context": "Sam bought apples.\nThen he went home.",
            "response": "Sam bought apples.\n",
        }
        row = dolly_record(raw, 4)
        self.assertEqual(row["prompt"], raw["instruction"] + "\n\nContext:\n" + raw["context"])
        self.assertEqual(row["answer"], raw["response"])
        self.assertEqual(row["license"], "CC-BY-SA-3.0")
        self.assertEqual(dolly_record({"instruction": "Hello?", "response": "Hi!"}, 5)["prompt"], "Hello?")

    def test_malformed_dolly_rows_are_rejected(self):
        for raw in ({}, {"instruction": "", "response": "Hi"},
                    {"instruction": "Hello", "context": None, "response": "Hi"},
                    {"instruction": "Hello", "response": "\x00bad"}):
            self.assertIsNone(dolly_record(raw, 1))

    def test_split_deduplicates_normalized_prompts_and_is_order_independent(self):
        rows = [dolly_record({"instruction": f"Question {i}?", "response": f"Answer {i}."}, i)
                for i in range(8)]
        duplicate = dict(rows[0], id="duplicate", prompt="  QUESTION   0?\n", answer="Another answer.")
        rows.append(duplicate)
        train, evaluation = deterministic_split(rows, eval_count=3)
        self.assertEqual(len(train), 5)
        self.assertEqual(len(evaluation), 3)
        self.assertEqual((train, evaluation), deterministic_split(reversed(rows), eval_count=3))
        normalized = lambda row: " ".join(row["prompt"].casefold().split())
        self.assertTrue(set(map(normalized, train)).isdisjoint(map(normalized, evaluation)))

    def test_split_handles_empty_input_and_rejects_invalid_counts(self):
        self.assertEqual(deterministic_split([]), ([], []))
        for count in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                deterministic_split([], eval_count=count)


if __name__ == "__main__":
    unittest.main()
