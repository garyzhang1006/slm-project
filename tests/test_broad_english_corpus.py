import unittest

from cognition_slm.data import format_prompt, validate_record
from cognition_slm.tokenizer import ByteTokenizer
from scripts.broad_english_corpus import paragraph_records


class BroadEnglishCorpusTests(unittest.TestCase):
    opening = "The village library opens every morning."
    remainder = (
        "Readers gather near the windows to study the history of their community. "
        "The librarian helps them find useful books and explains where each subject "
        "belongs. Everyone can borrow a book and return it the following week."
    )

    def raw(self, text=None, **changes):
        return dict({
            "id": "fixture-document", "text": text or self.opening + " " + self.remainder,
            "url": "https://example.org/library", "language": "en",
        }, **changes)

    def test_complete_text_and_provenance_are_preserved(self):
        row, = paragraph_records(self.raw())
        self.assertEqual(row["prompt"], "Continue this English passage:\n" + self.opening)
        self.assertEqual(row["answer"], self.remainder)
        self.assertEqual(row["source"], "HuggingFaceFW/fineweb-edu; https://example.org/library")
        self.assertEqual(row["license"], "ODC-BY-1.0 (database); underlying text rights retained")

    def test_token_budget_includes_format_and_special_tokens(self):
        raw = self.raw(self.opening + " " + self.remainder + " We also serve café guests.")
        row, = paragraph_records(raw)
        example = validate_record(row)
        size = len(ByteTokenizer().encode(format_prompt(example) + example.answer.strip()))
        self.assertEqual(paragraph_records(raw, max_bytes=size), [row])
        self.assertEqual(paragraph_records(raw, max_bytes=size - 1), [])
        self.assertEqual(paragraph_records(self.raw(self.opening + " " + self.remainder * 10)), [])

    def test_multiple_paragraphs_have_stable_distinct_ids_and_three_record_limit(self):
        paragraph = self.opening + " " + self.remainder
        for separator in ("\n", "\n\n"):
            raw = self.raw(separator.join([paragraph] * 5))
            rows = paragraph_records(raw)
            self.assertEqual(len(rows), 3)
            self.assertEqual(len({row["id"] for row in rows}), 3)
            self.assertEqual(rows, paragraph_records(raw))
            self.assertNotEqual(rows[0]["id"], paragraph_records(dict(raw, id="another"))[0]["id"])

    def test_blank_line_boundaries_preserve_wrapped_lines(self):
        paragraph = self.opening + "\n" + self.remainder.replace("The librarian", "\nThe librarian")
        row, = paragraph_records(self.raw(paragraph + "\n\nShort fragment"))
        self.assertEqual(row["answer"], self.remainder.replace("The librarian", "\nThe librarian"))

    def test_incomplete_short_and_non_english_text_is_rejected(self):
        for raw in (
            self.raw(language="fr"), self.raw("A short paragraph. Nothing else."),
            self.raw(self.opening + " " + self.remainder.rstrip(".")),
            self.raw("Hi. " + self.remainder), self.raw("x" * 181 + ". " + self.remainder),
            self.raw(id=""), self.raw(url=None), self.raw(language=None),
            self.raw(self.opening + "\x00 " + self.remainder), self.raw("\ud800"), {}, None,
        ):
            self.assertEqual(paragraph_records(raw), [])

    def test_quoted_sentence_boundaries_are_preserved(self):
        opening = '"The village library opens every morning."'
        row, = paragraph_records(self.raw(opening + " " + self.remainder + ' "Please visit!"'))
        self.assertTrue(row["prompt"].endswith(opening))
        self.assertTrue(row["answer"].endswith('"Please visit!"'))

    def test_secret_like_document_is_rejected_including_clean_paragraphs(self):
        paragraph = self.opening + " " + self.remainder
        raw = self.raw(paragraph + "\n\n" + "sk-" + "x" * 24)
        self.assertEqual(paragraph_records(raw), [])
        self.assertEqual(paragraph_records(self.raw(url="https://example.org/sk-" + "x" * 24)), [])

    def test_invalid_budgets_raise(self):
        for budget in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                paragraph_records(self.raw(), max_bytes=budget)


if __name__ == "__main__":
    unittest.main()
