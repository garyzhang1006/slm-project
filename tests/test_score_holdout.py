import contextlib
import importlib
import io
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def scorer():
    # Kaggle packages only the scripts a runner needs, so skip when the scorer is absent.
    if not (ROOT / "scripts" / "score_holdout.py").exists():
        raise unittest.SkipTest("score_holdout.py is not packaged here")
    return importlib.import_module("scripts.score_holdout")


class ScoreHoldoutTests(unittest.TestCase):
    def setUp(self):
        self.module = scorer()
        self.rows = json.loads((ROOT / "data/simple_questions_holdout.json").read_text())["rows"]

    def test_normalization(self):
        normalize = self.module.normalize_answer
        self.assertEqual(normalize("  Seven. "), "7")
        self.assertEqual(normalize("Twenty-one"), "21")
        self.assertEqual(normalize("twenty"), "20")
        self.assertEqual(normalize("Apple,  banana, PEAR!"), "apple banana pear")
        self.assertEqual(normalize("She doesn’t."), "she doesnt")
        self.assertEqual(normalize("3.5 cups"), "3.5 cups")
        # A sign flips the value, so "-25" must not match 25; a dash between words or numbers is not a sign.
        self.assertEqual(normalize("-25."), "-25")
        self.assertEqual(normalize("40 - 15 = 25, or 3-4"), "40 15 25 or 3 4")
        subtraction = {"id": "m", "category": "arithmetic", "expected_rubric": "25"}
        self.assertEqual(self.module.score_answer(subtraction, "-25"), {"exact": False, "contains": False})
        self.assertEqual(normalize("1,000 meters, or 2,500,000"), "1000 meters or 2500000")
        self.assertEqual(normalize("1,2, 3"), "1 2 3")
        thousand = {"id": "k", "category": "counting_time", "expected_rubric": "1000"}
        self.assertEqual(self.module.score_answer(thousand, "1,000"), {"exact": True, "contains": True})

    def test_exact_and_whole_token_contains(self):
        row = {"id": "x", "category": "math", "expected_rubric": "3"}
        self.assertEqual(self.module.score_answer(row, "Three."), {"exact": True, "contains": True})
        self.assertEqual(self.module.score_answer(row, "The answer is 3"), {"exact": False, "contains": True})
        self.assertEqual(self.module.score_answer(row, "13"), {"exact": False, "contains": False})
        row = {"id": "y", "category": "fact", "expected_rubric": "cat", "accepted_answers": ["cat", "kitten"]}
        self.assertTrue(self.module.score_answer(row, "A kitten")["contains"])
        self.assertIsNone(self.module.score_answer({"category": "unknown", "expected_rubric": "x"}, "x"))

    def test_inflected_answers_match_except_where_form_is_tested(self):
        horse = {"id": "h", "category": "colors_animals", "expected_rubric": "neigh",
                 "accepted_answers": ["neigh", "whinny"]}
        self.assertTrue(self.module.score_answer(horse, "A horse neighs.")["contains"])
        self.assertTrue(self.module.score_answer(horse, "whinnies")["exact"])
        self.assertFalse(self.module.score_answer(horse, "moos")["contains"])
        plural = {"id": "p", "category": "plurals", "expected_rubric": "mice"}
        self.assertFalse(self.module.score_answer(plural, "mices")["contains"])
        english = {"id": "e", "category": "english", "expected_rubric": "book"}
        self.assertFalse(self.module.score_answer(english, "books")["exact"])
        # "Write only the word window." tests the exact form too.
        copy = next(row for row in self.rows if row["id"] == "simple-v1-17")
        self.assertEqual((copy["category"], copy["expected_rubric"]), ("instruction", "window"))
        self.assertFalse(self.module.score_answer(copy, "windows")["contains"])
        short = {"id": "s", "category": "yes_no", "expected_rubric": "no"}
        self.assertFalse(self.module.score_answer(short, "nos")["exact"])

    def test_real_holdout_totals(self):
        expected = {row["id"]: row["expected_rubric"] for row in self.rows}
        predictions = [{"id": key, "answer": value.upper() + "."} for key, value in expected.items()]
        predictions[0] = {"id": predictions[0]["id"], "answer": "wrong"}
        report = self.module.score_predictions(self.rows, predictions[:-1])
        manual = sum(row["category"] == "unknown" for row in self.rows)
        scored = len(self.rows) - manual
        self.assertEqual(report["total"]["scored"], scored)
        self.assertEqual(report["total"]["manual_review"], manual)
        self.assertEqual(report["total"]["missing"], 1)
        self.assertEqual(report["total"]["exact"], scored - 2)
        self.assertEqual(sum(bucket["scored"] for bucket in report["categories"].values()), scored)
        self.assertIsNone(report["categories"]["unknown"]["exact_accuracy"])
        self.assertEqual(report["unknown_ids"], [])

    def test_unanswered_manual_rows_count_as_missing(self):
        predictions = [{"id": row["id"], "answer": row["expected_rubric"]}
                       for row in self.rows if row["category"] != "unknown"]
        report = self.module.score_predictions(self.rows, predictions)
        manual = sum(row["category"] == "unknown" for row in self.rows)
        self.assertEqual(report["categories"]["unknown"]["missing"], manual)
        self.assertEqual(report["categories"]["unknown"]["manual_review"], manual)
        self.assertEqual(report["total"]["missing"], manual)

    def test_rejects_bad_predictions(self):
        for predictions in ([{"id": "a"}], [{"id": 1, "answer": "x"}],
                            [{"id": "a", "answer": "x"}, {"id": "a", "answer": "y"}]):
            with self.subTest(predictions=predictions), self.assertRaises(ValueError):
                self.module.score_predictions(self.rows, predictions)

    def test_cli_reads_jsonl_and_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            jsonl = Path(directory) / "predictions.jsonl"
            jsonl.write_text("".join(json.dumps({"id": row["id"], "answer": row["expected_rubric"]}) + "\n"
                                     for row in self.rows) + json.dumps({"id": "extra", "answer": "x"}) + "\n")
            report = Path(directory) / "report.json"
            report.write_text(json.dumps({"simple_questions": [{"id": self.rows[0]["id"], "answer": "7"}]}))
            single = Path(directory) / "single.jsonl"
            single.write_text(json.dumps({"id": self.rows[0]["id"], "answer": "7"}) + "\n")
            self.assertEqual(self.module.load_predictions(single), [{"id": self.rows[0]["id"], "answer": "7"}])
            stdout, stderr = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                self.assertEqual(self.module.main([str(jsonl)]), 0)
                self.assertEqual(self.module.main([str(report), "--json"]), 0)
                self.assertEqual(self.module.main([str(Path(directory) / "missing.json")]), 2)
            lines = stdout.getvalue().splitlines()
            total = next(line for line in lines if line.startswith("TOTAL"))
            self.assertIn("exact 22/22 (100.0%)", total)
            self.assertIn("extra", stderr.getvalue())
            self.assertIn("error:", stderr.getvalue())

    def test_jsonl_answer_keeps_raw_line_separator(self):
        # json.dumps(ensure_ascii=False) leaves U+2028 raw, and str.splitlines() would cut the record there.
        predictions = [{"id": "a", "answer": "one\u2028two"}, {"id": "b", "answer": "three"}]
        with tempfile.TemporaryDirectory() as directory:
            jsonl = Path(directory) / "predictions.jsonl"
            jsonl.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions),
                             encoding="utf-8")
            self.assertEqual(self.module.load_predictions(jsonl), predictions)

    def test_cli_rejects_empty_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            empty = Path(directory) / "predictions.jsonl"
            empty.write_text("\n")
            stdout, stderr = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                self.assertEqual(self.module.main([str(empty)]), 2)
        self.assertIn("no predictions", stderr.getvalue())
        self.assertEqual(stdout.getvalue(), "")

    def test_cli_scores_another_file_with_categories(self):
        holdout = ROOT / "data/everyday_eval.json"
        rows = json.loads(holdout.read_text())["rows"]
        with tempfile.TemporaryDirectory() as directory:
            # A stage5 report holds both lists; --holdout picks the matching one.
            report = Path(directory) / "eval_report.json"
            report.write_text(json.dumps({
                "simple_questions": [{"id": self.rows[0]["id"], "answer": "7"}],
                "everyday_eval": [{"id": row["id"], "answer": row["accepted_answers"][-1]} for row in rows]}))
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(self.module.main([str(report), "--holdout", str(holdout)]), 0)
        lines = stdout.getvalue().splitlines()
        self.assertIn(f"exact {len(rows)}/{len(rows)} (100.0%)", lines[-1])
        categories = {row["category"] for row in rows}
        self.assertEqual({line.split()[0] for line in lines[:-1]}, categories)


if __name__ == "__main__":
    unittest.main()
