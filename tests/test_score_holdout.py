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
        # Trailing zeros do not change a number, so "$8.00" answers 8 and "3.50" is 3.5.
        self.assertEqual(normalize("He pays $8.00, or 3.50, or 10.0"), "he pays 8 or 3.5 or 10")
        # A leading dot is a decimal point, so ".5" is a half and never 5.
        self.assertEqual([normalize(text) for text in (".5", "-.5", "1.2.3")], ["0.5", "-0.5", "1.2 3"])
        rows = {row["id"]: row for row in json.loads((ROOT / "data/everyday_eval.json").read_text())["rows"]}
        self.assertEqual(self.module.score_answer(rows["everyday-v1-193"], "$8.00"), {"exact": True, "contains": True})
        self.assertEqual(self.module.score_answer(rows["everyday-v1-015"], ".5"), {"exact": False, "contains": False})
        # A sign flips the value, so "-25" must not match 25; a dash between words or numbers is not a sign.
        self.assertEqual(normalize("-25."), "-25")
        self.assertEqual(normalize("40 - 15 = 25, or 3-4"), "40 15 25 or 3 4")
        subtraction = {"id": "m", "category": "arithmetic", "expected_rubric": "25"}
        self.assertEqual(self.module.score_answer(subtraction, "-25"), {"exact": False, "contains": False})
        for minus in ("\u2212", "\u2013", "\uff0d"):
            with self.subTest(minus=hex(ord(minus))):
                self.assertEqual(self.module.score_answer(subtraction, f"{minus}25"), {"exact": False, "contains": False})
        self.assertEqual(normalize("pages 3\u20134"), "pages 3 4")
        carbon = {row["id"]: row for row in json.loads((ROOT / "data/everyday_eval.json").read_text())["rows"]}["everyday-v1-154"]
        self.assertEqual(self.module.score_answer(carbon, "CO\u2082."), {"exact": True, "contains": True})
        # Times written with dots or :00 match the plain form, and other minutes stay apart.
        self.assertEqual([normalize(text) for text in ("2 p.m.", "2:00 PM", "7 a.m", "2:30 pm")],
                         ["2 pm", "2 pm", "7 am", "2 30 pm"])
        # The same with no space after the digit.
        self.assertEqual([normalize(text) for text in ("2:00pm", "2p.m.", "3a.m", "2:00:30")], ["2 pm", "2 pm", "3 am", "2 30"])
        self.assertEqual(normalize("1,000 meters, or 2,500,000"), "1000 meters or 2500000")
        self.assertEqual(normalize("1,2, 3"), "1 2 3")
        thousand = {"id": "k", "category": "counting_time", "expected_rubric": "1000"}
        self.assertEqual(self.module.score_answer(thousand, "1,000"), {"exact": True, "contains": True})
        # Spelled numbers of 100 and above are one number, so "a hundred cents" still answers "100".
        self.assertEqual(normalize("A dollar is worth a hundred cents."), "a dollar is worth 100 cents")
        self.assertEqual(normalize("One hundred and forty-four"), "144")
        self.assertEqual(normalize("two thousand twenty-six, or a thousand"), "2026 or 1000")
        self.assertEqual(normalize("three four hundred, a hundred thousand"), "3 400 100000")
        self.assertEqual(normalize("a hundreds a"), "a hundreds a")
        self.assertEqual(normalize("One hundred. Ten decades make a century."), "100 10 decades make a century")
        self.assertEqual(normalize("Two hundred, three hundred; twenty-one"), "200 300 21")
        hundred = {"id": "c", "category": "counting_time", "expected_rubric": "100"}
        self.assertEqual(self.module.score_answer(hundred, "A dollar is worth a hundred cents."),
                         {"exact": False, "contains": True})
        self.assertEqual(self.module.score_answer({**hundred, "expected_rubric": "144"}, "one hundred forty-four"),
                         {"exact": True, "contains": True})

    def test_exact_and_whole_token_contains(self):
        row = {"id": "x", "category": "math", "expected_rubric": "3"}
        self.assertEqual(self.module.score_answer(row, "Three."), {"exact": True, "contains": True})
        self.assertEqual(self.module.score_answer(row, "The answer is 3"), {"exact": False, "contains": True})
        self.assertEqual(self.module.score_answer(row, "13"), {"exact": False, "contains": False})
        row = {"id": "y", "category": "fact", "expected_rubric": "cat", "accepted_answers": ["cat", "kitten"]}
        self.assertTrue(self.module.score_answer(row, "A kitten")["contains"])
        self.assertIsNone(self.module.score_answer({"category": "unknown", "expected_rubric": "x"}, "x"))

    def test_a_reply_made_only_of_accepted_answers_is_exact(self):
        # "I don't know. You haven't told me." joins two accepted refusals, yet it scored contains and never exact.
        refusal = {"id": "r", "category": "abstain", "expected_rubric": "I don't know",
                   "accepted_answers": ["I don't know", "you haven't told me"]}
        self.assertEqual(self.module.score_answer(refusal, "I don't know. You haven't told me."),
                         {"exact": True, "contains": True})
        self.assertFalse(self.module.score_answer({"id": "x", "category": "math", "expected_rubric": "3"}, "3. It is 4.")["exact"])

    def test_an_abstain_refusal_may_name_what_it_does_not_know(self):
        # The trained "I don't know your name. You haven't told me." scored contains but never exact.
        refusal = {"id": "r", "category": "abstain", "expected_rubric": "I don't know",
                   "accepted_answers": ["I don't know", "you haven't told me"]}
        for answer in ("I don't know your name. You haven't told me.", "I don't know what you ate.",
                       "I don't know where you are, so you haven't told me.", "I don't know who your best friend is."):
            with self.subTest(answer=answer):
                self.assertEqual(self.module.score_answer(refusal, answer), {"exact": True, "contains": True})
        for answer in ("I don't know, but it is Paris.", "I don't know. It is 42.", "I don't know your name, it's Sam.",
                       "I don't know your name. It's Sam.", "I don't know it is Paris.", "Sam. I don't know.",
                       "I don't know what you ate but I think it was pizza.", "I don't know your name is Sam.",
                       "I don't know what you are wearing maybe a red shirt."):
            with self.subTest(answer=answer):
                self.assertFalse(self.module.score_answer(refusal, answer)["exact"])
        # Other categories keep the sentence rule, so naming the object there still misses exact.
        other = {**refusal, "category": "fact"}
        self.assertEqual(self.module.score_answer(other, "I don't know your name. You haven't told me."),
                         {"exact": False, "contains": True})
        # The holdout's abstain questions stay with a human reviewer.
        holdout = {row["id"]: row for row in self.rows}
        for key in ("simple-v1-21", "simple-v1-22"):
            self.assertIsNone(self.module.score_answer(holdout[key], "I don't know. You haven't told me."))

    def test_everyday_keys_accept_the_common_forms_of_their_answer(self):
        # v1-143 took "the Atlantic Ocean" but not "Atlantic Ocean", and v1-185 took "five" but not "5 pm".
        everyday = {row["id"]: row for row in json.loads((ROOT / "data/everyday_eval.json").read_text())["rows"]}
        for key, answer in (("everyday-v1-143", "Atlantic Ocean"), ("everyday-v1-143", "The Atlantic."),
                            ("everyday-v1-185", "5 pm"), ("everyday-v1-185", "5 p.m.")):
            with self.subTest(key=key, answer=answer):
                self.assertTrue(self.module.score_answer(everyday[key], answer)["exact"])

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
        everyday = {row["id"]: row for row in json.loads((ROOT / "data/everyday_eval.json").read_text())["rows"]}
        for key, inflected in (("everyday-v1-063", "answered"), ("everyday-v1-067", "evens"), ("everyday-v1-075", "bottoms")):
            with self.subTest(key=key):
                self.assertEqual(everyday[key]["category"], "opposites")
                self.assertEqual(self.module.score_answer(everyday[key], inflected), {"exact": False, "contains": False})

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

    def test_reads_simple_questions_audit_report(self):
        # scripts/kaggle_simple_questions_audit.py stores each holdout row with its answer under "rows".
        audit = {"status": "complete_pending_manual_review",
                 "rows": [{**row, "answer": row["expected_rubric"]} for row in self.rows]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "simple_questions_audit.json"
            path.write_text(json.dumps(audit))
            predictions = self.module.load_predictions(path)
        report = self.module.score_predictions(self.rows, predictions)
        self.assertEqual(report["total"]["exact"], report["total"]["scored"])
        self.assertEqual(report["total"]["missing"], 0)

    def test_cli_rejects_empty_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            empty = Path(directory) / "predictions.jsonl"
            empty.write_text("\n")
            stdout, stderr = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                self.assertEqual(self.module.main([str(empty)]), 2)
        self.assertIn("no predictions", stderr.getvalue())
        self.assertEqual(stdout.getvalue(), "")

    def test_cli_rejects_a_holdout_without_a_rows_list(self):
        # A predictions list passed as --holdout used to end in a TypeError traceback instead of exit code 2.
        with tempfile.TemporaryDirectory() as directory:
            predictions = Path(directory) / "predictions.json"
            predictions.write_text(json.dumps([{"id": self.rows[0]["id"], "answer": "7"}]))
            numbers = Path(directory) / "numbers.json"
            numbers.write_text(json.dumps({"rows": 5}))
            for holdout in (predictions, numbers):
                stdout, stderr = io.StringIO(), io.StringIO()
                with self.subTest(holdout=holdout.name), contextlib.redirect_stdout(stdout), \
                        contextlib.redirect_stderr(stderr):
                    self.assertEqual(self.module.main([str(predictions), "--holdout", str(holdout)]), 2)
                    self.assertIn("expected a question file with a rows list", stderr.getvalue())
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
