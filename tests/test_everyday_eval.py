import importlib
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compute import short_facts  # noqa: E402

EVERYDAY = ROOT / "data/everyday_eval.json"
HOLDOUT = ROOT / "data/simple_questions_holdout.json"
ROW_KEYS = {"id", "category", "prompt", "expected_rubric", "task_type"}


def scorer():
    # Kaggle packages only the scripts a runner needs, so skip when the scorer is absent.
    if not (ROOT / "scripts" / "score_holdout.py").exists():
        raise unittest.SkipTest("score_holdout.py is not packaged here")
    return importlib.import_module("scripts.score_holdout")


def template_keys(tokens: list[str]) -> set[tuple]:
    """One key per token position with that token wildcarded, so prompts differing in one slot collide."""
    masked = ["#" if token.replace(".", "").isdigit() else token for token in tokens]
    return {(len(masked), index, *masked[:index], *masked[index + 1:]) for index in range(len(masked))}


class EverydayEvalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = scorer()
        cls.document = json.loads(EVERYDAY.read_text(encoding="utf-8"))
        cls.rows = cls.document["rows"]
        cls.holdout = json.loads(HOLDOUT.read_text(encoding="utf-8"))["rows"]
        cls.facts = short_facts.short_fact_rows()

    def normalized(self, prompts) -> set[str]:
        return {self.module.normalize_answer(prompt) for prompt in prompts}

    def test_schema_matches_the_holdout(self):
        self.assertEqual(self.document["purpose"], "evaluation_only")
        self.assertEqual(self.document["license"], "CC0-1.0")
        self.assertTrue(200 <= len(self.rows) <= 300, len(self.rows))
        for row in self.rows:
            with self.subTest(row=row.get("id")):
                self.assertLessEqual(ROW_KEYS, set(row))
                self.assertTrue(all(isinstance(row[key], str) and row[key].strip() for key in ROW_KEYS))
                self.assertEqual(row["task_type"], "language_generation")
                # The eval runner stops generation at a newline, so prompts stay on one line.
                self.assertNotIn("\n", row["prompt"])
                self.assertNotIn(row["category"], self.module.MANUAL_CATEGORIES)
        self.assertGreaterEqual(len({row["category"] for row in self.rows}), 10)

    def test_ids_and_prompts_are_unique(self):
        self.assertEqual(len({row["id"] for row in self.rows}), len(self.rows))
        self.assertEqual(len(self.normalized(row["prompt"] for row in self.rows)), len(self.rows))

    def test_every_row_has_accepted_answers_led_by_the_rubric(self):
        for row in self.rows:
            with self.subTest(row=row["id"]):
                answers = row["accepted_answers"]
                self.assertTrue(answers and all(isinstance(answer, str) for answer in answers))
                self.assertEqual(answers[0], row["expected_rubric"])
                self.assertTrue(all(self.module.normalize_answer(answer) for answer in answers))

    def test_scorer_accepts_each_rows_first_answer(self):
        for row in self.rows:
            with self.subTest(row=row["id"]):
                self.assertEqual(self.module.score_answer(row, row["accepted_answers"][0]),
                                 {"exact": True, "contains": True})
        predictions = [{"id": row["id"], "answer": row["accepted_answers"][0]} for row in self.rows]
        report = self.module.score_predictions(self.rows, predictions)
        self.assertEqual(report["total"]["exact"], len(self.rows))
        self.assertEqual(report["total"]["manual_review"], 0)

    def test_prompts_do_not_contain_their_answer(self):
        # Contains-match would credit a model that only echoes the prompt. Reading passages, yes/no
        # questions and "X or Y" choices name the answer by design.
        for row in self.rows:
            if row["category"] in ("reading", "yes_no") or " or " in row["prompt"]:
                continue
            prompt = f" {self.module.normalize_answer(row['prompt'])} "
            for answer in self.module.accepted_answers(row):
                with self.subTest(row=row["id"], answer=answer):
                    self.assertNotIn(f" {answer} ", prompt)

    def test_no_prompt_overlap_with_holdout_or_short_facts(self):
        everyday = self.normalized(row["prompt"] for row in self.rows)
        holdout = self.normalized(row["prompt"] for row in self.holdout)
        facts = self.normalized(row["prompt"] for row in self.facts)
        self.assertEqual(everyday & holdout, set())
        self.assertEqual(everyday & facts, set())
        self.assertEqual(holdout & facts, set())

    def test_no_shared_question_template_with_short_facts_or_holdout(self):
        # Two prompts share a template when they match after masking numbers and one more word,
        # as "What is the capital of Peru?" does with "What is the capital of France?".
        index = {}
        for source, prompts in (("short_facts", [row["prompt"] for row in self.facts]),
                                ("holdout", [row["prompt"] for row in self.holdout])):
            for prompt in prompts:
                for key in template_keys(self.module.normalize_answer(prompt).split()):
                    index.setdefault(key, (source, prompt))
        clashes = []
        for row in self.rows:
            for key in template_keys(self.module.normalize_answer(row["prompt"]).split()):
                if key in index:
                    clashes.append((row["id"], row["prompt"], *index[key]))
                    break
        self.assertEqual(clashes, [])


if __name__ == "__main__":
    unittest.main()
