import ast
import unittest

from scripts.build_curriculum_data import build_rows


class CurriculumDataTests(unittest.TestCase):
    def test_splits_are_valid_and_ids_are_unique(self):
        train = build_rows("train")
        evaluation = build_rows("eval")
        self.assertGreater(len(train), 200)
        self.assertGreater(len(evaluation), 100)
        self.assertEqual(len({row["id"] for row in train}), len(train))
        self.assertEqual(len({row["id"] for row in evaluation}), len(evaluation))
        self.assertTrue({row["id"] for row in train}.isdisjoint(
            row["id"] for row in evaluation
        ))
        self.assertTrue({row["task_type"] for row in train} >= {
            "language_generation", "code_generation", "code_debugging",
            "code_explanation", "algorithm_reasoning",
        })

    def test_code_answers_parse_as_python(self):
        for row in build_rows("train") + build_rows("eval"):
            if row["task_type"] in {"code_generation", "code_debugging"}:
                ast.parse(row["answer"])
                compile(row["answer"], "<curriculum-answer>", "exec")

    def test_evaluation_prompts_are_not_training_prompts(self):
        def prompts(split):
            return {
                (row["task_type"], " ".join(row["prompt"].split()).casefold())
                for row in build_rows(split)
            }

        self.assertEqual(prompts("train") & prompts("eval"), set())

    def test_explicit_binary_signatures_match_answers(self):
        for split in ("train", "eval"):
            for row in build_rows(split):
                if "-python-binary-" in row["id"] and row["id"].endswith("-0"):
                    function = ast.parse(row["answer"]).body[0]
                    signature = f"{function.name}({', '.join(arg.arg for arg in function.args.args)})"
                    self.assertIn(signature, row["prompt"])


if __name__ == "__main__":
    unittest.main()
