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
        self.assertTrue({row["task_type"] for row in train} >= {
            "language_generation", "code_generation", "code_debugging",
            "code_explanation", "algorithm_reasoning",
        })

    def test_code_answers_parse_as_python(self):
        for row in build_rows("train") + build_rows("eval"):
            if row["task_type"] in {"code_generation", "code_debugging"}:
                ast.parse(row["answer"])


if __name__ == "__main__":
    unittest.main()
