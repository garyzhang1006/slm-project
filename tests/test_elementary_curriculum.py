import json
import re
import unittest

from scripts.elementary_curriculum import LICENSE, SOURCE, build_curriculum, normalize_prompt
from cognition_slm.data import validate_record


class ElementaryCurriculumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train, cls.dev, cls.manifest = build_curriculum(set())

    def test_deterministic_grouped_splits(self):
        self.assertEqual((self.train, self.dev, self.manifest), build_curriculum(set()))
        self.assertGreaterEqual(len(self.train), 12000)
        self.assertLessEqual(len(self.train), 20000)
        self.assertGreaterEqual(len(self.dev), 128)
        train_groups = {row["id"].split("-")[2] for row in self.train}
        dev_groups = {row["id"].split("-")[2] for row in self.dev}
        self.assertTrue(train_groups.isdisjoint(dev_groups))
        self.assertEqual(train_groups, set(self.manifest["groups"]["train"]))
        self.assertEqual(dev_groups, set(self.manifest["groups"]["dev"]))
        self.assertEqual(set(self.manifest["counts"]["dev"]), {"arithmetic", "attribute", "location", "transfer", "copy", "sort", "grammar"})

    def test_canonical_unique_bounded_records(self):
        rows = self.train + self.dev
        self.assertEqual(len(rows), len({row["id"] for row in rows}))
        self.assertEqual(len(rows), len({normalize_prompt(row["prompt"]) for row in rows}))
        for row in rows:
            validate_record(row)
            self.assertEqual(row["source"], SOURCE)
            self.assertEqual(row["license"], LICENSE)
            self.assertEqual(row["task_type"], "language_generation")
            self.assertLessEqual(len(json.dumps(row).encode()), 512)
            self.assertFalse(any(ord(c) < 32 or ord(c) == 127 for c in row["prompt"] + row["answer"]))

    def test_reserved_normalization_excludes_both_splits(self):
        selected = [self.train[0]["prompt"], self.dev[0]["prompt"]]
        train, dev, manifest = build_curriculum({"  " + p.upper().replace(" ", "  ") for p in selected})
        self.assertEqual(manifest["reserved_prompts_excluded"], 2)
        self.assertTrue({normalize_prompt(p) for p in selected}.isdisjoint(normalize_prompt(r["prompt"]) for r in train + dev))

    def test_all_arithmetic_answers_and_commutative_groups(self):
        group_by_case = {}
        operators = set()
        for row in self.train + self.dev:
            if "-arithmetic-" not in row["id"]:
                continue
            a, operation, b = re.search(r"(\d+) ([+*/-]|plus|minus|times|divided by) (\d+)", row["prompt"]).groups()
            operation = {"plus": "+", "minus": "-", "times": "*", "divided by": "/"}.get(operation, operation)
            a, b = int(a), int(b)
            expected = {"+": a + b, "-": a - b, "*": a * b, "/": a // b}[operation]
            if operation == "/":
                self.assertEqual(a % b, 0)
            self.assertIn(row["answer"], (str(expected), f"The answer is {expected}."))
            operators.add(operation)
            operands = tuple(sorted((a, b))) if operation in {"+", "*"} else (a, b)
            key = (operation, operands)
            group = row["id"].split("-")[2]
            self.assertEqual(group_by_case.setdefault(key, group), group)
        self.assertEqual(operators, {"+", "-", "*", "/"})


if __name__ == "__main__":
    unittest.main()
