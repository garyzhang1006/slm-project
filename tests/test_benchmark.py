import unittest
from pathlib import Path

from cognition_slm.benchmark import _scalar_metrics, parse_model_spec


class BenchmarkTests(unittest.TestCase):
    def test_parse_model_spec_requires_label_and_path(self):
        self.assertEqual(parse_model_spec("modern=artifacts/modern.pt"), ("modern", Path("artifacts/modern.pt")))
        with self.assertRaises(ValueError):
            parse_model_spec("artifacts/modern.pt")

    def test_scalar_metrics_include_static_code_metrics_when_present(self):
        result = {
            "task_accuracy": 0.5,
            "error_accuracy": 0.75,
            "confidence_bucket_accuracy": 0.25,
            "exact_match_accuracy": 0.0,
            "code_metrics": {
                "syntax_validity": 0.5,
                "required_symbol_recall": 0.25,
                "static_score": 0.375,
            },
        }
        metrics = _scalar_metrics(result)
        self.assertEqual(metrics["code_static_score"], 0.375)
        self.assertEqual(metrics["task_accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
