import unittest
from pathlib import Path
from unittest.mock import patch

from cognition_slm.benchmark import _scalar_metrics, benchmark, main, parse_model_spec


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

    def test_scalar_metrics_skip_unscorable_task_accuracy(self):
        result = {"task_accuracy": None, "error_accuracy": 1.0,
                  "confidence_bucket_accuracy": 1.0, "exact_match_accuracy": 0.0}
        self.assertNotIn("task_accuracy", _scalar_metrics(result))

    def test_cli_loads_each_checkpoint_once_on_requested_device(self):
        import torch

        model = torch.nn.Linear(1, 1)
        model.config = type("Config", (), {"architecture": "legacy"})()
        result = {"task_accuracy": 1.0, "task_records": 1, "error_accuracy": 1.0,
                  "confidence_bucket_accuracy": 1.0, "exact_match_accuracy": 1.0}
        # Not cpu, which benchmark() also falls back to, so a dropped --device would fail this.
        argv = ["benchmark", "--model", "a=a.pt", "--data", "d.jsonl", "--device", "meta"]
        with patch("sys.argv", argv), \
                patch("cognition_slm.benchmark.load_jsonl", return_value=[]), \
                patch("cognition_slm.benchmark.load_checkpoint", return_value=(model, None)) as load, \
                patch("cognition_slm.benchmark.evaluate", return_value=result), \
                patch("builtins.print"):
            main()
        load.assert_called_once_with(Path("a.pt"), torch.device("meta"))

    def test_benchmark_defaults_to_cpu(self):
        import torch

        model = torch.nn.Linear(1, 1)
        model.config = type("Config", (), {"architecture": "legacy"})()
        result = {"task_accuracy": 1.0, "task_records": 1, "error_accuracy": 1.0,
                  "confidence_bucket_accuracy": 1.0, "exact_match_accuracy": 1.0}
        with patch("cognition_slm.benchmark.load_jsonl", return_value=[]), \
                patch("cognition_slm.benchmark.load_checkpoint", return_value=(model, None)) as load, \
                patch("cognition_slm.benchmark.evaluate", return_value=result):
            benchmark([("a", Path("a.pt"))], "d.jsonl", 1)
        load.assert_called_once_with(Path("a.pt"), torch.device("cpu"))

    def test_deltas_skip_task_accuracy_a_legacy_reference_could_not_score(self):
        import torch

        model = torch.nn.Linear(1, 1)
        model.config = type("Config", (), {"architecture": "legacy"})()
        legacy = {"task_accuracy": None, "task_records": 0, "error_accuracy": 0.5,
                  "confidence_bucket_accuracy": 0.5, "exact_match_accuracy": 0.0}
        modern = {"task_accuracy": 0.9, "task_records": 2, "error_accuracy": 0.75,
                  "confidence_bucket_accuracy": 0.5, "exact_match_accuracy": 0.5}
        with patch("cognition_slm.benchmark.load_jsonl", return_value=[]), \
                patch("cognition_slm.benchmark.load_checkpoint", return_value=(model, None)), \
                patch("cognition_slm.benchmark.evaluate", side_effect=[legacy, modern]):
            result = benchmark([("old", Path("a.pt")), ("new", Path("b.pt"))], "d.jsonl", 1)
        delta = result["deltas"][0]["metrics"]
        # A missing metric must not be treated as 0.0, which would fake a +0.9 gain.
        self.assertNotIn("task_accuracy", delta)
        self.assertEqual(delta["error_accuracy"], 0.25)


if __name__ == "__main__":
    unittest.main()
