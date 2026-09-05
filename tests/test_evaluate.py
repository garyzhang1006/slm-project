import unittest


try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None, "PyTorch not installed")
class EvaluationTests(unittest.TestCase):
    def test_exact_match_preserves_python_structure(self):
        from cognition_slm.evaluate import _normalize

        self.assertNotEqual(_normalize("def f():\n    return 1"), _normalize("def f():\nreturn 1"))
        self.assertEqual(_normalize("```python\ndef f():\n    return 1\n```"), "def f():\n    return 1")

    def test_classification_metrics_report_confusion_and_calibration(self):
        from cognition_slm.evaluate import classification_metrics

        logits = torch.tensor([[4.0, 0.0], [0.0, 4.0], [3.0, 1.0], [1.0, 3.0]])
        targets = torch.tensor([0, 1, 1, 0])
        metrics = classification_metrics(logits, targets, calibration_bins=4)

        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["macro_f1"], 0.5)
        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [1, 1]])
        self.assertGreater(metrics["ece"], 0.0)
        self.assertLess(metrics["ece"], 1.0)


if __name__ == "__main__":
    unittest.main()
