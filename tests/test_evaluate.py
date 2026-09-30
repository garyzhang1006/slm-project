import math
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
        # code_eval reads a py fence as Python, so exact match strips it too.
        self.assertEqual(_normalize("```py\ndef f():\n    return 1\n```"), "def f():\n    return 1")

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

    def test_calibration_error_weights_each_bin_by_its_absolute_gap(self):
        # Two records sure at 0.6 and right, two sure at 0.9 and wrong: 0.5 * 0.4 + 0.5 * 0.9. Signed gaps
        # would cancel to 0.25, and dropping the bin weight would give 1.3.
        from cognition_slm.evaluate import classification_metrics

        logits = torch.tensor([[math.log(1.5), 0.0]] * 2 + [[math.log(9.0), 0.0]] * 2)
        metrics = classification_metrics(logits, torch.tensor([0, 0, 1, 1]))
        self.assertAlmostEqual(metrics["ece"], 0.65, places=5)

    def test_macro_f1_ignores_classes_absent_from_targets_and_predictions(self):
        from cognition_slm.evaluate import classification_metrics

        logits = torch.tensor([[4.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 4)
        metrics = classification_metrics(logits, torch.zeros(4, dtype=torch.long))
        self.assertEqual(metrics["macro_f1"], 1.0)

    def _tiny_model(self, **overrides):
        from cognition_slm.config import ModelConfig
        from cognition_slm.model import CognitionSLM
        from cognition_slm.tokenizer import ByteTokenizer

        torch.manual_seed(0)
        config = ModelConfig(block_size=256, n_layer=1, n_head=2, n_embd=16, **overrides)
        return CognitionSLM(config), ByteTokenizer()

    def _example(self, **overrides):
        from cognition_slm.data import validate_record

        return validate_record({
            "id": "q", "prompt": "Say hi.", "answer": "hi", "task_type": "language_generation",
            "confidence": 0.5, "error_category": "none", "source": "test", "license": "CC0-1.0",
            **overrides,
        })

    def test_legacy_checkpoint_skips_task_head_for_unknown_task_label(self):
        from cognition_slm.config import LEGACY_TASK_TYPES
        from cognition_slm.evaluate import evaluate

        model, tokenizer = self._tiny_model(task_types=LEGACY_TASK_TYPES)
        examples = [self._example(), self._example(id="c", task_type="code_generation")]
        result = evaluate(model, tokenizer, examples, max_new_tokens=1)
        self.assertEqual(result["records"], 2)
        self.assertEqual(result["task_records"], 1)
        self.assertEqual(sum(map(sum, result["task_metrics"]["confusion_matrix"])), 1)

        result = evaluate(model, tokenizer, examples[:1], max_new_tokens=1)
        self.assertIsNone(result["task_accuracy"])
        self.assertIsNone(result["task_metrics"])

    def test_over_length_prompt_fails_before_generation(self):
        from unittest.mock import patch
        from cognition_slm.evaluate import evaluate

        model, tokenizer = self._tiny_model()
        examples = [self._example(id="short"), self._example(id="long", prompt="x" * 300)]
        with patch("cognition_slm.evaluate.generate_text") as generate:
            with self.assertRaisesRegex(ValueError, "block_size 256: long;"):
                evaluate(model, tokenizer, examples, max_new_tokens=1)
        generate.assert_not_called()

    def test_prompt_filling_block_size_exactly_fails_before_generation(self):
        from unittest.mock import patch
        from cognition_slm.data import format_prompt
        from cognition_slm.evaluate import evaluate

        model, tokenizer = self._tiny_model()
        base = len(tokenizer.encode(format_prompt(self._example(prompt="x")), add_eos=False))
        exact = self._example(id="exact", prompt="x" * (256 - base + 1))
        self.assertEqual(len(tokenizer.encode(format_prompt(exact), add_eos=False)), 256)
        # generate_text rejects a prompt that fills the window, so the pre-check must too.
        with patch("cognition_slm.evaluate.generate_text") as generate:
            with self.assertRaisesRegex(ValueError, "block_size 256: exact;"):
                evaluate(model, tokenizer, [self._example(id="short"), exact], max_new_tokens=1)
        generate.assert_not_called()

    def test_cli_rejects_bad_arguments_before_loading_the_checkpoint(self):
        from io import StringIO
        from unittest.mock import patch
        from cognition_slm.evaluate import main

        cases = [
            (["--max-new-tokens", "0"], {"return_value": [self._example()]}, "--max-new-tokens must be positive"),
            ([], {"side_effect": FileNotFoundError("eval.jsonl")}, "eval.jsonl"),
        ]
        for extra, loaded, message in cases:
            with self.subTest(message=message):
                argv = ["evaluate", "--checkpoint", "model.pt", "--data", "eval.jsonl", *extra]
                stderr = StringIO()
                with patch("sys.argv", argv), patch("sys.stderr", stderr), \
                        patch("cognition_slm.evaluate.load_jsonl", **loaded), \
                        patch("cognition_slm.evaluate.load_checkpoint") as load:
                    with self.assertRaises(SystemExit):
                        main()
                load.assert_not_called()
                self.assertIn(message, stderr.getvalue())

    def test_cli_reports_checkpoint_load_errors_without_a_traceback(self):
        from io import StringIO
        from unittest.mock import patch
        from cognition_slm.evaluate import main

        argv = ["evaluate", "--checkpoint", "model.pt", "--data", "eval.jsonl", "--device", "cpu"]
        stderr = StringIO()
        with patch("sys.argv", argv), patch("sys.stderr", stderr), \
                patch("cognition_slm.evaluate.load_jsonl", return_value=[self._example()]), \
                patch("cognition_slm.evaluate.load_checkpoint",
                      side_effect=ValueError("checkpoint must be a dictionary")):
            with self.assertRaises(SystemExit):
                main()
        self.assertIn("checkpoint must be a dictionary", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
