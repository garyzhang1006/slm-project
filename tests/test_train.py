import unittest

from cognition_slm.train import _lr_scale


class TrainingTests(unittest.TestCase):
    def test_warmup_and_cosine_schedule_boundaries(self):
        self.assertAlmostEqual(_lr_scale(0, total_steps=20, warmup_steps=5), 0.2)
        self.assertAlmostEqual(_lr_scale(4, total_steps=20, warmup_steps=5), 1.0)
        self.assertAlmostEqual(_lr_scale(19, total_steps=20, warmup_steps=5), 0.0)
        self.assertAlmostEqual(_lr_scale(0, total_steps=20, warmup_steps=0), 1.0)


class TrainingIntegrationTests(unittest.TestCase):
    def setUp(self):
        import json
        import tempfile
        from pathlib import Path

        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.data = self.root / "train.jsonl"
        examples = [
            {"id": str(index), "prompt": "Write Python.", "answer": answer,
             "task_type": "code_generation", "confidence": 0.8,
             "error_category": "none", "source": "test", "license": "CC0-1.0"}
            for index, answer in enumerate(("return 1", "def example():\n    return 123456789"))
        ]
        self.data.write_text("\n".join(json.dumps(item) for item in examples))

    def args(self, name="model.pt", **changes):
        from cognition_slm.train import build_parser

        args = build_parser().parse_args([
            "--data", str(self.data), "--out", str(self.root / name),
            "--device", "cpu", "--steps", "3", "--warmup-steps", "0",
            "--block-size", "256", "--n-layer", "1", "--n-head", "2",
            "--n-embd", "16", "--save-every", "1", "--batch-size", "2",
        ])
        for key, value in changes.items():
            setattr(args, key, value)
        return args

    def test_periodic_checkpoint_resume_matches_uninterrupted_dropout(self):
        import torch
        from unittest.mock import patch
        from cognition_slm.train import _atomic_save, train

        first = self.root / "first.pt"

        def capture(torch_module, payload, path):
            _atomic_save(torch_module, payload, path)
            if payload["metadata"]["step"] == 1:
                first.write_bytes(path.read_bytes())

        with patch("cognition_slm.train._atomic_save", side_effect=capture):
            result = train(self.args(dropout=0.1))
        resumed = train(self.args("resumed.pt", resume=str(first), block_size=2048))
        self.assertEqual(resumed["resumed_from_step"], 1)
        self.assertEqual(resumed["block_size"], 256)
        self.assertGreater(result["parameter_count"], 0)
        final = torch.load(result["checkpoint"], weights_only=True)
        restored = torch.load(resumed["checkpoint"], weights_only=True)
        for name, weight in final["model_state_dict"].items():
            torch.testing.assert_close(weight, restored["model_state_dict"][name], rtol=0, atol=0)
        self.assertIn("torch_rng_state", restored)
        self.assertEqual(len(restored["optimizer_state_dict"]["param_groups"]), 2)

    def test_accumulation_matches_combined_batch_with_unequal_lengths(self):
        import torch
        from cognition_slm.train import train

        combined = train(self.args("combined.pt", steps=1, batch_size=4))
        accumulated = train(self.args(
            "accumulated.pt", steps=1, batch_size=1, gradient_accumulation_steps=4,
        ))
        self.assertEqual(accumulated["effective_batch_size"], 4)
        self.assertAlmostEqual(combined["final_loss"], accumulated["final_loss"], places=5)
        left = torch.load(combined["checkpoint"], weights_only=True)["model_state_dict"]
        right = torch.load(accumulated["checkpoint"], weights_only=True)["model_state_dict"]
        for name in left:
            torch.testing.assert_close(left[name], right[name], rtol=1e-4, atol=2e-6)

    def test_cpu_rejects_cuda_precision(self):
        from cognition_slm.train import train

        for precision in ("fp16", "bf16"):
            with self.assertRaisesRegex(ValueError, "require a CUDA device"):
                train(self.args(precision=precision))

    def test_runtime_rejects_invalid_accumulation_and_save_interval(self):
        from cognition_slm.train import train

        for changes in ({"gradient_accumulation_steps": 0}, {"save_every": 0}):
            with self.assertRaisesRegex(ValueError, "must be positive"):
                train(self.args(**changes))

    def test_atomic_save_failure_preserves_checkpoint(self):
        from unittest.mock import Mock
        from cognition_slm.train import _atomic_save

        destination = self.root / "existing.pt"
        destination.write_bytes(b"old checkpoint")
        fake_torch = Mock()
        fake_torch.save.side_effect = OSError("disk full")
        with self.assertRaisesRegex(OSError, "disk full"):
            _atomic_save(fake_torch, {}, destination)
        self.assertEqual(destination.read_bytes(), b"old checkpoint")
        self.assertEqual(list(self.root.glob(".existing.pt.*")), [])

    def test_validation_loss_does_not_depend_on_batch_size(self):
        import torch
        from cognition_slm.config import ModelConfig
        from cognition_slm.data import encode_examples, load_jsonl
        from cognition_slm.model import CognitionSLM
        from cognition_slm.tokenizer import ByteTokenizer
        from cognition_slm.train import _validation_summary

        encoded = encode_examples(load_jsonl(self.data), ByteTokenizer(), 256)
        model = CognitionSLM(ModelConfig(
            block_size=256, n_layer=1, n_head=2, n_embd=16, architecture="modern",
        ))
        one = _validation_summary(torch, model, encoded, 1, torch.device("cpu"))
        two = _validation_summary(torch, model, encoded, 2, torch.device("cpu"))
        self.assertEqual(one["supervised_tokens"], two["supervised_tokens"])
        self.assertAlmostEqual(one["lm_loss"], two["lm_loss"], places=5)
        self.assertAlmostEqual(one["loss"], two["loss"], places=5)

    def test_legacy_single_optimizer_group_remains_loadable(self):
        import torch
        from cognition_slm.config import ModelConfig
        from cognition_slm.model import CognitionSLM
        from cognition_slm.train import train

        config = ModelConfig(block_size=256, n_layer=1, n_head=2, n_embd=16)
        model = CognitionSLM(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        checkpoint = self.root / "legacy.pt"
        torch.save({
            "model_config": config.to_dict(), "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(), "metadata": {"step": 0},
        }, checkpoint)
        result = train(self.args("legacy-resumed.pt", resume=str(checkpoint), steps=1))
        saved = torch.load(result["checkpoint"], weights_only=True)
        self.assertEqual(len(saved["optimizer_state_dict"]["param_groups"]), 1)
        self.assertEqual(saved["model_config"]["architecture"], "legacy")

    def test_cuda_fp16_checkpoint_roundtrip(self):
        import math
        import torch
        from cognition_slm.train import train

        if not torch.cuda.is_available():
            self.skipTest("CUDA required for fp16 training")
        initial = train(self.args("fp16.pt", device="cuda", precision="fp16", steps=6))
        resumed = train(self.args(
            "fp16-resumed.pt", device="cuda", precision="fp16", steps=8,
            resume=initial["checkpoint"],
        ))
        self.assertTrue(math.isfinite(resumed["final_loss"]))
        saved = torch.load(resumed["checkpoint"], map_location="cpu", weights_only=True)
        self.assertTrue(saved["scaler_state_dict"])
        self.assertTrue(saved["cuda_rng_state_all"])
        self.assertEqual(resumed["resumed_from_step"], 6)
        self.assertGreater(resumed["optimizer_steps"], 0)

    def test_overflow_does_not_advance_schedule(self):
        from unittest.mock import Mock
        from cognition_slm.train import _scaled_optimizer_step

        scaler = Mock()
        optimizer = Mock()
        scheduler = Mock()
        scaler.get_scale.side_effect = [8.0, 4.0, 4.0, 4.0]
        self.assertFalse(_scaled_optimizer_step(scaler, optimizer, scheduler))
        scheduler.step.assert_not_called()
        self.assertTrue(_scaled_optimizer_step(scaler, optimizer, scheduler))
        scheduler.step.assert_called_once_with()

    def test_all_skipped_updates_save_diagnostics_and_fail(self):
        import torch
        from unittest.mock import patch
        from cognition_slm.train import train

        args = self.args(steps=1)
        with patch("cognition_slm.train._scaled_optimizer_step", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "no optimizer updates"):
                train(args)
        saved = torch.load(args.out, weights_only=True)
        self.assertEqual(saved["metadata"]["optimizer_steps"], 0)
        self.assertEqual(saved["metadata"]["skipped_optimizer_steps"], 1)


if __name__ == "__main__":
    unittest.main()
