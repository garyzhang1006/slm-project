import unittest

from cognition_slm.train import _lr_scale


class TrainingTests(unittest.TestCase):
    def test_time_budget_requires_positive_finite_seconds(self):
        from argparse import Namespace
        from cognition_slm.train import _runtime_options

        self.assertEqual(_runtime_options(Namespace()), ("fp32", 1, 100))
        for seconds in (0, -1, float("nan"), float("inf"), -float("inf")):
            with self.subTest(seconds=seconds):
                with self.assertRaisesRegex(ValueError, "max_seconds must be positive and finite"):
                    _runtime_options(Namespace(max_seconds=seconds))
        self.assertEqual(_runtime_options(Namespace(max_seconds=0.1)), ("fp32", 1, 100))

    def test_warmup_and_cosine_schedule_boundaries(self):
        self.assertAlmostEqual(_lr_scale(0, total_steps=20, warmup_steps=5), 0.2)
        self.assertAlmostEqual(_lr_scale(4, total_steps=20, warmup_steps=5), 1.0)
        self.assertAlmostEqual(_lr_scale(20, total_steps=20, warmup_steps=5), 0.0)
        self.assertAlmostEqual(_lr_scale(0, total_steps=20, warmup_steps=0), 1.0)

    def test_final_update_has_positive_learning_rate(self):
        # LambdaLR applies lambda(total_steps - 1) to the last update of a run.
        for total, warmup in ((20, 5), (100, 5), (2, 0), (1, 0)):
            with self.subTest(total=total, warmup=warmup):
                self.assertGreater(_lr_scale(total - 1, total, warmup), 0.0)

    def test_epoch_sampling_covers_every_record_once_per_pass(self):
        from cognition_slm.train import _sample_indices

        first = _sample_indices(7, 10, 0, 10)
        second = _sample_indices(7, 10, 10, 10)
        self.assertEqual(sorted(first), list(range(10)))
        self.assertEqual(sorted(second), list(range(10)))
        self.assertNotEqual(first, second)
        self.assertEqual(_sample_indices(7, 10, 0, 25), first + second + _sample_indices(7, 10, 20, 5))
        self.assertEqual(_sample_indices(7, 10, 3, 4), first[3:7])
        self.assertNotEqual(_sample_indices(8, 10, 0, 10), first)

    def test_nonfinite_loss_skips_fp16_update_and_fails_otherwise(self):
        from unittest.mock import Mock
        from cognition_slm.train import _skip_nonfinite_update

        scaler = Mock()
        scaler.get_scale.return_value = 1024.0
        optimizer = Mock()
        _skip_nonfinite_update("fp16", scaler, optimizer, 3)
        optimizer.zero_grad.assert_called_once_with(set_to_none=True)
        scaler.update.assert_called_once_with(new_scale=512.0)
        for precision in ("fp32", "bf16"):
            with self.assertRaisesRegex(RuntimeError, "non-finite training loss at step 3"):
                _skip_nonfinite_update(precision, Mock(), Mock(), 3)

    def test_cuda_rng_restore_ignores_visible_gpu_count(self):
        import torch
        from unittest.mock import Mock
        from cognition_slm.train import _restore_cuda_rng

        states = [torch.tensor([0], dtype=torch.uint8), torch.tensor([1], dtype=torch.uint8)]
        for saved_device, expected in (("cuda", 0), ("cuda:1", 1), ("cuda:5", 1)):
            with self.subTest(saved_device=saved_device):
                fake_torch = Mock()
                fake_torch.cuda.device_count.return_value = 1
                device = torch.device("cuda")
                _restore_cuda_rng(fake_torch, {
                    "cuda_rng_state_all": states, "metadata": {"device": saved_device},
                }, device)
                restored, target = fake_torch.cuda.set_rng_state.call_args.args
                self.assertTrue(torch.equal(restored, states[expected]))
                self.assertEqual(target, device)
        fake_torch = Mock()
        _restore_cuda_rng(fake_torch, {"cuda_rng_state_all": states}, torch.device("cpu"))
        _restore_cuda_rng(fake_torch, {"cuda_rng_state_all": []}, torch.device("cuda"))
        fake_torch.cuda.set_rng_state.assert_not_called()


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

    def test_resume_learning_rate_must_match_unless_overridden(self):
        import torch
        from cognition_slm.train import train

        parent = train(self.args("parent.pt", steps=1, learning_rate=3e-4))
        with self.assertRaisesRegex(ValueError, "--override-learning-rate"):
            train(self.args("rejected.pt", resume=parent["checkpoint"], learning_rate=1e-5))
        kept = train(self.args("kept.pt", resume=parent["checkpoint"]))
        saved = torch.load(kept["checkpoint"], weights_only=True)
        self.assertEqual(saved["metadata"]["learning_rate"], 3e-4)
        self.assertEqual(saved["scheduler_state_dict"]["base_lrs"], [3e-4, 3e-4])
        overridden = train(self.args(
            "overridden.pt", resume=parent["checkpoint"], learning_rate=1e-5,
            override_learning_rate=True,
        ))
        saved = torch.load(overridden["checkpoint"], weights_only=True)
        self.assertEqual(saved["metadata"]["learning_rate"], 1e-5)
        self.assertEqual(saved["scheduler_state_dict"]["base_lrs"], [1e-5, 1e-5])
        for group in saved["optimizer_state_dict"]["param_groups"]:
            self.assertEqual(group["initial_lr"], 1e-5)
            self.assertLessEqual(group["lr"], 1e-5)

    def test_override_learning_rate_applies_to_first_update_of_step_zero_restart(self):
        import torch
        from unittest.mock import patch
        from cognition_slm.train import train

        parent = train(self.args("parent.pt", steps=1, learning_rate=3e-4))
        payload = torch.load(parent["checkpoint"], weights_only=True)
        payload["metadata"]["step"] = 0
        payload.pop("scheduler_state_dict")
        restart = self.root / "restart.pt"
        torch.save(payload, restart)
        rates = []
        step = torch.optim.AdamW.step

        def record(optimizer, *args, **kwargs):
            rates.append({group["lr"] for group in optimizer.param_groups})
            return step(optimizer, *args, **kwargs)

        with patch.object(torch.optim.AdamW, "step", record):
            train(self.args("restarted.pt", resume=str(restart), steps=2,
                            learning_rate=1e-5, override_learning_rate=True))
        self.assertEqual(rates[0], {1e-5})

    def test_resume_without_warmup_steps_keeps_the_checkpoints_warmup(self):
        # The flag's default of 5 replaced a saved warmup of 2, so the resumed step ran at 2/5 of the peak rate.
        import torch
        from unittest.mock import patch
        from cognition_slm.train import build_parser, train

        parent = train(self.args("parent.pt", steps=1, learning_rate=3e-4, warmup_steps=2))
        args = self.args("resumed.pt", resume=parent["checkpoint"])
        args.warmup_steps = build_parser().get_default("warmup_steps")
        rates = []
        step = torch.optim.AdamW.step

        def record(optimizer, *arguments, **kwargs):
            rates.append({group["lr"] for group in optimizer.param_groups})
            return step(optimizer, *arguments, **kwargs)

        with patch.object(torch.optim.AdamW, "step", record):
            resumed = train(args)
        self.assertEqual(len(rates[0]), 1)
        self.assertAlmostEqual(rates[0].pop(), 3e-4)
        saved = torch.load(resumed["checkpoint"], weights_only=True)
        self.assertEqual(saved["metadata"]["warmup_steps"], 2)

    def test_resume_rejects_a_checkpoint_warmup_longer_than_steps(self):
        # An inherited warmup past --steps kept every resumed step in warmup, so the rate never decayed.
        from cognition_slm.train import build_parser, train

        parent = train(self.args("parent.pt", steps=1, warmup_steps=5))
        args = self.args("resumed.pt", resume=parent["checkpoint"])
        args.warmup_steps = build_parser().get_default("warmup_steps")
        with self.assertRaisesRegex(ValueError, "warmup of 5 steps exceeds --steps 3"):
            train(args)

    def test_resume_applies_and_records_the_requested_weight_decay(self):
        import torch
        from cognition_slm.train import train

        parent = train(self.args("parent.pt", steps=1, weight_decay=0.01))
        resumed = train(self.args("resumed.pt", resume=parent["checkpoint"], steps=2, weight_decay=0.1))
        saved = torch.load(resumed["checkpoint"], weights_only=True)
        self.assertEqual(saved["metadata"]["weight_decay"], 0.1)
        groups = saved["optimizer_state_dict"]["param_groups"]
        self.assertEqual([group["weight_decay"] for group in groups], [0.1, 0.0])

    def test_resume_keeps_the_checkpoint_weight_decay_when_the_flag_is_omitted(self):
        # The parser default of 0.01 replaced a resumed 0.1 run's decay, so it no longer matched an unbroken run.
        import torch
        from cognition_slm.train import DEFAULT_WEIGHT_DECAY, train

        fresh = train(self.args("fresh.pt", steps=1))
        self.assertEqual(torch.load(fresh["checkpoint"], weights_only=True)["metadata"]["weight_decay"],
                         DEFAULT_WEIGHT_DECAY)
        parent = train(self.args("parent.pt", steps=1, weight_decay=0.1))
        resumed = train(self.args("resumed.pt", resume=parent["checkpoint"], steps=2))
        saved = torch.load(resumed["checkpoint"], weights_only=True)
        self.assertEqual(saved["metadata"]["weight_decay"], 0.1)
        groups = saved["optimizer_state_dict"]["param_groups"]
        self.assertEqual([group["weight_decay"] for group in groups], [0.1, 0.0])

    def test_resume_rejects_a_different_seed_mid_stream(self):
        from cognition_slm.train import train

        parent = train(self.args("parent.pt", steps=1, seed=7))
        with self.assertRaisesRegex(ValueError, "--seed 8 differs from checkpoint seed 7; pass --seed 7"):
            train(self.args("rejected.pt", resume=parent["checkpoint"], steps=2, seed=8))
        resumed = train(self.args("resumed.pt", resume=parent["checkpoint"], steps=2, seed=7))
        self.assertEqual(resumed["resumed_from_step"], 1)

    def test_resume_continues_sample_stream_from_checkpoint(self):
        import torch
        from cognition_slm.train import train

        parent = train(self.args("parent.pt", steps=1, gradient_accumulation_steps=3))
        saved = torch.load(parent["checkpoint"], weights_only=True)
        self.assertEqual(saved["metadata"]["samples_seen"], 6)
        legacy = dict(saved, metadata={k: v for k, v in saved["metadata"].items() if k != "samples_seen"})
        legacy_path = self.root / "legacy-sampling.pt"
        torch.save(legacy, legacy_path)
        resumed = train(self.args("resumed.pt", resume=str(legacy_path), steps=2,
                                  gradient_accumulation_steps=3))
        saved = torch.load(resumed["checkpoint"], weights_only=True)
        self.assertEqual(saved["metadata"]["samples_seen"], 12)

    def test_resume_rejects_changed_training_data_unless_allowed(self):
        import json
        import torch
        from cognition_slm.train import train

        parent = train(self.args("parent.pt", steps=1))
        saved = torch.load(parent["checkpoint"], weights_only=True)["metadata"]
        self.assertEqual(saved["training_source"], str(self.data.resolve()))
        self.assertEqual(saved["block_size"], 256)
        other = self.root / "other.jsonl"
        other.write_text(json.dumps({
            "id": "x", "prompt": "Explain.", "answer": "Because.", "task_type": "code_generation",
            "confidence": 0.5, "error_category": "none", "source": "test", "license": "CC0-1.0",
        }))
        with self.assertRaisesRegex(ValueError, "records 2 -> 1.*training_source_sha256.*--allow-data-change"):
            train(self.args("rejected.pt", data=str(other), resume=parent["checkpoint"], steps=2))
        text = self.root / "web.txt"
        text.write_text("raw pretraining text for the objective check")
        with self.assertRaisesRegex(ValueError, "objective 'sft' -> 'pretrain'"):
            train(self.args("objective.pt", data=None, pretrain_text=str(text),
                            resume=parent["checkpoint"], steps=2))
        allowed = train(self.args("allowed.pt", data=str(other), resume=parent["checkpoint"], steps=2,
                                  allow_data_change=True))
        self.assertEqual(allowed["resumed_from_step"], 1)
        saved = torch.load(allowed["checkpoint"], weights_only=True)["metadata"]
        # The sample stream restarted at 0 and drew one batch of 2 from the new data.
        self.assertEqual(saved["samples_seen"], 2)
        self.assertEqual(saved["training_source"], str(other.resolve()))

    def test_resume_without_data_fingerprint_warns(self):
        import torch
        from cognition_slm.train import train

        parent = train(self.args("parent.pt", steps=1))
        payload = torch.load(parent["checkpoint"], weights_only=True)
        for key in ("training_source", "training_source_sha256", "block_size", "records"):
            del payload["metadata"][key]
        legacy = self.root / "legacy.pt"
        torch.save(payload, legacy)
        with self.assertWarnsRegex(UserWarning, "predates training-source metadata"):
            resumed = train(self.args("resumed.pt", resume=str(legacy), steps=2))
        self.assertEqual(resumed["resumed_from_step"], 1)

    def test_resume_without_fused_adamw_drops_the_checkpoints_fused_flag(self):
        # Loading an optimizer state restores its execution flags, so a fused run resumed without --fused-adamw
        # kept stepping fused, even on a CPU, while its metadata said fused_adamw false.
        import torch
        from cognition_slm.train import train

        parent = train(self.args("parent.pt", steps=1))
        payload = torch.load(parent["checkpoint"], weights_only=True)
        for group in payload["optimizer_state_dict"]["param_groups"]:
            group["fused"] = True
        fused = self.root / "fused.pt"
        torch.save(payload, fused)
        resumed = train(self.args("resumed.pt", resume=str(fused), steps=2))
        saved = torch.load(resumed["checkpoint"], weights_only=True)
        self.assertFalse(any(group["fused"] for group in saved["optimizer_state_dict"]["param_groups"]))
        self.assertFalse(saved["metadata"]["fused_adamw"])

    def test_unwritable_out_fails_before_any_step(self):
        from unittest.mock import patch
        from cognition_slm.train import train

        blocker = self.root / "file.txt"
        blocker.write_text("x")
        for out, message in ((blocker / "model.pt", "cannot write --out"), (self.root, "is a directory")):
            args = self.args(steps=1)
            args.out = str(out)
            with self.subTest(out=out.name), patch("cognition_slm.train._batch", side_effect=AssertionError("trained")):
                with self.assertRaisesRegex(ValueError, message):
                    train(args)

    def test_a_record_that_cannot_be_encoded_names_its_file(self):
        # encode_examples names only the record, so a bad --eval-data row read as if --data held it.
        import json
        import re
        from cognition_slm.train import train

        bad = self.root / "bad.jsonl"
        bad.write_text(json.dumps({"id": "q17", "prompt": "x" * 300, "answer": "y",
                                   "task_type": "code_generation", "confidence": 0.8,
                                   "error_category": "none", "source": "test", "license": "CC0-1.0"}))
        message = f"^{re.escape(str(bad))}: example 'q17' has no answer tokens within block_size 256"
        for flag in ("data", "eval_data"):
            with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, message):
                train(self.args(dry_run=True, **{flag: str(bad)}))

    def test_allow_data_change_requires_resume(self):
        import contextlib
        import io
        import sys
        from unittest.mock import patch
        from cognition_slm.train import main

        argv = ["train", "--data", str(self.data), "--allow-data-change"]
        with contextlib.redirect_stderr(io.StringIO()) as stderr, patch.object(sys, "argv", argv):
            with self.assertRaises(SystemExit) as raised:
                main()
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("--allow-data-change requires --resume", stderr.getvalue())

    def test_unreadable_files_are_usage_errors(self):
        import contextlib
        import io
        import sys
        from unittest.mock import patch
        from cognition_slm.train import main

        # Tests may run as root, which reads any file, so the errors are raised where train() would meet them.
        for error in (PermissionError(13, "Permission denied", "data.jsonl"),
                      NotADirectoryError(20, "Not a directory", "notes.txt/data.jsonl")):
            with self.subTest(error=type(error).__name__), contextlib.redirect_stderr(io.StringIO()) as stderr, \
                    patch.object(sys, "argv", ["train", "--data", str(self.data)]), \
                    patch("cognition_slm.train.train", side_effect=error):
                with self.assertRaises(SystemExit) as raised:
                    main()
                self.assertEqual(raised.exception.code, 2)
                self.assertIn(error.strerror, stderr.getvalue())

    def test_nonfinite_optimizer_flags_are_usage_errors(self):
        import contextlib
        import io
        import sys
        from unittest.mock import patch
        from cognition_slm.train import main

        # nan passes a "< 0" check, and nan or inf here makes the weights non-finite after the first update.
        for flag in ("--learning-rate", "--grad-clip", "--weight-decay"):
            for value in ("nan", "inf"):
                argv = ["train", "--data", str(self.data), flag, value]
                with self.subTest(flag=flag, value=value), contextlib.redirect_stderr(io.StringIO()) as stderr, \
                        patch.object(sys, "argv", argv), patch("cognition_slm.train.train") as train:
                    with self.assertRaises(SystemExit) as raised:
                        main()
                    self.assertEqual(raised.exception.code, 2)
                    self.assertIn(f"{flag} ", stderr.getvalue())
                    self.assertIn("finite", stderr.getvalue())
                    train.assert_not_called()

    def test_out_must_not_overwrite_input_data(self):
        import contextlib
        import io
        import sys
        from unittest.mock import patch
        from cognition_slm.train import main

        held_out = self.root / "eval.jsonl"
        held_out.write_text(self.data.read_text())
        # Another spelling of the same file, which only resolve() reveals; Path keeps the "..".
        (self.root / "sub").mkdir()
        respelled = f"{self.root}/sub/../{self.data.name}"
        for argv, message in ((["train", "--data", str(self.data), "--out", str(self.data)], "--out must not be"),
                              (["train", "--data", str(self.data), "--eval-data", str(held_out), "--out", str(held_out)],
                               "--out must not be"),
                              (["train", "--data", str(self.data), "--out", respelled], "--out must not be"),
                              (["train", "--data", str(self.data), "--eval-data", respelled],
                               "--eval-data must be different from the training data")):
            with self.subTest(argv=argv[3:]), contextlib.redirect_stderr(io.StringIO()) as stderr, \
                    patch.object(sys, "argv", argv), patch("cognition_slm.train.train") as run:
                with self.assertRaises(SystemExit) as raised:
                    main()
                self.assertEqual(raised.exception.code, 2)
                self.assertIn(message, stderr.getvalue())
                run.assert_not_called()

    def test_cli_reports_training_input_errors_without_a_traceback(self):
        import contextlib
        import io
        import sys
        from unittest.mock import patch
        from cognition_slm.train import main

        missing = self.root / "missing.jsonl"
        out = str(self.root / "model.pt")
        for data, message in ((missing, str(missing)), (self.root, str(self.root))):
            argv = ["train", "--data", str(data), "--out", out]
            with self.subTest(data=data.name), contextlib.redirect_stderr(io.StringIO()) as stderr, \
                    patch.object(sys, "argv", argv):
                with self.assertRaises(SystemExit) as raised:
                    main()
                self.assertEqual(raised.exception.code, 2)
                self.assertIn(message, stderr.getvalue())
        argv = ["train", "--data", str(self.data), "--out", out, "--resume", out,
                "--steps", "3", "--warmup-steps", "0"]
        with contextlib.redirect_stderr(io.StringIO()) as stderr, patch.object(sys, "argv", argv), \
                patch("cognition_slm.train.train", side_effect=ValueError("--steps must exceed checkpoint step 3")):
            with self.assertRaises(SystemExit) as raised:
                main()
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("--steps must exceed checkpoint step 3", stderr.getvalue())

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

    def test_time_budget_saves_and_resumes_exactly_with_final_validation(self):
        import torch
        from unittest.mock import patch
        from cognition_slm.train import train

        full = train(self.args("full.pt", dropout=0.1))
        validation = self.root / "validation.jsonl"
        validation.write_bytes(self.data.read_bytes())
        with patch("cognition_slm.train.monotonic", side_effect=[100.0, 111.0, 112.0]):
            limited = train(self.args(
                "limited.pt", dropout=0.1, max_seconds=10.0, save_every=100,
                eval_every=100, eval_data=str(validation),
            ))
        self.assertEqual(limited["steps"], 1)
        self.assertEqual(limited["requested_steps"], 3)
        self.assertEqual(limited["completed_steps"], 1)
        self.assertEqual(limited["stopped_reason"], "time_budget")
        self.assertEqual(limited["duration_seconds"], 12.0)
        self.assertEqual(limited["validation"]["step"], 1)
        saved = torch.load(limited["checkpoint"], weights_only=True)
        self.assertEqual(saved["metadata"]["step"], 1)
        self.assertEqual(saved["metadata"]["requested_steps"], 3)
        self.assertEqual(saved["metadata"]["stopped_reason"], "time_budget")
        self.assertTrue(saved["optimizer_state_dict"]["state"])
        self.assertEqual(saved["scheduler_state_dict"]["last_epoch"], 1)
        self.assertIn("torch_rng_state", saved)
        resumed = train(self.args("resumed.pt", resume=limited["checkpoint"]))
        self.assertEqual(resumed["resumed_from_step"], 1)
        self.assertEqual(resumed["completed_steps"], 2)
        self.assertEqual(resumed["stopped_reason"], "steps_completed")
        expected = torch.load(full["checkpoint"], weights_only=True)
        actual = torch.load(resumed["checkpoint"], weights_only=True)
        for name, weight in expected["model_state_dict"].items():
            torch.testing.assert_close(weight, actual["model_state_dict"][name], rtol=0, atol=0)
        torch.testing.assert_close(expected["torch_rng_state"], actual["torch_rng_state"], rtol=0, atol=0)

    def test_time_budget_preserves_checkpoint_if_final_validation_fails(self):
        import torch
        from unittest.mock import patch
        from cognition_slm.train import train

        args = self.args(max_seconds=1.0, save_every=100, eval_every=100, eval_data=str(self.data))
        with patch("cognition_slm.train.monotonic", side_effect=[0.0, 2.0, 3.0]):
            with patch("cognition_slm.train._validation_summary", side_effect=RuntimeError("validation failed")):
                with self.assertRaisesRegex(RuntimeError, "validation failed"):
                    train(args)
        saved = torch.load(args.out, weights_only=True)
        self.assertEqual(saved["metadata"]["step"], 1)
        self.assertEqual(saved["metadata"]["stopped_reason"], "time_budget")
        self.assertTrue(saved["optimizer_state_dict"]["state"])

    def test_finishing_requested_steps_at_deadline_reports_completion(self):
        from unittest.mock import patch
        from cognition_slm.train import train

        with patch("cognition_slm.train.monotonic", side_effect=[0.0, 1.0, 1.0]):
            result = train(self.args(steps=1, max_seconds=1.0))
        self.assertEqual(result["steps"], 1)
        self.assertEqual(result["stopped_reason"], "steps_completed")

    def test_cpu_rejects_cuda_precision(self):
        from cognition_slm.train import train

        for precision in ("fp16", "bf16"):
            with self.assertRaisesRegex(ValueError, "require a CUDA device"):
                train(self.args(precision=precision))

    def test_device_flags_are_checked_before_the_data_and_checkpoint_load(self):
        from unittest.mock import patch
        from cognition_slm.train import train

        resume = str(self.root / "missing.pt")
        for changes, message in (({"fused_adamw": True}, "fused-adamw requires a CUDA"),
                                 ({"precision": "fp16"}, "require a CUDA device"),
                                 ({"device": "nonsense"}, "not a torch device")):
            with self.subTest(changes=changes), \
                 patch("cognition_slm.train.load_jsonl") as load_data, \
                 patch("cognition_slm.train.load_checkpoint_payload") as load_checkpoint, \
                 self.assertRaisesRegex(ValueError, message):
                train(self.args(resume=resume, **changes))
            load_data.assert_not_called()
            load_checkpoint.assert_not_called()

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
        with self.assertWarnsRegex(UserWarning, "predates training-source metadata"):
            result = train(self.args("legacy-resumed.pt", resume=str(checkpoint), steps=1))
        saved = torch.load(result["checkpoint"], weights_only=True)
        self.assertEqual(len(saved["optimizer_state_dict"]["param_groups"]), 1)
        self.assertEqual(saved["model_config"]["architecture"], "legacy")

    def test_resume_rejects_task_types_the_checkpoint_lacks(self):
        import json
        import torch
        from cognition_slm.config import LEGACY_TASK_TYPES, ModelConfig
        from cognition_slm.data import DataValidationError
        from cognition_slm.model import CognitionSLM
        from cognition_slm.train import train

        config = ModelConfig(block_size=256, n_layer=1, n_head=2, n_embd=16, task_types=LEGACY_TASK_TYPES)
        checkpoint = self.root / "five-class.pt"
        torch.save({"model_config": config.to_dict(), "model_state_dict": CognitionSLM(config).state_dict(),
                    "metadata": {"step": 0}}, checkpoint)
        record = json.loads(self.data.read_text().splitlines()[0])
        self.data.write_text(json.dumps({**record, "task_type": "language_generation"}))
        # The five-class task head would otherwise fail inside the loss with "Target 5 is out of bounds".
        with self.assertRaisesRegex(DataValidationError, "language_generation"):
            train(self.args("rejected.pt", resume=str(checkpoint)))

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
