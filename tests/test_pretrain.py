import json
import tempfile
import unittest
from pathlib import Path

from cognition_slm.data import DataValidationError, load_pretrain_text, pack_pretrain_text
from cognition_slm.tokenizer import BOS_ID, EOS_ID, ByteTokenizer


class PretrainDataTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_txt_is_one_document_and_jsonl_reads_text_fields(self):
        text = self.root / "web.txt"
        text.write_text("first line\n\nsecond paragraph\n", encoding="utf-8")
        self.assertEqual(load_pretrain_text(text), ["first line\n\nsecond paragraph\n"])
        records = self.root / "web.jsonl"
        records.write_text('{"text": "alpha"}\n\n{"text": "beta", "url": "x"}\n', encoding="utf-8")
        self.assertEqual(load_pretrain_text(records), ["alpha", "beta"])

    def test_invalid_inputs_are_rejected(self):
        cases = {
            "missing.jsonl": '{"body": "alpha"}\n',
            "blank.jsonl": '{"text": "   "}\n',
            "broken.jsonl": '{"text": \n',
            "empty.jsonl": "\n",
            "empty.txt": "  \n",
        }
        for name, content in cases.items():
            with self.subTest(name=name):
                path = self.root / name
                path.write_text(content, encoding="utf-8")
                with self.assertRaises(DataValidationError):
                    load_pretrain_text(path)
        with self.assertRaises(FileNotFoundError):
            load_pretrain_text(self.root / "absent.txt")

    def test_packing_joins_documents_with_eos_into_full_rows(self):
        tokenizer = ByteTokenizer()
        documents = ["abcdefg", "hijklmnop", "qr"]
        rows = pack_pretrain_text(documents, tokenizer, 8)
        stream = [token for row in rows for token in row["input_ids"]]
        expected = [token for document in documents for token in tokenizer.encode(document)]
        self.assertEqual(stream, expected)
        self.assertTrue(all(len(row["input_ids"]) == 8 for row in rows[:-1]))
        self.assertEqual(stream.count(EOS_ID), 3)
        self.assertEqual(stream.count(BOS_ID), 3)
        self.assertEqual({row["answer_start"] for row in rows}, {1})
        self.assertTrue(all(row["task_label"] is None for row in rows))
        self.assertEqual(rows, pack_pretrain_text(documents, tokenizer, 8))

    def test_single_token_tail_is_dropped(self):
        rows = pack_pretrain_text(["abcde"], ByteTokenizer(), 6)
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(rows[0]["input_ids"]), 6)
        rows = pack_pretrain_text(["abcdef"], ByteTokenizer(), 6)
        self.assertEqual([len(row["input_ids"]) for row in rows], [6, 2])

    def test_unlabeled_batch_supervises_every_position_without_aux_losses(self):
        import torch
        from cognition_slm.config import ModelConfig
        from cognition_slm.model import CognitionSLM
        from cognition_slm.train import _batch, _validation_summary

        rows = pack_pretrain_text(["hello world", "second doc"], ByteTokenizer(), 16)
        batch = _batch(torch, rows, [0, 1], torch.device("cpu"))
        self.assertIsNone(batch[2])
        self.assertIsNone(batch[3])
        self.assertIsNone(batch[4])
        self.assertEqual(int(batch[6].sum()), sum(len(row["input_ids"]) - 1 for row in rows))
        model = CognitionSLM(ModelConfig(block_size=16, n_layer=1, n_head=2, n_embd=16))
        output = model(batch[0], attention_mask=batch[1], pool_positions=batch[5], lm_loss_mask=batch[6])
        self.assertIsNotNone(output.lm_loss)
        self.assertIsNone(output.task_loss)
        summary = _validation_summary(torch, model, rows, 1, torch.device("cpu"))
        self.assertAlmostEqual(summary["loss"], summary["lm_loss"], places=6)
        self.assertIsNone(summary["task_accuracy"])


class PretrainTrainingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.text = self.root / "web.jsonl"
        self.text.write_text("\n".join(
            json.dumps({"text": f"document {index} says hello to the model."}) for index in range(6)
        ), encoding="utf-8")
        self.eval_text = self.root / "held.txt"
        self.eval_text.write_text("held out text for evaluation", encoding="utf-8")
        self.sft = self.root / "sft.jsonl"
        self.sft.write_text(json.dumps({
            "id": "0", "prompt": "Write Python.", "answer": "return 1",
            "task_type": "code_generation", "confidence": 0.8,
            "error_category": "none", "source": "test", "license": "CC0-1.0",
        }), encoding="utf-8")

    def args(self, name="model.pt", source=("--pretrain-text",), **changes):
        from cognition_slm.train import build_parser

        source = list(source) + ([str(self.text)] if len(source) == 1 else [])
        args = build_parser().parse_args(source + [
            "--out", str(self.root / name), "--device", "cpu", "--steps", "3",
            "--warmup-steps", "0", "--block-size", "128", "--n-layer", "1", "--n-head", "2",
            "--n-embd", "16", "--save-every", "1", "--batch-size", "2",
        ])
        for key, value in changes.items():
            setattr(args, key, value)
        return args

    def test_pretrain_run_counts_every_target_and_evaluates_raw_text(self):
        import torch
        from cognition_slm.train import train

        result = train(self.args(pretrain_eval_text=str(self.eval_text), eval_every=1))
        self.assertEqual(result["objective"], "pretrain")
        self.assertEqual(result["records"], 6)
        rows = pack_pretrain_text(load_pretrain_text(self.text), ByteTokenizer(), 128)
        self.assertEqual(result["packed_rows"], len(rows))
        self.assertEqual(result["truncated_records"], 0)
        # Every packed row supervises all len - 1 targets.
        self.assertGreater(result["supervised_tokens"], 6 * 40)
        self.assertEqual(result["processed_input_tokens"] - result["supervised_tokens"], 6)
        self.assertIsNone(result["validation"]["task_accuracy"])
        saved = torch.load(result["checkpoint"], weights_only=True)
        self.assertEqual(saved["metadata"]["objective"], "pretrain")
        self.assertEqual(saved["metadata"]["samples_seen"], 6)
        # No labels reach the auxiliary heads, so their zero-initialized biases never move.
        self.assertEqual(float(saved["model_state_dict"]["task_head.bias"].abs().sum()), 0.0)

    def test_pretrain_resume_matches_uninterrupted_run(self):
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
        resumed = train(self.args("resumed.pt", resume=str(first)))
        self.assertEqual(resumed["resumed_from_step"], 1)
        final = torch.load(result["checkpoint"], weights_only=True)
        restored = torch.load(resumed["checkpoint"], weights_only=True)
        for name, weight in final["model_state_dict"].items():
            torch.testing.assert_close(weight, restored["model_state_dict"][name], rtol=0, atol=0)

    def test_pretrain_accepts_sft_eval_data(self):
        from cognition_slm.train import train

        result = train(self.args(eval_data=str(self.sft), steps=1))
        self.assertIsNotNone(result["validation"]["task_accuracy"])

    def test_zero_aux_weight_leaves_heads_untouched_in_sft(self):
        import torch
        from cognition_slm.train import train

        weighted = train(self.args("weighted.pt", source=("--data", str(self.sft)), steps=1))
        zero = train(self.args("zero.pt", source=("--data", str(self.sft)), steps=1, aux_loss_weight=0.0))
        self.assertEqual(weighted["aux_loss_weight"], 0.25)
        self.assertEqual(zero["aux_loss_weight"], 0.0)
        weighted_bias = torch.load(weighted["checkpoint"], weights_only=True)["model_state_dict"]["task_head.bias"]
        zero_bias = torch.load(zero["checkpoint"], weights_only=True)["model_state_dict"]["task_head.bias"]
        self.assertGreater(float(weighted_bias.abs().sum()), 0.0)
        self.assertEqual(float(zero_bias.abs().sum()), 0.0)

    def test_resume_keeps_checkpoint_aux_weight_unless_overridden(self):
        import torch
        from cognition_slm.train import train

        sft = ("--data", str(self.sft))
        first = train(self.args("zero.pt", source=sft, steps=1, aux_loss_weight=0.0))
        resumed = train(self.args("resumed.pt", source=sft, steps=2, resume=first["checkpoint"]))
        self.assertEqual(resumed["aux_loss_weight"], 0.0)
        bias = torch.load(resumed["checkpoint"], weights_only=True)["model_state_dict"]["task_head.bias"]
        self.assertEqual(float(bias.abs().sum()), 0.0)
        overridden = train(self.args(
            "override.pt", source=sft, steps=2, resume=first["checkpoint"], aux_loss_weight=0.5
        ))
        self.assertEqual(overridden["aux_loss_weight"], 0.5)
        # Checkpoints written before the flag existed trained with the old fixed 0.25.
        legacy = torch.load(first["checkpoint"], weights_only=True)
        del legacy["metadata"]["aux_loss_weight"]
        torch.save(legacy, self.root / "legacy.pt")
        resumed = train(self.args("legacy-out.pt", source=sft, steps=2, resume=str(self.root / "legacy.pt")))
        self.assertEqual(resumed["aux_loss_weight"], 0.25)

    def test_dry_run_reports_packing(self):
        from cognition_slm.train import train

        result = train(self.args(dry_run=True))
        self.assertEqual(result["status"], "dry-run")
        self.assertEqual(result["objective"], "pretrain")
        self.assertLessEqual(result["max_tokens"], 128)

    def test_cli_requires_exactly_one_source_and_valid_weight(self):
        import contextlib
        import io
        import sys
        from unittest.mock import patch
        from cognition_slm.train import build_parser, main

        parser = build_parser()
        help_text = parser.format_help()
        for flag in ("--pretrain-text", "--pretrain-eval-text", "--aux-loss-weight"):
            self.assertIn(flag, help_text)
        for argv in ([], ["--data", str(self.sft), "--pretrain-text", str(self.text)]):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parser.parse_args(argv)
        invalid = (
            ["--pretrain-text", str(self.text), "--aux-loss-weight", "-1"],
            ["--pretrain-text", str(self.text), "--pretrain-eval-text", str(self.text)],
            ["--pretrain-text", str(self.text), "--eval-data", str(self.sft),
             "--pretrain-eval-text", str(self.eval_text)],
        )
        for argv in invalid:
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with patch.object(sys, "argv", ["train"] + argv), self.assertRaises(SystemExit) as raised:
                    main()
                self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
