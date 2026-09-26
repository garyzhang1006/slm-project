import importlib.util
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "compute" / "lora_baseline.py"


def runner():
    # Kaggle bundles for other stages may omit compute/, so skip rather than fail there.
    if not RUNNER.exists():
        raise unittest.SkipTest("compute/lora_baseline.py is not packaged here")
    spec = importlib.util.spec_from_file_location("compute_lora_baseline", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LoraBaselineTests(unittest.TestCase):
    def setUp(self):
        self.module = runner()

    def test_import_has_no_heavy_dependencies(self):
        for name in ("torch", "transformers", "peft"):
            self.assertNotIn(name, vars(self.module))

    def test_dependencies_remove_incompatible_torchao(self):
        calls = []
        with mock.patch.object(self.module.importlib.util, "find_spec", return_value=object()), \
                mock.patch.object(self.module.subprocess, "run", side_effect=lambda cmd, **_: calls.append(cmd)), \
                mock.patch("importlib.metadata.version", return_value="1.0"):
            self.module.ensure_dependencies()
        self.assertEqual(len(calls), 1)
        self.assertIn("uninstall", calls[0])
        self.assertEqual(calls[0][-2:], ["--yes", "torchao"])

    def test_model_is_pinned(self):
        self.assertEqual(self.module.MODEL_ID, "HuggingFaceTB/SmolLM2-360M-Instruct")
        self.assertRegex(self.module.MODEL_REVISION, r"^[0-9a-f]{40}$")
        self.assertEqual(self.module.MODEL_LICENSE, "apache-2.0")

    def test_mask_prompt_trains_on_answer_only(self):
        ids, labels = self.module.mask_prompt([1, 2, 3], [4, 5], max_length=10)
        self.assertEqual(ids, [1, 2, 3, 4, 5])
        self.assertEqual(labels, [-100, -100, -100, 4, 5])
        self.assertEqual(self.module.mask_prompt([1, 2, 3], [4, 5], max_length=4), ([1, 2, 3, 4], [-100] * 3 + [4]))
        self.assertIsNone(self.module.mask_prompt([1, 2, 3], [4], max_length=3))

    def test_collate_pads_without_loss_or_attention(self):
        batch = self.module.collate([([1, 2, 3], [-100, 2, 3]), ([7], [7])], pad_id=0)
        self.assertEqual(batch["input_ids"], [[1, 2, 3], [7, 0, 0]])
        self.assertEqual(batch["labels"], [[-100, 2, 3], [7, -100, -100]])
        self.assertEqual(batch["attention_mask"], [[1, 1, 1], [1, 0, 0]])

    def test_schedule(self):
        planned = self.module.planned_optimizer_steps
        self.assertEqual(planned(160, 8, 2, 2.0, 3000), 20)
        self.assertEqual(planned(161, 8, 2, 1.0, 3000), 11)
        self.assertEqual(planned(10_000, 8, 2, 2.0, 100), 100)
        rate = self.module.learning_rate_at
        self.assertAlmostEqual(rate(0, 100, 10, 1.0), 0.1)
        self.assertAlmostEqual(rate(9, 100, 10, 1.0), 1.0)
        self.assertAlmostEqual(rate(10, 100, 10, 1.0), 1.0)
        self.assertAlmostEqual(rate(99, 100, 10, 1.0), 1 / 90)
        self.assertAlmostEqual(rate(0, 1, 0, 2e-4), 2e-4)

    def test_first_line_stops_at_newline(self):
        self.assertEqual(self.module.first_line("  Seven.\nQuestion: more"), "Seven.")
        self.assertEqual(self.module.first_line(""), "")

    def test_holdout_overlap_is_dropped(self):
        holdout = json.loads((ROOT / "data/simple_questions_holdout.json").read_text())["rows"]
        question = holdout[0]["prompt"]
        rows = [{"prompt": f"  {question.upper()} ", "answer": "7"},
                {"prompt": "Name a color.", "answer": question},
                {"prompt": "Name a color.", "answer": "Blue."}]
        kept, dropped = self.module.drop_holdout_overlap(rows, holdout)
        self.assertEqual((kept, dropped), ([rows[2]], 2))
        # The bare question, reworded punctuation, or a longer prompt around it still counts as a copy.
        rows = [{"prompt": "What is 4 plus 9?", "answer": "13"},
                {"prompt": "what is 4 plus 9 -- reply with the number", "answer": "13"},
                {"prompt": "Quick quiz. What is 4 plus 9? Say it.", "answer": "13"},
                {"prompt": "What is 4 plus 10?", "answer": "14"},
                {"prompt": "What is it called when water freezes?", "answer": "Ice."}]
        kept, dropped = self.module.drop_holdout_overlap(rows, holdout)
        self.assertEqual((kept, dropped), (rows[3:], 3))

    def test_encode_tokenizes_rendered_template(self):
        test = self

        class Encoding:
            def __init__(self, ids):
                self.input_ids = ids

        class FakeTokenizer:
            chat_template = "chatml"
            eos_token_id = 99

            def apply_chat_template(self, messages, add_generation_prompt, tokenize):
                # Newer transformers return a BatchEncoding when tokenize=True, so encode must not ask for it.
                test.assertFalse(tokenize)
                test.assertTrue(add_generation_prompt)
                return "|".join(message["content"] for message in messages) + "|assistant"

            def __call__(self, text, add_special_tokens):
                test.assertFalse(add_special_tokens)
                return Encoding([len(word) for word in text.split()])

        tokenizer = FakeTokenizer()
        prompt_ids = self.module.encode(tokenizer, "Hi there?")
        self.assertIsInstance(prompt_ids, list)
        self.assertTrue(all(isinstance(value, int) for value in prompt_ids))
        prompt_ids, answer_ids = self.module.encode(tokenizer, "Hi there?", " Hello. ")
        self.assertEqual(answer_ids, [6, 99])
        tokenizer.chat_template = None
        self.assertEqual(self.module.encode(tokenizer, "Hi?", "Yes")[1], [3, 99])

    def test_find_input_requires_exactly_one(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "found 0"):
                self.module.find_input("sft_train.jsonl", root)
            (root / "a").mkdir()
            (root / "a" / "sft_train.jsonl").write_text("")
            self.assertEqual(self.module.find_input("sft_train.jsonl", root), root / "a" / "sft_train.jsonl")
            (root / "b").mkdir()
            (root / "b" / "sft_train.jsonl").write_text("")
            with self.assertRaisesRegex(RuntimeError, "found 2"):
                self.module.find_input("sft_train.jsonl", root)
        self.assertRaises(RuntimeError, self.module.find_input, "x", Path(directory) / "missing")

    def test_load_sft_rows_validates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text(json.dumps({"prompt": "Hi?", "answer": "Hello."}) + "\n\n")
            self.assertEqual(self.module.load_sft_rows(path), [{"prompt": "Hi?", "answer": "Hello."}])
            path.write_text(json.dumps({"prompt": "Hi?", "answer": " "}) + "\n")
            with self.assertRaisesRegex(ValueError, "rows.jsonl:1"):
                self.module.load_sft_rows(path)
            path.write_text("")
            with self.assertRaisesRegex(ValueError, "no rows"):
                self.module.load_sft_rows(path)

    def test_messages_and_arguments(self):
        messages = self.module.build_messages("What is 2 plus 2?")
        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        self.assertIn("English", messages[0]["content"])
        args = self.module.parse_args([])
        self.assertEqual((args.batch_size, args.gradient_accumulation_steps, args.max_new_tokens), (8, 2, 64))
        with self.assertRaises(SystemExit):
            self.module.parse_args(["--batch-size", "0"])

    def test_main_refuses_to_run_off_kaggle(self):
        if Path("/kaggle/working").is_dir():
            self.skipTest("running on Kaggle")
        with self.assertRaisesRegex(RuntimeError, "Kaggle"):
            self.module.main([])

    def test_runner_avoids_em_dashes(self):
        self.assertIsNone(re.search("—", RUNNER.read_text(encoding="utf-8")))


if __name__ == "__main__":
    unittest.main()
