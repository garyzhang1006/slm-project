import importlib.util
import json
from pathlib import Path
import re
import sys
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

    def test_english_probes_are_not_training_prompts(self):
        from compute import short_facts
        from compute.stage3_sft_data import question_key

        trained = {question_key(row["prompt"]) for row in short_facts.short_fact_rows()}
        self.assertEqual([prompt for prompt in self.module.ENGLISH_PROBES if question_key(prompt) in trained], [])

    def test_dependencies_remove_incompatible_torchao(self):
        calls = []
        with mock.patch.object(self.module.importlib.util, "find_spec", return_value=object()), \
                mock.patch.object(self.module.subprocess, "run", side_effect=lambda cmd, **_: calls.append(cmd)), \
                mock.patch("importlib.metadata.version", return_value="1.0"):
            self.module.ensure_dependencies()
        self.assertEqual(len(calls), 1)
        self.assertIn("uninstall", calls[0])
        self.assertEqual(calls[0][-2:], ["--yes", "torchao"])

    def test_missing_packages_install_the_versions_the_reported_adapters_used(self):
        calls = []
        present = {"transformers"}
        with mock.patch.object(self.module.importlib.util, "find_spec",
                               side_effect=lambda name: object() if name in present else None), \
                mock.patch.object(self.module.subprocess, "run", side_effect=lambda cmd, **_: calls.append(cmd)), \
                mock.patch("importlib.metadata.version", return_value="1.0"):
            self.module.ensure_dependencies()
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][-2:], ["peft==0.19.1", "accelerate==1.13.0"])

    def test_left_pad_aligns_prompt_ends(self):
        ids, mask = self.module.left_pad([[5, 6, 7], [8]], pad_id=0)
        self.assertEqual(ids, [[5, 6, 7], [0, 0, 8]])
        self.assertEqual(mask, [[1, 1, 1], [0, 0, 1]])

    def test_generate_answers_batches_and_keeps_order(self):
        import torch

        class Tokenizer:
            chat_template = None
            pad_token_id = eos_token_id = 0

            def __call__(self, text, add_special_tokens):
                # Only the question text, so each row gets its own ids and a reordered answer shows up.
                question = text.split("Question: ", 1)[1].split("\nAnswer:", 1)[0]
                return type("Encoded", (), {"input_ids": [ord(character) for character in question]})()

            def decode(self, ids, skip_special_tokens):
                return "".join(chr(int(value)) for value in ids if int(value) != 0)

        class Model:
            device = "cpu"
            calls = []

            def train(self, mode):
                self.mode = mode

            def generate(self, input_ids, attention_mask, **settings):
                self.calls.append(input_ids.shape[0])
                # Echo each row's last prompt character, then a newline and EOS padding.
                last = input_ids[:, -1:]
                tail = torch.tensor([[ord("\n"), 0]] * input_ids.shape[0])
                return torch.cat([input_ids, last, tail], dim=1)

        model = Model()
        prompts = ["Is it a?", "Longer question ending in b", "c", "Pick d", "Say e"]
        answers = self.module.generate_answers(torch, model, Tokenizer(), prompts, 4, batch_size=2)
        self.assertEqual(model.calls, [2, 2, 1])
        self.assertFalse(model.mode)
        self.assertEqual(len(answers), 5)
        self.assertEqual(answers, ["?", "b", "c", "d", "e"])

    def test_best_adapter_keeps_lowest_loss_and_restores_it(self):
        import torch

        lora = torch.nn.Parameter(torch.tensor([1.0]))
        frozen = torch.nn.Parameter(torch.tensor([5.0]), requires_grad=False)
        named = lambda: [("lora_A", lora), ("base", frozen)]
        best = self.module.BestAdapter()
        self.assertFalse(best.offer(None, 1, named()))
        self.assertFalse(best.offer(float("nan"), 1, named()))
        self.assertTrue(best.offer(2.0, 250, named()))
        self.assertEqual(set(best.weights), {"lora_A"})
        lora.data.fill_(9.0)
        self.assertFalse(best.offer(2.5, 500, named()))
        self.assertEqual(best.step, 250)
        best.restore(named())
        self.assertEqual(lora.item(), 1.0)
        self.assertTrue(best.offer(1.5, 750, named()))

    def test_model_choice_sets_base_and_defaults(self):
        small = self.module.parse_args([])
        self.assertEqual((small.model_id, small.epochs, small.gradient_checkpointing),
                         (self.module.MODEL_ID, 2.0, False))
        large = self.module.parse_args(["--model", "1.7b"])
        self.assertEqual(large.model_id, "HuggingFaceTB/SmolLM2-1.7B-Instruct")
        self.assertRegex(large.model_revision, r"^[0-9a-f]{40}$")
        self.assertEqual((large.batch_size, large.gradient_accumulation_steps, large.epochs), (4, 4, 1.0))
        self.assertTrue(large.gradient_checkpointing)
        self.assertEqual(self.module.parse_args(["--model", "1.7b", "--batch-size", "2"]).batch_size, 2)
        with self.assertRaises(SystemExit), mock.patch("sys.stderr"):
            self.module.parse_args(["--model", "7b"])

    def test_kaggle_runs_finish_their_epochs_within_the_hours_the_watcher_reserves(self):
        # The 2026-09-29 Kaggle runs encoded 20,744 training rows and measured these seconds per optimizer
        # step, eval passes included. Neither the step cap nor the timer may cut the planned epochs short.
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from compute.run_pipeline import STAGE_HOURS
        from compute.stages import STAGES

        for stage, epochs, seconds_per_step in (("lora", 2.0, 1.73), ("lora_1b7", 1.0, 8.96)):
            with self.subTest(stage=stage):
                args = self.module.parse_args(STAGES[stage].get("args", []))
                self.assertEqual(args.epochs, epochs)
                steps = self.module.planned_optimizer_steps(20_744, args.batch_size, args.gradient_accumulation_steps,
                                                            args.epochs, args.max_steps)
                self.assertLess(steps, args.max_steps)
                self.assertLess(steps * seconds_per_step * 1.1, args.max_seconds)
                self.assertGreaterEqual(STAGE_HOURS[stage], args.max_seconds / 3600 + 1.0)

    def test_base_model_is_written_beside_both_exports(self):
        source = RUNNER.read_text()
        self.assertIn("write_json(adapter / BASE_MODEL_FILE, base)", source)
        self.assertIn("write_json(merged / BASE_MODEL_FILE, base)", source)
        self.assertNotIn("from_pretrained(MODEL_ID", source)

    def test_base_model_defaults_to_360m_for_older_adapters(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = Path(directory)
            self.assertEqual(self.module.base_model(adapter), (self.module.MODEL_ID, self.module.MODEL_REVISION))
            (adapter / "base_model.json").write_text(json.dumps(self.module.MODELS["1.7b"]))
            self.assertEqual(self.module.base_model(adapter)[0], "HuggingFaceTB/SmolLM2-1.7B-Instruct")

    def test_eval_every_is_validated(self):
        self.assertEqual(self.module.parse_args([]).eval_every, 250)
        with self.assertRaises(SystemExit), mock.patch("sys.stderr"):
            self.module.parse_args(["--eval-every", "-1"])

    def test_training_stays_in_fp32(self):
        # fp16 autocast made the base SmolLM2 eval loss NaN on Kaggle and ruined the first adapter.
        source = RUNNER.read_text()
        self.assertNotIn("torch.float16", source)
        self.assertNotIn("GradScaler", source)
        self.assertIn("isfinite(loss)", source)

    def test_merged_export_follows_final_eval(self):
        # merge_and_unload rewrites the base layers, so it must run after the adapter is saved and scored.
        source = RUNNER.read_text()
        self.assertLess(source.index('report["final"] = evaluate_holdout'), source.index("merge_and_unload()"))
        self.assertLess(source.index("model.save_pretrained(adapter)"), source.index("merge_and_unload()"))
        self.assertIn('"lora-merged"', source)
        self.assertIn("adapter_sha256=", source)

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

    def test_window_weights_share_the_update_by_answer_tokens(self):
        # The model's loss is a mean per micro-batch, so equal weights would give each token of a short-answer
        # micro-batch ten times the pull of one in a long-answer micro-batch.
        short, long = [([0] * 17, [-100] + [5] * 16)], [([0] * 161, [-100] + [5] * 160)]
        self.assertEqual(self.module.window_weights([short, long]), [16 / 176, 160 / 176])
        # The causal loss shifts labels left by one, so a first-position label never counts.
        self.assertEqual(self.module.window_weights([[([0, 0], [5, 6])], [([0, 0], [-100, 6])]]), [0.5, 0.5])
        source = RUNNER.read_text()
        self.assertIn("for rows, weight in zip(window, window_weights(window)):", source)
        self.assertIn("loss = model(**batch).loss * weight", source)

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

    def test_everyday_eval_copies_are_dropped_too(self):
        screened = self.module.screened_rows(ROOT)
        counts = [len(json.loads((ROOT / name).read_text())["rows"])
                  for name in ("data/simple_questions_holdout.json", "data/everyday_eval.json")]
        self.assertEqual(len(screened), sum(counts))
        rows = [{"prompt": "Which planet is closest to the sun?", "answer": "Mercury"},
                {"prompt": "Quiz time. Which planet is closest to the sun? One word.", "answer": "Mercury"},
                {"prompt": "Which planet is famous for its bright rings?", "answer": "Saturn"}]
        kept, dropped = self.module.drop_holdout_overlap(rows, screened)
        self.assertEqual((kept, dropped), (rows[2:], 2))

    def test_project_rows_survive_the_trainer_screen(self):
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from compute import short_facts

        rows = short_facts.short_fact_rows()
        kept = {row["prompt"] for row in self.module.drop_holdout_overlap(rows, self.module.screened_rows(ROOT))[0]}
        self.assertEqual([row["prompt"] for row in rows if row["prompt"] not in kept], [])

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

    def test_find_input_names_the_kernel_that_writes_the_file(self):
        missing = Path(tempfile.gettempdir()) / "no-such-kaggle-input"
        with self.assertRaisesRegex(RuntimeError, "slm-sft-data"):
            self.module.find_input("sft_eval.jsonl", missing)
        with self.assertRaises(RuntimeError) as caught:
            self.module.find_input("adapter_config.json", missing)
        self.assertIn("slm-lora-baseline", str(caught.exception))
        self.assertNotIn("slm-sft-data", str(caught.exception))

    def test_load_sft_rows_validates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text(json.dumps({"prompt": "Hi?", "answer": "Hello."}) + "\n\n")
            self.assertEqual(self.module.load_sft_rows(path), [{"prompt": "Hi?", "answer": "Hello."}])
            path.write_text(json.dumps({"prompt": "Hi?", "answer": " "}) + "\n")
            with self.assertRaisesRegex(ValueError, "rows.jsonl:1"):
                self.module.load_sft_rows(path)
            # U+2028 stays raw under ensure_ascii=False; the reader must not split the record there.
            row = {"prompt": "Hi\u2028there?", "answer": "Hello\u2029again."}
            path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            self.assertEqual(self.module.load_sft_rows(path), [row])
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
