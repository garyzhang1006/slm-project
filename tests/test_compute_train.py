"""Pure-helper tests for the pretrain, SFT and evaluation stage runners (no torch, no network)."""

import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from compute import stage2_pretrain as pretrain  # noqa: E402
from compute import stage4_sft as sft  # noqa: E402
from compute import stage5_evaluate as evaluate  # noqa: E402


def needs(module):
    if importlib.util.find_spec(module) is None:
        raise unittest.SkipTest(f"{module} is not available here")


def write_documents(path, texts):
    path.write_text("".join(json.dumps({"text": text, "source": "test"}) + "\n" for text in texts))


class ImportTests(unittest.TestCase):
    def test_runners_import_without_torch_or_training(self):
        code = ("import sys; import compute.stage2_pretrain, compute.stage4_sft, compute.stage5_evaluate; "
                "print(sorted({'torch', 'datasets', 'transformers'} & set(sys.modules)))")
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                                env={"PYTHONPATH": str(ROOT / "src")}, check=True)
        self.assertEqual(result.stdout.strip(), "[]")

    def test_main_refuses_to_run_off_kaggle(self):
        if Path("/kaggle/working").is_dir():
            self.skipTest("running on Kaggle")
        for runner, argv in ((pretrain, ["--session", "1"]), (sft, []), (evaluate, [])):
            with self.assertRaisesRegex(RuntimeError, "Kaggle"):
                runner.main(argv)


class ShardTests(unittest.TestCase):
    def test_selection_starts_at_offset_and_stops_at_length(self):
        sizes = [10, 10, 10, 10, 10]
        self.assertEqual(pretrain.select_shard_indices(sizes, 0, 25), [0, 1, 2])
        self.assertEqual(pretrain.select_shard_indices(sizes, 20, 15), [2, 3])
        # A start inside a document skips to the next document boundary.
        self.assertEqual(pretrain.select_shard_indices(sizes, 15, 10), [2])

    def test_selection_wraps_once_without_repeating_documents(self):
        sizes = [10, 10, 10, 10]
        self.assertEqual(pretrain.select_shard_indices(sizes, 30, 30), [3, 0, 1])
        self.assertEqual(pretrain.select_shard_indices(sizes, 0, 1000), [0, 1, 2, 3])
        # Offsets past the corpus continue on the next pass.
        self.assertEqual(pretrain.select_shard_indices(sizes, 50, 10), [1])

    def test_selection_rejects_empty_input(self):
        with self.assertRaises(ValueError):
            pretrain.select_shard_indices([], 0, 10)
        with self.assertRaises(ValueError):
            pretrain.select_shard_indices([5], 0, 0)

    def test_document_tokens_count_utf8_bytes_plus_bos_and_eos(self):
        self.assertEqual(pretrain.document_tokens(json.dumps({"text": "héllo"})), 8)

    def test_write_shard_copies_the_selected_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            source, destination = Path(directory) / "train.jsonl", Path(directory) / "shard.jsonl"
            write_documents(source, ["a" * 8, "b" * 8, "c" * 8, "d" * 8])
            summary = pretrain.write_shard(source, destination, 10, 15)
            rows = [json.loads(line)["text"] for line in destination.read_text().splitlines()]
            self.assertEqual(rows, ["b" * 8, "c" * 8])
            self.assertEqual((summary["documents"], summary["tokens"], summary["corpus_tokens"]), (2, 20, 40))
            self.assertEqual(summary["sha256"], pretrain.digest(destination))


class PlanTests(unittest.TestCase):
    def setUp(self):
        needs("compute.stages")

    def test_unmeasured_session_sizes_shard_for_a_faster_gpu(self):
        plan = pretrain.plan_session(0, 10_000, 100, 3600, 1.0, measured=False)
        self.assertEqual(plan["shard_steps"], int(3600 // (1.0 * pretrain.UNMEASURED_SPEEDUP)))
        self.assertEqual((plan["start_token"], plan["shard_tokens"]), (0, plan["shard_steps"] * 100))

    def test_last_session_is_capped_at_remaining_steps(self):
        plan = pretrain.plan_session(9_900, 10_000, 100, 3600, 1.0, measured=True)
        self.assertEqual((plan["remaining_steps"], plan["shard_steps"], plan["start_token"]), (100, 100, 990_000))

    def test_finished_run_refuses_another_session(self):
        with self.assertRaisesRegex(ValueError, "sft"):
            pretrain.plan_session(10_000, 10_000, 100, 3600, 1.0, measured=True)

    def test_default_run_has_enough_room_to_chain_sessions(self):
        from compute import stages
        plan = pretrain.plan_session(0, stages.PRETRAIN_TOTAL_STEPS, stages.TOKENS_PER_STEP,
                                     stages.PRETRAIN_SESSION_SECONDS, stages.SECONDS_PER_STEP_ESTIMATE, False)
        self.assertLess(plan["shard_steps"], stages.PRETRAIN_TOTAL_STEPS)
        # Host RAM: the trainer keeps about two 8-byte list slots per shard token.
        self.assertLess(plan["shard_tokens"] * 16, 20 * 1024**3)


class CommandTests(unittest.TestCase):
    """The runners' trainer arguments must pass cognition_slm.train's own parser and checks."""

    def parse(self, arguments):
        from cognition_slm import train

        args = train.build_parser().parse_args(arguments)
        train._runtime_options(args)
        self.assertLessEqual(args.warmup_steps, args.steps)
        self.assertEqual(args.aux_loss_weight, 0)
        self.assertEqual((args.precision, args.device), ("fp16", "cuda"))
        return args

    def test_pretrain_fresh_and_resumed_sessions(self):
        needs("compute.stages")
        from compute import stages
        path = Path("x.jsonl")
        fresh = self.parse(pretrain.pretrain_arguments(path, Path("e.jsonl"), Path("o.pt"), 39_600.7, None))
        self.assertEqual((fresh.preset, fresh.resume, fresh.max_seconds), ("slm-160m", None, 39_600))
        self.assertEqual(fresh.steps, stages.PRETRAIN_TOTAL_STEPS)
        self.assertEqual(fresh.batch_size * fresh.gradient_accumulation_steps * 2048, stages.TOKENS_PER_STEP)
        resumed = self.parse(pretrain.pretrain_arguments(path, Path("e.jsonl"), Path("o.pt"), 100, Path("r.pt")))
        # A resumed run must not restate the rate, or train.py rejects a mismatch with the checkpoint.
        self.assertEqual((resumed.resume, resumed.learning_rate, resumed.steps), ("r.pt", None, fresh.steps))
        # Each session trains on a new shard, which the resume data guard would otherwise reject.
        self.assertTrue(resumed.allow_data_change)
        self.assertFalse(fresh.allow_data_change)

    def test_sft_restarts_schedule_at_new_rate(self):
        args = self.parse(sft.sft_arguments(Path("i.pt"), Path("t.jsonl"), Path("v.jsonl"), Path("l.pt"),
                                            Path("b.pt"), 200))
        self.assertTrue(args.override_learning_rate)
        self.assertEqual((args.learning_rate, args.resume, args.warmup_steps), (1e-4, "i.pt", 20))

    def test_sft_keeps_the_lowest_eval_loss_weights_apart_from_the_last_step(self):
        args = self.parse(sft.sft_arguments(Path("i.pt"), Path("t.jsonl"), Path("v.jsonl"), Path("l.pt"),
                                            Path("b.pt"), 200))
        self.assertEqual((args.out, args.best_out, args.eval_data), ("l.pt", "b.pt", "v.jsonl"))
        self.assertEqual(sft.EPOCHS, 3)


class ReportTests(unittest.TestCase):
    def test_final_training_report_reads_last_json_object(self):
        stdout = "step=1 loss=5.0\n{\n  \"old\": 1\n}\nstep=2 loss=4.0\n" + json.dumps(
            {"steps": 2, "completed_steps": 2, "duration_seconds": 10.0}, indent=2) + "\n"
        report = pretrain.final_training_report(stdout)
        self.assertEqual(report["steps"], 2)
        self.assertEqual(pretrain.measured_seconds_per_step(report), 5.0)
        # A warning printed after the report (stderr is merged) must not break parsing.
        self.assertEqual(pretrain.final_training_report(stdout + "UserWarning: leaked semaphore\n")["steps"], 2)
        with self.assertRaises(RuntimeError):
            pretrain.final_training_report("step=1 loss=5.0\n")

    def test_seconds_per_step_ignores_empty_runs(self):
        self.assertIsNone(pretrain.measured_seconds_per_step({"completed_steps": 0, "duration_seconds": 5}))
        self.assertIsNone(pretrain.measured_seconds_per_step({}))

    def test_bits_per_byte_converts_nats(self):
        self.assertAlmostEqual(pretrain.bits_per_byte(math.log(2)), 1.0)
        self.assertIsNone(pretrain.bits_per_byte(None))


class InputTests(unittest.TestCase):
    def test_find_one_requires_exactly_one_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "found 0"):
                pretrain.find_one("x.jsonl", root)
            (root / "a").mkdir()
            (root / "a/x.jsonl").write_text("")
            self.assertEqual(pretrain.find_one("x.jsonl", root), root / "a/x.jsonl")
            (root / "b").mkdir()
            (root / "b/x.jsonl").write_text("")
            with self.assertRaisesRegex(RuntimeError, "found 2"):
                pretrain.find_one("x.jsonl", root)

    def test_latest_checkpoint_prefers_highest_session(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for session in (2, 3):
                project = root / f"slm-160m-pretrain-{session}/slm-project"
                (project / "artifacts").mkdir(parents=True)
                (project / "artifacts" / sft.PRETRAIN_NAME).write_text("")
                (project / f"pretrain_session_{session}.json").write_text("{}")
            latest = sft.latest_checkpoint(root=root)
            self.assertEqual(sft.session_number(latest), 3)
            (root / "slm-160m-pretrain-3/slm-project/pretrain_session_3.json").unlink()
            (root / "slm-160m-pretrain-3/slm-project/pretrain_session_2.json").write_text("{}")
            with self.assertRaisesRegex(RuntimeError, "latest"):
                sft.latest_checkpoint(root=root)


class SftTests(unittest.TestCase):
    def test_steps_cover_epochs_within_bounds(self):
        self.assertEqual(sft.sft_steps(32_000, 32, 3), 3000)
        self.assertEqual(sft.sft_steps(10, 32, 3), sft.MIN_STEPS)
        self.assertEqual(sft.sft_steps(10_000_000, 32, 3), sft.MAX_STEPS)
        with self.assertRaises(ValueError):
            sft.sft_steps(0)

    def test_probes_in_training_flags_repeated_prompts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sft_train.jsonl"
            prompt = evaluate.ENGLISH_PROBES[1][1]
            path.write_text(json.dumps({"prompt": "  " + prompt.upper()}) + "\n"
                            + json.dumps({"prompt": "Unrelated question?"}) + "\n")
            self.assertEqual(sft.probes_in_training(path), [evaluate.ENGLISH_PROBES[1][0]])

    def test_holdout_prompts_in_training_are_detected(self):
        rows = json.loads((ROOT / "data/simple_questions_holdout.json").read_text())["rows"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sft_train.jsonl"
            path.write_text(json.dumps({"prompt": "Unrelated question?"}) + "\n")
            self.assertEqual(sft.holdout_prompts_in_training(path), [])
            path.write_text(path.read_text() + json.dumps({"prompt": rows[4]["prompt"].lower() + " "}) + "\n")
            self.assertEqual(sft.holdout_prompts_in_training(path), [rows[4]["id"]])

    def test_ship_best_compacts_the_best_checkpoint_and_drops_the_last_step(self):
        from cognition_slm.config import ModelConfig

        class JsonTorch:
            """Stands in for torch.save and torch.load, which load_checkpoint_payload and ship_best call."""

            def save(self, payload, path):
                Path(path).write_text(json.dumps(payload))

            def load(self, path, **_):
                return json.loads(Path(path).read_text())

        fake = JsonTorch()
        with tempfile.TemporaryDirectory() as directory:
            best, last = Path(directory) / sft.OUTPUT_NAME, Path(directory) / sft.LAST_STEP_NAME
            with self.assertRaisesRegex(RuntimeError, "no validation finished"):
                sft.ship_best(fake, best, last)
            fake.save({"model_config": ModelConfig().to_dict(), "model_state_dict": {"w": 2},
                       "optimizer_state_dict": {"state": {}}, "metadata": {"best_step": 250}}, best)
            fake.save({"model_config": ModelConfig().to_dict(), "model_state_dict": {"w": 3}}, last)
            self.assertEqual(sft.ship_best(fake, best, last), {"best_step": 250})
            self.assertFalse(last.exists())
            self.assertEqual(sorted(path.name for path in Path(directory).iterdir()), [sft.OUTPUT_NAME])
            shipped = fake.load(best)
            self.assertEqual(sorted(shipped), ["metadata", "model_config", "model_state_dict"])
            self.assertEqual(shipped["model_state_dict"], {"w": 2})

    def test_report_records_the_shipped_best_loss_beside_the_last_steps(self):
        training = {"steps": 3000, "best_step": 1500, "best_eval_lm_loss": 1.25,
                    "validation": {"step": 3000, "lm_loss": 1.5}}
        self.assertEqual(sft.loss_summary(training), {"eval_lm_loss": 1.25, "best_step": 1500,
                                                      "best_eval_lm_loss": 1.25, "final_step": 3000,
                                                      "final_eval_lm_loss": 1.5})


class EvaluateTests(unittest.TestCase):
    def test_probes_are_unique_and_outside_the_holdout(self):
        rows = json.loads((ROOT / "data/simple_questions_holdout.json").read_text())["rows"]
        holdout = {sft.normalize(row["prompt"]) for row in rows}
        prompts = [sft.normalize(prompt) for _, prompt, _ in evaluate.ENGLISH_PROBES]
        self.assertEqual(len(set(prompts)), len(prompts))
        self.assertEqual(len({identifier for identifier, _, _ in evaluate.ENGLISH_PROBES}), len(prompts))
        self.assertFalse(holdout & set(prompts))

    def test_probes_are_not_training_prompts(self):
        # Five probes repeated short_facts prompts word for word, so they measured recall of trained rows.
        from compute import short_facts
        from compute.stage3_sft_data import question_key

        trained = {question_key(row["prompt"]) for row in short_facts.short_fact_rows()}
        self.assertEqual([identifier for identifier, prompt, _ in evaluate.ENGLISH_PROBES
                          if question_key(prompt) in trained], [])

    def test_eval_asks_with_the_task_tag_sft_trains_on(self):
        # The holdout tags two rows code_explanation, a prompt prefix no SFT record carries.
        from compute.stage3_sft_data import sft_record

        self.assertEqual(sft_record("x", "What is 2 * 6?", "12", "test", "CC0-1.0")["task_type"], evaluate.TASK_TYPE)
        self.assertNotIn('row["task_type"]', Path(evaluate.__file__).read_text())

    def test_looks_english(self):
        self.assertTrue(evaluate.looks_english("There are 7 days in a week."))
        self.assertTrue(evaluate.looks_english("12"))
        self.assertFalse(evaluate.looks_english(""))
        self.assertFalse(evaluate.looks_english("\x00\x01\x02\x03 ab"))
        self.assertFalse(evaluate.looks_english("这是中文句子"))

    def test_pass_gate_needs_more_than_three_exact(self):
        from score_holdout import score_predictions

        rows = json.loads((ROOT / "data/simple_questions_holdout.json").read_text())["rows"]
        scored = [row for row in rows if row["category"] != "unknown"]
        predictions = [{"id": row["id"], "answer": row["expected_rubric"]} for row in scored[:4]]
        gate = evaluate.pass_gate(score_predictions(rows, predictions))
        self.assertTrue(gate["passed"])
        self.assertEqual((gate["exact"], gate["scored"], gate["manual_review_rows"]), (4, 22, 2))
        gate = evaluate.pass_gate(score_predictions(rows, predictions[:3]))
        self.assertFalse(gate["passed"])


if __name__ == "__main__":
    unittest.main()
