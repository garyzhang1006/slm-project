import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if not (ROOT / "scripts" / "score_holdout.py").exists():
    raise unittest.SkipTest("score_holdout.py is not packaged here")
from compute import collect_results  # noqa: E402

EVERYDAY = json.loads((ROOT / "data/everyday_eval.json").read_text())["rows"]
HOLDOUT = json.loads((ROOT / "data/simple_questions_holdout.json").read_text())["rows"]


def predictions(rows, right):
    return [{"id": row["id"], "answer": row["expected_rubric"] if right else "zzz"} for row in rows]


def adapter_report(sha="ab" * 32):
    return {"status": "complete_pending_manual_review", "adapter_sha256": sha,
            "base": {"everyday_eval": {"predictions": predictions(EVERYDAY, False)},
                     "simple_questions": {"predictions": predictions(HOLDOUT, False)}},
            "lora": {"everyday_eval": {"predictions": predictions(EVERYDAY, True)},
                     "simple_questions": {"predictions": predictions(HOLDOUT, True)}}}


def write_report(directory: Path, slug: str, name: str, value) -> None:
    # Where Kaggle mounts an attached kernel's output, as a 2026-10 distill_manifest.json recorded it.
    path = directory / "notebooks" / "someone" / slug / "slm-project" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value if isinstance(value, str) else json.dumps(value))


class LocalReportTests(unittest.TestCase):
    def test_the_slug_picks_the_kernel_when_two_hold_the_same_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_report(root, "slm-lora-baseline", "lora_report.json", {"status": "small"})
            write_report(root, "slm-lora-1b7", "lora_report.json", {"status": "large"})
            # A slug that only starts with another kernel's slug names a different kernel.
            write_report(root, "slm-lora-1b7-eval", "lora_report.json", {"status": "other"})
            self.assertEqual(collect_results.local_report(root, "slm-lora-baseline", "lora_report.json"),
                             {"status": "small"})
            self.assertEqual(collect_results.local_report(root, "slm-lora-1b7", "lora_report.json"),
                             {"status": "large"})

    def test_missing_ambiguous_or_unreadable_reports_read_as_none(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_report(root, "slm-160m-eval", "eval_report.json", "{not json")
            write_report(root, "slm-160m-sft", "sft_report.json", "[1, 2]")
            write_report(root, "slm-distill-data", "a/distill_manifest.json", {"rows": 1})
            write_report(root, "slm-distill-data", "b/distill_manifest.json", {"rows": 2})
            for slug, filename in (("slm-160m-eval", "eval_report.json"), ("slm-160m-sft", "sft_report.json"),
                                   ("slm-distill-data", "distill_manifest.json"),
                                   ("slm-lora-eval", "lora_eval_report.json"),
                                   ("slm-160m-eval", "sft_report.json")):
                with self.subTest(slug=slug, filename=filename):
                    self.assertIsNone(collect_results.local_report(root, slug, filename))
            self.assertIsNone(collect_results.local_report(root / "absent", "slm-160m-eval", "eval_report.json"))


class CollectTests(unittest.TestCase):
    def test_collect_reads_sessions_until_the_first_gap(self):
        calls = []

        def fetch(slug, filename):
            calls.append((slug, filename))
            return {"session": 1} if filename == "pretrain_session_1.json" else None

        results = collect_results.collect(fetch)
        self.assertEqual(results["pretrain"], [{"session": 1}])
        self.assertIn(("slm-160m-pretrain-2", "pretrain_session_2.json"), calls)
        self.assertNotIn(("slm-160m-pretrain-3", "pretrain_session_3.json"), calls)
        self.assertIn(("slm-lora-eval", "lora_eval_report.json"), calls)
        self.assertIsNone(results["eval"])

    def test_rescore_uses_current_answer_keys(self):
        scores = collect_results.rescore(adapter_report())
        self.assertEqual(set(scores), {"everyday_eval", "simple_questions"})
        everyday = scores["everyday_eval"]
        self.assertEqual(everyday["base"]["total"]["exact"], 0)
        self.assertEqual(everyday["lora"]["total"]["exact"], everyday["lora"]["total"]["scored"])
        self.assertEqual(collect_results.rescore({}), {})


class RenderTests(unittest.TestCase):
    def test_empty_results_say_what_is_missing(self):
        empty = {"pretrain": [], "lora": None, "lora_eval": None, "distill": None, "sft": None, "eval": None}
        text = collect_results.render(empty, {}, "2026-09-26 15:00")
        for phrase in ("No finished pretrain session yet.", "No LoRA report yet.", "No finished lora_eval report",
                       "No distill_data manifest yet.", "No SFT report yet.", "No eval report yet."):
            self.assertIn(phrase, text)

    def test_full_results(self):
        scored = {"total": {"exact": 18, "scored": 22}}
        results = {
            "pretrain": [{"session": 1, "step_reached": 4570, "seconds_per_step": 8.72,
                          "eval_bits_per_byte": 1.289, "status": "session_complete_resume_next"}],
            "lora": {"status": "complete_pending_manual_review", "adapter_sha256": "cd" * 32,
                     "baseline": {"scores": {"total": {"exact": 5, "scored": 22}}}, "final": {"scores": scored},
                     "training": {"initial_eval_loss": 1.96, "best_eval_loss": 1.55, "best_step": 750,
                                  "final_eval_loss": 1.6}},
            "lora_eval": adapter_report(),
            "distill": {"rows": 5000, "prompts": 6000, "dropped": {"empty_or_invalid": 1000}},
            "sft": {"status": "complete", "pretrain_session": 6, "training": {"steps": 3000},
                    "eval_lm_loss": 1.2},
            "eval": {"simple_questions_scores": scored, "everyday_eval_scores": scored,
                     "looks_english_rate": 0.9, "heldout_text": {"bits_per_byte": 1.1}}}
        text = collect_results.render(results, collect_results.rescore(results["lora_eval"]), "now")
        self.assertIn("| 1 | 4570 | 8.72 | 1.289 | session_complete_resume_next |", text)
        self.assertIn("Holdout exact: base 5/22, adapter 18/22", text)
        self.assertIn("best 1.550 at step 750", text)
        self.assertIn("different adapter", text)  # lora_eval used ab..., the LoRA report says cd...
        rows = len(EVERYDAY) - sum(row["category"] == "unknown" for row in EVERYDAY)
        self.assertIn(f"base 0/{rows} exact and 0/{rows} contains, LoRA {rows}/{rows} exact", text)
        self.assertIn("| base exact | LoRA exact | base contains | LoRA contains | scored |", text)
        self.assertIn("| id | prompt | LoRA answer |", text)
        manual = [row["id"] for row in HOLDOUT if row["category"] == "unknown"]
        self.assertTrue(manual)
        for identifier in manual:
            self.assertIn(f"| {identifier} |", text)
        self.assertIn("5000 rows kept from 6000 prompts (dropped: empty_or_invalid 1000).", text)
        self.assertIn("3000 steps, eval LM loss 1.200.", text)
        results["sft"]["best_step"] = 1500
        text = collect_results.render(results, collect_results.rescore(results["lora_eval"]), "now")
        self.assertIn("3000 steps, eval LM loss 1.200 from step 1500.", text)
        self.assertIn("Held-out text bits/byte: 1.100", text)
        self.assertNotIn("—", text)

    def test_eval_of_an_older_sft_checkpoint_is_flagged(self):
        results = {"pretrain": [], "lora": None, "lora_eval": None, "distill": None,
                   "sft": {"status": "complete", "sha256": "new"}, "eval": {"sha256": "old"}}
        self.assertIn("older SFT checkpoint", collect_results.render(results, {}, "now"))
        results["eval"]["sha256"] = "new"
        self.assertNotIn("older SFT checkpoint", collect_results.render(results, {}, "now"))

    def test_slm_eval_is_rescored_with_the_current_keys(self):
        # The stored counts stand for keys that changed after the kernel ran; the saved answers decide.
        stale = {"total": {"exact": 999, "contains": 999, "scored": 999}}
        evaluation = {"simple_questions": [{**row, "answer": "zzz"} for row in HOLDOUT],
                      "everyday_eval": [{**row, "answer": row["expected_rubric"]} for row in EVERYDAY],
                      "simple_questions_scores": stale, "everyday_eval_scores": stale}
        results = {"pretrain": [], "lora": None, "lora_eval": None, "distill": None, "sft": None, "eval": evaluation}
        text = collect_results.render(results, {}, "now")
        holdout = len(HOLDOUT) - sum(row["category"] == "unknown" for row in HOLDOUT)
        everyday = len(EVERYDAY) - sum(row["category"] == "unknown" for row in EVERYDAY)
        self.assertIn(f"- Holdout exact: 0/{holdout}\n", text)
        self.assertIn(f"- Everyday exact: {everyday}/{everyday}\n", text)
        self.assertNotIn("999/999", text)
        # An older report kept no answers, so its stored counts are shown and marked as the kernel's.
        results["eval"] = {"simple_questions_scores": stale}
        text = collect_results.render(results, {}, "now")
        self.assertIn("- Holdout exact: 999/999 (scored with the kernel's answer keys)", text)
        self.assertIn("- Everyday exact: n/a\n", text)

    def test_lora_holdout_line_is_rescored_from_saved_answers(self):
        stale = {"total": {"exact": 22, "contains": 22, "scored": 22}}
        lora = {"status": "complete_pending_manual_review",
                "baseline": {"predictions": [{**row, "answer": "zzz"} for row in HOLDOUT], "scores": stale},
                "final": {"predictions": [{**row, "answer": "zzz"} for row in HOLDOUT], "scores": stale}}
        text = "\n".join(collect_results.lora_sections("m", lora, None, {}))
        scored = len(HOLDOUT) - sum(row["category"] == "unknown" for row in HOLDOUT)
        self.assertIn(f"- Holdout exact: base 0/{scored}, adapter 0/{scored}\n", text)
        self.assertNotIn("22/22", text)
        del lora["final"]["predictions"]
        text = "\n".join(collect_results.lora_sections("m", lora, None, {}))
        self.assertIn(f"- Holdout exact: base 0/{scored}, adapter 22/22 (scored with the kernel's answer keys)",
                      text)

    def test_manual_review_cells_stay_on_one_table_row(self):
        report = adapter_report()
        identifier = next(row["id"] for row in HOLDOUT if row["category"] == "unknown")
        for row in report["lora"]["simple_questions"]["predictions"]:
            if row["id"] == identifier:
                # A lone carriage return ends a Markdown line, so it would split the table row.
                row.update(prompt="Who is\tit?", answer="I do not\rknow | sorry\nreally")
        empty = {"pretrain": [], "lora": None, "lora_eval": report, "distill": None, "sft": None, "eval": None}
        text = collect_results.render(empty, collect_results.rescore(report), "now")
        self.assertIn(f"| {identifier} | Who is it? | I do not know / sorry really |", text.splitlines())

    def test_large_adapter_gets_its_own_sections(self):
        empty = {"pretrain": [], "lora": None, "lora_eval": None, "distill": None, "sft": None, "eval": None,
                 "lora_1b7": {"status": "complete_pending_manual_review", "adapter_sha256": "ab" * 32},
                 "lora_1b7_eval": adapter_report()}
        text = collect_results.render(empty, {}, "now", collect_results.rescore(adapter_report()))
        self.assertIn("## LoRA adapter (SmolLM2-1.7B-Instruct)", text)
        large = text.split("## SmolLM2-1.7B-Instruct: LoRA vs base", 1)[1]
        self.assertIn("| category | base exact |", large)
        # Both reports name the same adapter, so the note that the predictions came from another one stays out.
        self.assertNotIn("different adapter", large)
        small = text.split("## SmolLM2-360M-Instruct: LoRA vs base", 1)[1].split("## LoRA adapter", 1)[0]
        self.assertIn("No finished lora_eval report yet.", small)

    def test_readme_quotes_the_everyday_scores_in_results(self):
        # A re-score with wider answer keys moved RESULTS.md to 149 and 102, and README kept 141 and 100.
        if not (ROOT / "compute/RESULTS.md").exists():
            self.skipTest("compute/RESULTS.md is not packaged here")
        results = (ROOT / "compute/RESULTS.md").read_text()
        small, large = re.findall(r"everyday_eval: base \d+/252 exact and \d+/252 contains, LoRA (\d+)/252 exact",
                                  results)
        self.assertIn(f"({large} and {small} of 252)", (ROOT / "README.md").read_text())

    def test_main_writes_the_file(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "RESULTS.md"
            fake = mock.Mock()
            fake.report.return_value = None
            with mock.patch.object(collect_results, "Kaggle", return_value=fake), mock.patch("builtins.print"):
                self.assertEqual(collect_results.main(["--owner", "someone", "--out", str(out)]), 0)
            self.assertTrue(out.read_text().startswith("# Results"))

    def test_main_keeps_answers_only_from_reports_that_came_back(self):
        def report(slug, filename):
            return adapter_report() if (slug, filename) == ("slm-lora-eval", "lora_eval_report.json") else None

        with tempfile.TemporaryDirectory() as directory:
            reports = Path(directory) / "reports"
            reports.mkdir()
            # The 1.7B report fails to download this time, so its answers from an earlier collection must survive.
            kept = reports / "lora_eval_1b7.json"
            kept.write_text('{"kernel": "slm-lora-1b7-eval"}\n')
            fake = mock.Mock()
            fake.report.side_effect = report
            with mock.patch.object(collect_results, "Kaggle", return_value=fake), mock.patch("builtins.print"):
                self.assertEqual(collect_results.main(["--owner", "someone", "--out", str(Path(directory) / "R.md"),
                                                       "--reports-dir", str(reports)]), 0)
            self.assertEqual(kept.read_text(), '{"kernel": "slm-lora-1b7-eval"}\n')
            text = (reports / "lora_eval_360m.json").read_text()
            saved = json.loads(text)
            self.assertEqual(text, json.dumps(saved, indent=2, sort_keys=True) + "\n")
            self.assertEqual(saved["kernel"], "slm-lora-eval")
            self.assertEqual(saved["adapter_sha256"], "ab" * 32)
            self.assertEqual(set(saved["predictions"]), set(collect_results.EVAL_FILES))
            everyday = saved["predictions"]["data/everyday_eval.json"]
            self.assertEqual(everyday["lora"], predictions(EVERYDAY, True))
            self.assertEqual([row["id"] for row in everyday["base"]], [row["id"] for row in EVERYDAY])
            self.assertEqual(saved["predictions"]["data/simple_questions_holdout.json"]["base"],
                             predictions(HOLDOUT, False))

    def test_a_report_from_a_run_that_died_partway_keeps_the_saved_answers(self):
        # lora_eval.py rewrites its report after every set, so a run that dies keeps base answers and no LoRA ones.
        with tempfile.TemporaryDirectory() as directory:
            reports = Path(directory)
            partial = {**adapter_report(), "status": "evaluating", "lora": {}}
            self.assertEqual(collect_results.write_predictions({"lora_1b7_eval": partial}, reports), [])
            self.assertEqual(list(reports.iterdir()), [])
            self.assertEqual(collect_results.write_predictions({"lora_1b7_eval": adapter_report()}, reports),
                             [reports / "lora_eval_1b7.json"])

    def test_main_reads_attached_outputs_without_the_kaggle_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs, reports, out = (Path(directory) / name for name in ("input", "reports", "RESULTS.md"))
            write_report(inputs, "slm-160m-pretrain-1", "pretrain_session_1.json",
                         {"session": 1, "step_reached": 4570, "seconds_per_step": 8.72,
                          "eval_bits_per_byte": 1.289, "status": "session_complete_resume_next"})
            write_report(inputs, "slm-lora-baseline", "lora_report.json",
                         {"status": "small adapter", "adapter_sha256": "ab" * 32})
            write_report(inputs, "slm-lora-1b7", "lora_report.json", {"status": "large adapter"})
            write_report(inputs, "slm-lora-eval", "lora_eval_report.json", adapter_report())
            with mock.patch.object(collect_results, "Kaggle") as kaggle, mock.patch("builtins.print"):
                self.assertEqual(collect_results.main(["--input-dir", str(inputs), "--out", str(out),
                                                       "--reports-dir", str(reports)]), 0)
            kaggle.assert_not_called()
            small, large = out.read_text().split("## LoRA adapter (SmolLM2-1.7B-Instruct)", 1)
            self.assertIn("| 1 | 4570 | 8.72 | 1.289 | session_complete_resume_next |", small)
            self.assertIn("- Status: small adapter", small)
            self.assertIn("- Status: large adapter", large)
            rows = len(EVERYDAY) - sum(row["category"] == "unknown" for row in EVERYDAY)
            self.assertIn(f"LoRA {rows}/{rows} exact", small)
            self.assertNotIn("different adapter", small)
            self.assertEqual(json.loads((reports / "lora_eval_360m.json").read_text())["kernel"], "slm-lora-eval")
            self.assertFalse((reports / "lora_eval_1b7.json").exists())

    def test_main_takes_owner_or_input_dir_but_not_both(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "RESULTS.md"
            # A folder that is not there would turn every report into "not yet" and wipe the real numbers.
            for argv in ([], ["--owner", "someone", "--input-dir", directory],
                         ["--input-dir", str(Path(directory) / "absent")]):
                with self.subTest(argv=argv), mock.patch("sys.stderr"), self.assertRaises(SystemExit):
                    collect_results.main(argv + ["--out", str(out)])
            self.assertFalse(out.exists())

    def test_header_names_the_answer_keys(self):
        empty = {"pretrain": [], "lora": None, "lora_eval": None, "distill": None, "sft": None, "eval": None}
        header = collect_results.render(empty, {}, "now").split("## slm-160m pretraining", 1)[0]
        for name in collect_results.EVAL_FILES:
            digest = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            self.assertIn(f"`{name}` {digest[:12]}", header)


if __name__ == "__main__":
    unittest.main()
