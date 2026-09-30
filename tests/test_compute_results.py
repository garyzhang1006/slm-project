import json
from pathlib import Path
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
    return {"adapter_sha256": sha,
            "base": {"everyday_eval": {"predictions": predictions(EVERYDAY, False)},
                     "simple_questions": {"predictions": predictions(HOLDOUT, False)}},
            "lora": {"everyday_eval": {"predictions": predictions(EVERYDAY, True)},
                     "simple_questions": {"predictions": predictions(HOLDOUT, True)}}}


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
        self.assertIn("eval LM loss 1.200", text)
        self.assertIn("Held-out text bits/byte: 1.100", text)
        self.assertNotIn("—", text)

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
        small = text.split("## SmolLM2-360M-Instruct: LoRA vs base", 1)[1].split("## LoRA adapter", 1)[0]
        self.assertIn("No finished lora_eval report yet.", small)

    def test_main_writes_the_file(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "RESULTS.md"
            fake = mock.Mock()
            fake.report.return_value = None
            with mock.patch.object(collect_results, "Kaggle", return_value=fake), mock.patch("builtins.print"):
                self.assertEqual(collect_results.main(["--owner", "someone", "--out", str(out)]), 0)
            self.assertTrue(out.read_text().startswith("# Results"))


if __name__ == "__main__":
    unittest.main()
