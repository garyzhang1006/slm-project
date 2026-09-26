import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
# Kaggle packages only the runners a kernel needs, so each test skips when its runner is absent.
sys.path.insert(0, str(ROOT / "scripts"))


def runner(name):
    if not (ROOT / "scripts" / f"{name}.py").exists():
        raise unittest.SkipTest(f"{name}.py is not packaged here")
    return importlib.import_module(f"scripts.{name}")


def squad_prompt(context, question):
    return f"Answer from the passage.\n\nPassage: {context}\n\nQuestion: {question}"


class LongRunHorizonTests(unittest.TestCase):
    def test_stage_horizon_is_reachable_within_budget(self):
        long_run = runner("kaggle_long_run")
        self.assertAlmostEqual(long_run.stage_steps(3.5 * 3600), 3142, delta=1)
        self.assertLess(long_run.stage_steps(5.5 * 3600), 100_000)
        self.assertGreater(long_run.stage_steps(1), long_run.WARMUP_STEPS)


class QualityRunTests(unittest.TestCase):
    def test_quality_probes_are_not_training_prompts(self):
        quality = runner("kaggle_500m_quality_run")
        from build_curriculum_data import build_rows

        rows = build_rows("train") + [json.loads(line) for line in
                                      (ROOT / "data/demo.jsonl").read_text().splitlines() if line.strip()]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            self.assertEqual(quality._probes_in_training(quality.QUALITY_PROBES, path), [])
            leaked = (("copied", "  Hello THERE ", "language_generation", 8),)
            path.write_text(json.dumps({"prompt": "hello there"}) + "\n")
            self.assertEqual(quality._probes_in_training(leaked, path), ["copied"])

    def test_failed_training_log_is_kept(self):
        quality = runner("kaggle_500m_quality_run")
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "train.log"
            command = [sys.executable, "-c", "import sys; print('step 1 loss 2.0'); "
                       "sys.stderr.write('RuntimeError: loss is NaN\\n'); sys.exit(3)"]
            with self.assertRaises(RuntimeError):
                quality._run_logged(command, log)
            text = log.read_text()
            self.assertIn("step 1 loss 2.0", text)
            self.assertIn("RuntimeError: loss is NaN", text)


class QARunHoldoutTests(unittest.TestCase):
    def test_holdout_spans_passages_before_repeating_one(self):
        qa = runner("kaggle_qa_run")
        validation = []
        for context in range(8):
            raw = {"context": f"passage {context}", "answers": {"text": [f"a{context}", f"b{context}"]}}
            key = qa.context_key(raw)
            for question in range(5):
                item = dict(raw, id=f"{context}-{question}")
                validation.append((key, item, {"id": item["id"], "context": context}))
        validation.sort(key=lambda item: (item[0], item[1]["id"]))
        for parity in (0, 1):
            contexts = len({key for key, _, _ in validation if int(key, 16) % 2 == parity})
            if contexts < 2:
                continue
            holdout = qa.spread_holdout(validation, parity, contexts)
            self.assertEqual(len({row["context"] for row, _ in holdout}), contexts)
            self.assertEqual(len(holdout[0][1]), 2)
            self.assertEqual(len(qa.spread_holdout(validation, parity, contexts + 1)), contexts + 1)

    def test_shuffled_control_changes_passage(self):
        qa = runner("kaggle_qa_run")
        prompts = [squad_prompt(f"passage {i // 5}", f"question {i}") for i in range(16)]
        shuffled = qa.shuffled_prompts(prompts)
        self.assertEqual(sorted(shuffled), sorted(prompts))
        for original, control in zip(prompts, shuffled):
            self.assertNotEqual(qa.passage(original), qa.passage(control))
        plain = ["What is 2 plus 2?", "Name a colour."]
        self.assertEqual(qa.shuffled_prompts(plain), plain[::-1])
        with self.assertRaises(RuntimeError):
            qa.shuffled_prompts([squad_prompt("same", "one"), squad_prompt("same", "two")])


if __name__ == "__main__":
    unittest.main()
