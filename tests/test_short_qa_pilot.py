import unittest
import json
from pathlib import Path
import tempfile

from scripts.kaggle_short_qa_pilot import prepare_phase_payload, prepare_subset


class ShortQAPhaseTests(unittest.TestCase):
    def test_broadening_preserves_probe_and_excludes_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [{"id": str(i), "prompt": f"Question {i}?", "answer": f"Answer {i}.",
                     "task_type": "language_generation", "confidence": 0.9,
                     "error_category": "none", "source": "databricks/databricks-dolly-15k",
                     "license": "CC-BY-SA-3.0"} for i in range(2200)]
            source = root / "source.jsonl"
            source.write_text("".join(json.dumps(row) + "\n" for row in rows))
            holdout = {"rows": [{"prompt": "Question 0?"}]}
            small = prepare_subset(source, holdout, root)
            broad = prepare_subset(source, holdout, root, broad=True)
            train = broad["splits"]["train"]["ids"]
            probe = broad["splits"]["probe"]["ids"]
            self.assertEqual(probe, small["splits"]["probe"]["ids"])
            self.assertEqual(train[:16], small["splits"]["train"]["ids"][:16])
            self.assertGreater(len(train), 2000)
            self.assertFalse(set(train) & set(probe))
            self.assertNotIn("0", train + probe)

    def test_new_schedule_preserves_learned_state(self):
        weights, moments, rng, scaler = object(), object(), object(), object()
        payload = {
            "model_state_dict": weights,
            "optimizer_state_dict": {
                "state": moments,
                "param_groups": [{"lr": 0.0, "initial_lr": 0.00005}],
            },
            "metadata": {"step": 6742, "learning_rate": 0.00005},
            "scheduler_state_dict": {"last_epoch": 6742},
            "torch_rng_state": rng,
            "scaler_state_dict": scaler,
        }
        result = prepare_phase_payload(payload)
        self.assertIs(result["model_state_dict"], weights)
        self.assertIs(result["optimizer_state_dict"]["state"], moments)
        self.assertIs(result["torch_rng_state"], rng)
        self.assertIs(result["scaler_state_dict"], scaler)
        self.assertEqual(result["metadata"]["step"], 0)
        self.assertNotIn("scheduler_state_dict", result)
        for group in result["optimizer_state_dict"]["param_groups"]:
            self.assertEqual(group["lr"], 0.00005)
            self.assertEqual(group["initial_lr"], 0.00005)
