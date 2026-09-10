import unittest

from scripts.kaggle_short_qa_pilot import prepare_phase_payload


class ShortQAPhaseTests(unittest.TestCase):
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
