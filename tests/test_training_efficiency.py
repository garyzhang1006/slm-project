import unittest

import test_train


class EfficiencyTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_train.TrainingIntegrationTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_cpu_fused_rejected(self):
        from cognition_slm.train import train
        with self.assertRaisesRegex(ValueError, "fused-adamw requires a CUDA"):
            train(self.fixture.args(fused_adamw=True))

    def test_counters_match_real_supervision(self):
        from cognition_slm.data import load_jsonl, encode_examples
        from cognition_slm.tokenizer import ByteTokenizer
        from cognition_slm.train import train
        import random
        args = self.fixture.args(steps=2, gradient_accumulation_steps=2)
        encoded = encode_examples(load_jsonl(args.data), ByteTokenizer(), 256)
        expected_input = expected_targets = 0
        for step in (1, 2):
            rng = random.Random(args.seed + step)
            for _ in range(args.batch_size * args.gradient_accumulation_steps):
                item = encoded[rng.randrange(len(encoded))]
                expected_input += len(item["input_ids"])
                expected_targets += len(item["input_ids"]) - item["answer_start"]
        result = train(args)
        self.assertEqual(result["processed_input_tokens"], expected_input)
        self.assertEqual(result["supervised_tokens"], expected_targets)
        self.assertEqual(result["updated_supervised_tokens"], expected_targets)
        self.assertGreater(result["supervised_tokens_per_second"], 0)

    def test_cuda_fused_resume_preserves_moments(self):
        import torch
        from cognition_slm.train import train
        if not torch.cuda.is_available():
            self.skipTest("CUDA required for fused optimizer")
        parent = self.fixture.root / "parent.pt"
        train(self.fixture.args(out=str(parent), device="cuda", steps=1))
        result = train(self.fixture.args(resume=str(parent), device="cuda", steps=2,
                                         fused_adamw=True))
        saved = torch.load(self.fixture.root / "model.pt", weights_only=True)
        self.assertEqual(result["resumed_from_step"], 1)
        self.assertTrue(all(group["fused"] for group in saved["optimizer_state_dict"]["param_groups"]))
        self.assertTrue(all(int(state["step"]) == 2 for state in saved["optimizer_state_dict"]["state"].values()))
