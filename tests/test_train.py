import unittest

from cognition_slm.train import _lr_scale


class TrainingTests(unittest.TestCase):
    def test_warmup_and_cosine_schedule_boundaries(self):
        self.assertAlmostEqual(_lr_scale(0, total_steps=20, warmup_steps=5), 0.2)
        self.assertAlmostEqual(_lr_scale(4, total_steps=20, warmup_steps=5), 1.0)
        self.assertAlmostEqual(_lr_scale(19, total_steps=20, warmup_steps=5), 0.0)
        self.assertAlmostEqual(_lr_scale(0, total_steps=20, warmup_steps=0), 1.0)


if __name__ == "__main__":
    unittest.main()
