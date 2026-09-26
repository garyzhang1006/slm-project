import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "compute" / "lora_eval.py"


def runner():
    # Kaggle bundles for other stages may omit compute/, so skip rather than fail there.
    if not RUNNER.is_file():
        raise unittest.SkipTest("compute/lora_eval.py is not packaged here")
    spec = importlib.util.spec_from_file_location("compute_lora_eval", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scores(**exact):
    return {"scores": {"categories": {name: {"exact": value, "scored": 10} for name, value in exact.items()}}}


class LoraEvalTests(unittest.TestCase):
    def setUp(self):
        self.module = runner()

    def test_import_has_no_heavy_dependencies(self):
        for name in ("torch", "transformers", "peft"):
            self.assertNotIn(name, vars(self.module))

    def test_eval_files_exist_and_have_report_keys(self):
        from scripts.score_holdout import REPORT_KEYS
        for name in self.module.EVAL_FILES:
            self.assertTrue((ROOT / name).is_file(), name)
            self.assertIn(Path(name).name, REPORT_KEYS)

    def test_adapter_dir_needs_exactly_one_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(RuntimeError):
                self.module.adapter_dir(root)
            adapter = root / "slm-lora-baseline" / "artifacts" / "lora-adapter"
            adapter.mkdir(parents=True)
            (adapter / "adapter_config.json").write_text("{}")
            self.assertEqual(self.module.adapter_dir(root), adapter)
            other = root / "copy"
            other.mkdir()
            (other / "adapter_config.json").write_text("{}")
            with self.assertRaises(RuntimeError):
                self.module.adapter_dir(root)

    def test_compare_lines_up_categories(self):
        result = self.module.compare(scores(arithmetic=3, plurals=5), scores(arithmetic=8, abstain=4))
        self.assertEqual(result["arithmetic"], {"base": 3, "adapter": 8, "scored": 10})
        self.assertEqual(result["plurals"], {"base": 5, "adapter": 0, "scored": 0})
        self.assertEqual(result["abstain"], {"base": 0, "adapter": 4, "scored": 10})

    def test_refuses_to_run_off_kaggle(self):
        if Path("/kaggle/working").is_dir():
            self.skipTest("running on Kaggle")
        with self.assertRaises(RuntimeError):
            self.module.main([])


if __name__ == "__main__":
    unittest.main()
