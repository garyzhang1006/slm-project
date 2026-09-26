import base64
import contextlib
import importlib
import io
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def module(name):
    # Kaggle bundles only what a stage needs, so skip when the compute folder is absent.
    if not (ROOT / "compute" / f"{name}.py").exists():
        raise unittest.SkipTest(f"compute/{name}.py is not packaged here")
    return importlib.import_module(f"compute.{name}")


def fake_root(directory: Path, runner: str) -> Path:
    for name in ("pyproject.toml", "src/cognition_slm/__init__.py", "src/cognition_slm/web/index.html",
                 "src/cognition_slm/web/style.css", "src/cognition_slm/web/app.js",
                 "scripts/score_holdout.py", "data/simple_questions_holdout.json",
                 "compute/stages.py", runner):
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {name}\n")
    # A stray checkpoint must never enter the payload.
    (directory / "artifacts").mkdir()
    (directory / "artifacts/model.pt").write_bytes(b"weights")
    return directory


class StageTableTests(unittest.TestCase):
    def test_stage_entries_follow_the_interface(self):
        stages = module("stages")
        self.assertEqual(set(stages.STAGES), {"corpus", "pretrain", "sft_data", "sft", "eval", "lora", "lora_eval"})
        for name, spec in stages.STAGES.items():
            self.assertEqual(set(spec), {"runner", "slug", "internet", "gpu", "attaches"}, name)
            self.assertRegex(spec["runner"], r"^compute/\w+\.py$")
        self.assertFalse(stages.STAGES["sft_data"]["gpu"])
        self.assertTrue(stages.STAGES["lora"]["internet"])
        self.assertEqual(stages.stage_attaches("lora_eval"), ["slm-lora-baseline"])

    def test_planned_steps(self):
        stages = module("stages")
        self.assertEqual(stages.planned_steps(1000, 10, 100), 90)
        self.assertEqual(stages.planned_steps(50, 10, 100), 0)
        with self.assertRaises(ValueError):
            stages.planned_steps(1000, 0, 0)
        with self.assertRaises(ValueError):
            stages.planned_steps(-1, 1, 0)

    def test_pretrain_budget_constants(self):
        stages = module("stages")
        self.assertEqual(stages.TOKENS_PER_STEP, 65_536)
        self.assertEqual(stages.PRETRAIN_TOTAL_STEPS, 22_889)
        self.assertLess(stages.PRETRAIN_SESSION_SECONDS, 12 * 3600)
        per_session = stages.planned_steps(stages.PRETRAIN_SESSION_SECONDS,
                                           stages.SECONDS_PER_STEP_ESTIMATE, stages.SESSION_RESERVE_SECONDS)
        self.assertEqual(stages.pretrain_sessions(), -(-stages.PRETRAIN_TOTAL_STEPS // per_session))
        self.assertEqual(stages.pretrain_sessions(100, 1, 60, 10), 2)

    def test_slugs_and_attachments_chain_sessions(self):
        stages = module("stages")
        self.assertEqual(stages.stage_slug("pretrain", 3), "slm-160m-pretrain-3")
        self.assertEqual(stages.stage_attaches("pretrain", 1), ["slm-160m-corpus"])
        self.assertEqual(stages.stage_attaches("pretrain", 3), ["slm-160m-corpus", "slm-160m-pretrain-2"])
        self.assertEqual(stages.stage_attaches("sft", pretrain_session=2), ["slm-160m-pretrain-2", "slm-sft-data"])
        last = stages.pretrain_sessions()
        self.assertEqual(stages.stage_attaches("sft")[0], f"slm-160m-pretrain-{last}")
        for bad in ((lambda: stages.stage_slug("pretrain")), (lambda: stages.stage_slug("pretrain", 0)),
                    (lambda: stages.stage_slug("sft", 2)), (lambda: stages.stage_slug("nope"))):
            with self.assertRaises(ValueError):
                bad()


class PackageTests(unittest.TestCase):
    def build(self, stage, session=None):
        package = module("package")
        runner = package.STAGES[stage]["runner"]
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = fake_root(Path(directory.name) / "project", runner)
        out = Path(directory.name) / "out"
        with contextlib.redirect_stdout(io.StringIO()):
            package.prepare(stage, out, "someone", session, root=root)
        return out, runner

    def payload(self, out):
        script = (out / "run.py").read_text()
        compile(script, "run.py", "exec")
        encoded = re.search(r"payload = '([^']+)'", script).group(1)
        return script, zipfile.ZipFile(io.BytesIO(base64.b64decode(encoded)))

    def test_pretrain_session_kernel(self):
        out, runner = self.build("pretrain", session=2)
        metadata = json.loads((out / "kernel-metadata.json").read_text())
        self.assertEqual(metadata["id"], "someone/slm-160m-pretrain-2")
        self.assertEqual(metadata["kernel_sources"], ["someone/slm-160m-corpus", "someone/slm-160m-pretrain-1"])
        self.assertEqual(metadata["machine_shape"], "NvidiaTeslaT4")
        self.assertTrue(metadata["enable_gpu"])
        self.assertFalse(metadata["enable_internet"])
        script, archive = self.payload(out)
        self.assertIn("['--session', '2']", script)
        names = set(archive.namelist())
        self.assertIn(runner, names)
        self.assertIn("scripts/score_holdout.py", names)
        self.assertNotIn("artifacts/model.pt", names)
        manifest = json.loads(archive.read("source-manifest.json"))
        self.assertEqual(manifest, json.loads((out / "source-manifest.json").read_text()))
        self.assertEqual(set(manifest), names - {"source-manifest.json"})

    def test_cpu_stage_has_no_gpu_shape_or_session_argv(self):
        out, _ = self.build("sft_data")
        metadata = json.loads((out / "kernel-metadata.json").read_text())
        self.assertFalse(metadata["enable_gpu"])
        self.assertTrue(metadata["enable_internet"])
        self.assertNotIn("machine_shape", metadata)
        self.assertEqual(metadata["kernel_sources"], [])
        script, _ = self.payload(out)
        self.assertIn("] + []", script)

    def test_cli_rejects_flags_for_the_wrong_stage(self):
        package = module("package")
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "out"
            for argv in (["--stage", "corpus", "--pretrain-session", "2", "--out", str(out)],
                         ["--stage", "corpus", "--session", "1", "--out", str(out)],
                         ["--stage", "pretrain", "--out", str(out)]):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    package.main(argv)
            self.assertFalse(out.exists())

    def test_missing_runner_is_reported(self):
        package = module("package")
        with tempfile.TemporaryDirectory() as directory:
            root = fake_root(Path(directory) / "project", "compute/other.py")
            with self.assertRaisesRegex(FileNotFoundError, "stage4_sft.py"):
                package.prepare("sft", Path(directory) / "out", root=root)
            self.assertFalse((Path(directory) / "out").exists())


if __name__ == "__main__":
    unittest.main()
