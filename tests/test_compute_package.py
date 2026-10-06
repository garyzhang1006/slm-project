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
from unittest import mock
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
        self.assertEqual(set(stages.STAGES), {"corpus", "pretrain", "sft_data", "sft", "eval", "lora", "lora_eval",
                                              "distill_data", "lora_1b7", "lora_1b7_eval", "results"})
        for name, spec in stages.STAGES.items():
            self.assertEqual(set(spec) - {"args"}, {"runner", "slug", "internet", "gpu", "attaches"}, name)
            self.assertRegex(spec["runner"], r"^compute/\w+\.py$")
        self.assertFalse(stages.STAGES["sft_data"]["gpu"])
        self.assertTrue(stages.STAGES["lora"]["internet"])
        self.assertEqual(stages.stage_attaches("lora_eval"), ["slm-lora-baseline"])

    def test_stages_that_download_at_run_time_have_internet(self):
        stages = module("stages")
        # These runners pip install, stream datasets or fetch Hugging Face models as they run, so a kernel pushed
        # without internet would fail only after waiting in the queue.
        online = {name for name, spec in stages.STAGES.items() if spec["internet"]}
        self.assertEqual(online, {"corpus", "sft_data", "lora", "distill_data", "lora_eval", "lora_1b7", "lora_1b7_eval"})

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
        self.assertEqual(stages.PRETRAIN_TOTAL_STEPS, 21_562)
        # The schedule stops short of one pass, so no session's shard runs past the end of the corpus.
        self.assertLess(stages.PRETRAIN_TOTAL_STEPS * stages.TOKENS_PER_STEP, stages.PRETRAIN_TARGET_BYTES)
        self.assertLess(stages.PRETRAIN_SESSION_SECONDS, 12 * 3600)
        per_session = stages.planned_steps(stages.PRETRAIN_SESSION_SECONDS,
                                           stages.SECONDS_PER_STEP_ESTIMATE, stages.SESSION_RESERVE_SECONDS)
        self.assertEqual(stages.pretrain_sessions(), -(-stages.PRETRAIN_TOTAL_STEPS // per_session))
        self.assertEqual(stages.pretrain_sessions(100, 1, 60, 10), 2)

    def test_session_five_finishes_the_schedule_at_ten_seconds_a_step(self):
        # A sixth session would train its steps at under 0.6% of the peak rate, so session 5 must end the cosine
        # even on a GPU slower than any measured so far.
        stages, pretrain = module("stages"), module("stage2_pretrain")
        start, slow = 17_662, 10.0
        self.assertEqual((stages.PRETRAIN_SESSIONS_RUN, stages.PRETRAIN_STEP_REACHED), (4, start))
        # stage2_pretrain.main's budget with no setup time; sessions 1 to 4 all trained the full 39,600 s.
        budget = min(stages.PRETRAIN_SESSION_SECONDS,
                     pretrain.KAGGLE_SESSION_LIMIT_SECONDS - pretrain.SETUP_AND_SAVE_RESERVE_SECONDS)
        self.assertEqual(budget, 39_600)
        plan = pretrain.plan_session(start, stages.PRETRAIN_TOTAL_STEPS, stages.TOKENS_PER_STEP, budget, slow, True)
        self.assertEqual(plan["shard_steps"], plan["remaining_steps"])
        expected = min(plan["remaining_steps"], stages.planned_steps(budget, slow, stages.SESSION_RESERVE_SECONDS))
        self.assertEqual(start + expected, stages.PRETRAIN_TOTAL_STEPS)
        # train.py stops at the first step whose duration reaches --max-seconds; the schedule's last step comes first.
        self.assertLess(plan["remaining_steps"] * slow, budget)
        self.assertEqual(stages.last_pretrain_session(), 5)
        self.assertEqual(stages.stage_attaches("sft")[0], "slm-160m-pretrain-5")

    def test_slugs_and_attachments_chain_sessions(self):
        stages = module("stages")
        self.assertEqual(stages.stage_slug("pretrain", 3), "slm-160m-pretrain-3")
        self.assertEqual(stages.stage_attaches("pretrain", 1), ["slm-160m-corpus"])
        self.assertEqual(stages.stage_attaches("pretrain", 3), ["slm-160m-corpus", "slm-160m-pretrain-2"])
        self.assertEqual(stages.stage_attaches("sft", pretrain_session=2), ["slm-160m-pretrain-2", "slm-sft-data", "slm-distill-data"])
        last = stages.last_pretrain_session()
        self.assertEqual(stages.stage_attaches("sft")[0], f"slm-160m-pretrain-{last}")
        for bad in ((lambda: stages.stage_slug("pretrain")), (lambda: stages.stage_slug("pretrain", 0)),
                    (lambda: stages.stage_slug("sft", 2)), (lambda: stages.stage_slug("nope"))):
            with self.assertRaises(ValueError):
                bad()
        with self.assertRaisesRegex(ValueError, "--pretrain-session needs k >= 1, got 0"):
            stages.stage_attaches("sft", pretrain_session=0)

    def test_results_attaches_only_kernels_that_have_run(self):
        stages = module("stages")
        lora = ["slm-lora-baseline", "slm-lora-eval", "slm-lora-1b7", "slm-lora-1b7-eval", "slm-distill-data"]
        run = [f"slm-160m-pretrain-{number}" for number in range(1, stages.PRETRAIN_SESSIONS_RUN + 1)]
        self.assertEqual(stages.stage_attaches("results"), lora + run)
        later = stages.PRETRAIN_SESSIONS_RUN + 1
        self.assertEqual(stages.stage_attaches("results", pretrain_session=later),
                         lora + run + [f"slm-160m-pretrain-{later}"])
        self.assertEqual(stages.stage_attaches("results", pretrain_session=1), lora + ["slm-160m-pretrain-1"])
        with self.assertRaisesRegex(ValueError, "--pretrain-session needs k >= 1, got 0"):
            stages.stage_attaches("results", pretrain_session=0)


class PackageTests(unittest.TestCase):
    def build(self, stage, session=None, pretrain_session=None, also=()):
        package = module("package")
        runner = package.STAGES[stage]["runner"]
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = fake_root(Path(directory.name) / "project", runner)
        out = Path(directory.name) / "out"
        with contextlib.redirect_stdout(io.StringIO()):
            package.prepare(stage, out, "someone", session, pretrain_session, root=root, also=also)
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

    def test_stage_args_reach_the_runner(self):
        out, _ = self.build("lora_1b7")
        metadata = json.loads((out / "kernel-metadata.json").read_text())
        self.assertEqual(metadata["id"], "someone/slm-lora-1b7")
        self.assertEqual(metadata["kernel_sources"], ["someone/slm-sft-data"])
        script, _ = self.payload(out)
        self.assertIn("] + ['--model', '1.7b', '--max-seconds', '14400']", script)

    def test_sft_kernel_attaches_the_chosen_pretrain_session(self):
        # The watcher passes the session it saw finish; dropped on the way, the kernel attached the estimated last one.
        package = module("package")
        chosen = module("stages").last_pretrain_session() + 1
        out, _ = self.build("sft", pretrain_session=chosen)
        metadata = json.loads((out / "kernel-metadata.json").read_text())
        self.assertEqual(metadata["kernel_sources"][0], f"someone/slm-160m-pretrain-{chosen}")
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(package, "prepare") as prepare:
            package.main(["--stage", "sft", "--pretrain-session", str(chosen), "--out", directory])
        self.assertEqual(prepare.call_args.args[4], chosen)

    def test_results_kernel_reads_the_attached_reports_on_cpu(self):
        out, runner = self.build("results", pretrain_session=2, also=("eval", "sft"))
        self.assertEqual(runner, "compute/collect_results.py")
        metadata = json.loads((out / "kernel-metadata.json").read_text())
        self.assertEqual(metadata["id"], "someone/slm-results")
        self.assertFalse(metadata["enable_gpu"])
        self.assertFalse(metadata["enable_internet"])
        self.assertNotIn("machine_shape", metadata)
        self.assertEqual(metadata["kernel_sources"], [f"someone/{name}" for name in (
            "slm-lora-baseline", "slm-lora-eval", "slm-lora-1b7", "slm-lora-1b7-eval", "slm-distill-data",
            "slm-160m-pretrain-1", "slm-160m-pretrain-2", "slm-160m-eval", "slm-160m-sft")])
        script, archive = self.payload(out)
        # run.py runs from /kaggle/working/slm-project, so the default --out and --reports-dir land in the output.
        self.assertIn("] + ['--input-dir', '/kaggle/input']", script)
        self.assertIn(runner, archive.namelist())
        out, _ = self.build("results")
        sources = json.loads((out / "kernel-metadata.json").read_text())["kernel_sources"]
        self.assertNotIn("someone/slm-160m-sft", sources)
        self.assertNotIn("someone/slm-160m-eval", sources)

    def test_also_applies_only_to_results_and_only_to_sft_or_eval(self):
        package = module("package")
        with tempfile.TemporaryDirectory() as directory:
            root = fake_root(Path(directory) / "project", "compute/collect_results.py")
            out = Path(directory) / "out"
            for stage, also in (("lora", ("sft",)), ("results", ("corpus",))):
                with self.subTest(stage=stage), self.assertRaisesRegex(ValueError, "--also"):
                    package.prepare(stage, out, "someone", root=root, also=also)
            self.assertFalse(out.exists())

    def test_cli_passes_the_results_flags(self):
        package = module("package")
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(package, "prepare") as prepare:
            package.main(["--stage", "results", "--pretrain-session", "3", "--also", "eval", "--also", "sft",
                          "--out", directory])
        self.assertEqual(prepare.call_args.args[4], 3)
        self.assertEqual(prepare.call_args.kwargs["also"], ("eval", "sft"))

    def test_cli_rejects_flags_for_the_wrong_stage(self):
        package = module("package")
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "out"
            for argv in (["--stage", "corpus", "--pretrain-session", "2", "--out", str(out)],
                         ["--stage", "eval", "--pretrain-session", "2", "--out", str(out)],
                         ["--stage", "lora", "--also", "sft", "--out", str(out)],
                         ["--stage", "results", "--also", "corpus", "--out", str(out)],
                         ["--stage", "corpus", "--session", "1", "--out", str(out)],
                         ["--stage", "pretrain", "--out", str(out)]):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    package.main(argv)
            self.assertFalse(out.exists())

    def test_output_inside_a_packaged_folder_is_refused(self):
        # A run.py left in compute/ or JSON left in data/ would ship inside the next stage's payload.
        package = module("package")
        with tempfile.TemporaryDirectory() as directory:
            root = fake_root(Path(directory) / "project", "compute/stage3_sft_data.py")
            # macOS disks ignore case, so Compute/ there is compute/, which resolve() does not reveal.
            for out in (root / "compute", root / "compute" / "kernel", root / "data", root / "Compute",
                        root / "DATA" / "kernel"):
                with self.subTest(out=out), self.assertRaisesRegex(ValueError, "outside"):
                    package.prepare("sft_data", out, root=root)
            self.assertFalse((root / "compute" / "run.py").exists())
            self.assertFalse((root / "data" / "kernel-metadata.json").exists())

    def test_missing_runner_is_reported(self):
        package = module("package")
        with tempfile.TemporaryDirectory() as directory:
            root = fake_root(Path(directory) / "project", "compute/other.py")
            with self.assertRaisesRegex(FileNotFoundError, "stage4_sft.py"):
                package.prepare("sft", Path(directory) / "out", root=root)
            self.assertFalse((Path(directory) / "out").exists())


class PackageWorkflowTests(unittest.TestCase):
    def test_workflow_offers_every_stage_and_keeps_inputs_out_of_the_script(self):
        workflow = ROOT / ".github/workflows/package-kernel.yml"
        if not workflow.exists():
            self.skipTest("the package-kernel workflow is not packaged here")
        stages = module("stages")
        text = workflow.read_text()
        options = re.search(r"\n {8}options:\n((?: {10}- \S+\n)+)", text).group(1)
        self.assertEqual(sorted(re.findall(r"- (\S+)", options)), sorted(stages.STAGES))
        # An input pasted into run: is script injection, so each one reaches the shell as an env variable.
        lines = [line for line in text.splitlines() if "${{ inputs." in line]
        self.assertEqual(len(lines), 5)
        for line in lines:
            self.assertRegex(line, r"^ +[A-Z_]+: \$\{\{ inputs\.\w+ \}\}$")
        actions = re.findall(r"uses: (\S+)", text)
        self.assertEqual(len(actions), 3)
        for action in actions:
            self.assertRegex(action, r"@[0-9a-f]{40}$")


if __name__ == "__main__":
    unittest.main()
