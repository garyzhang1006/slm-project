import ast
import base64
import contextlib
import importlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
# Kaggle packages only the runners a kernel needs, so each test skips when its runner is absent.
sys.path.insert(0, str(ROOT / "scripts"))


def runner(name):
    if not (ROOT / "scripts" / f"{name}.py").exists():
        raise unittest.SkipTest(f"{name}.py is not packaged here")
    return importlib.import_module(f"scripts.{name}")


def squad_prompt(context, question):
    return f"Answer from the passage.\n\nPassage: {context}\n\nQuestion: {question}"


class PrepareKaggleTests(unittest.TestCase):
    def test_bundle_holds_every_module_the_regression_suite_imports(self):
        # Every runner starts with `unittest discover -s tests`, and several tests import compute/ and the
        # elementary runner at module level, so a bundle without them fails before any training.
        prepare = runner("prepare_kaggle")
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            prepare.prepare(Path(directory), "someone", "slug", "kaggle_run.py")
            manifest = set(json.loads((Path(directory) / "source-manifest.json").read_text()))
        needed = {str(path.relative_to(ROOT)) for path in (ROOT / "compute").glob("*.py")}
        self.assertLessEqual(needed | {"scripts/kaggle_elementary_run.py"}, manifest)

    def test_bundled_results_tests_pass_without_the_files_the_bundle_leaves_out(self):
        # dcaaa65 added a test that reads compute/RESULTS.md, which no bundle ships, so the regression suite every
        # runner starts with failed before any training.
        prepare = runner("prepare_kaggle")
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            prepare.prepare(Path(directory) / "kernel", "someone", "slug", "kaggle_run.py")
            script = (Path(directory) / "kernel" / "run.py").read_text()
            payload = ast.literal_eval(next(line for line in script.splitlines()
                                            if line.startswith("payload = "))[len("payload = "):])
            bundle = Path(directory) / "bundle"
            with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))) as archive:
                archive.extractall(bundle)
            # The runners' own command, narrowed to the one module.
            result = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests",
                                     "-p", "test_compute_results.py"], cwd=bundle, capture_output=True, text=True,
                                    timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])

    def test_output_inside_a_packaged_folder_is_refused(self):
        # A run.py written into compute/ or tests/ would ship, stale, in every later bundle.
        prepare = runner("prepare_kaggle")
        for folder in ("compute", "tests", "data", "src/cognition_slm"):
            output = ROOT / folder / "kernel-bundle"
            with self.subTest(folder=folder), self.assertRaisesRegex(ValueError, "packaged"):
                prepare.prepare(output, "someone", "slug", "kaggle_run.py")
            self.assertFalse(output.exists())

    def test_parent_kernels_belong_to_the_owner(self):
        # README.md tells users to pass their own --owner, so chained runners must attach that account's kernels.
        prepare = runner("prepare_kaggle")
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            prepare.prepare(Path(directory), "someone", "slug", "kaggle_qa_run.py")
            metadata = json.loads((Path(directory) / "kernel-metadata.json").read_text())
        self.assertEqual(metadata["kernel_sources"], ["someone/slm-500m-english-corpus"])

    def test_studio_verify_attaches_the_language_quality_checkpoint(self):
        # kaggle_studio_verify.py looks for slm-500m-language-quality.pt, which the english-code-quality-v2
        # kernel wrote (reports/slm-500m-language-quality-verification.json).
        prepare = runner("prepare_kaggle")
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            prepare.prepare(Path(directory), "someone", "slug", "kaggle_studio_verify.py")
            metadata = json.loads((Path(directory) / "kernel-metadata.json").read_text())
        self.assertEqual(metadata["kernel_sources"], ["someone/slm-500m-english-code-quality-v2"])

    def test_launcher_bundles_into_a_private_folder(self):
        # A fixed folder in shared /tmp may already belong to another local user, who could swap the bundle.
        launcher = ROOT / "scripts" / "launch_500m_kaggle.sh"
        if not launcher.exists():
            self.skipTest("launch_500m_kaggle.sh is not packaged here")
        source = launcher.read_text()
        self.assertIn('OUTPUT_DIR="${3:-$(mktemp -d', source)
        self.assertNotIn("/tmp/slm-kaggle-500m-quality}", source)


class PrepareDataTests(unittest.TestCase):
    def test_flags_fill_only_missing_provenance(self):
        prepare_data = runner("prepare_data")
        base = {"prompt": "Name a color.", "answer": "Blue.", "task_type": "language_generation",
                "confidence": 0.9, "error_category": "none"}
        records = [{"id": "bare", **base}, {"id": "blank", **base, "source": " ", "license": ""},
                   {"id": "own", **base, "source": "wiki-dump", "license": "CC-BY-SA-4.0"}]
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "in.jsonl", Path(directory) / "out.jsonl"
            source.write_text("".join(json.dumps(record) + "\n" for record in records))
            with contextlib.redirect_stdout(io.StringIO()):
                prepare_data.main(["--input", str(source), "--output", str(output),
                                   "--source", "project-synthetic", "--license", "CC0-1.0"])
            rows = [json.loads(line) for line in output.read_text().split("\n") if line]
        written = {row["id"]: (row["source"], row["license"]) for row in rows}
        self.assertEqual(written, {"bare": ("project-synthetic", "CC0-1.0"),
                                   "blank": ("project-synthetic", "CC0-1.0"),
                                   "own": ("wiki-dump", "CC-BY-SA-4.0")})


    def test_bad_input_is_a_usage_error(self):
        prepare_data = runner("prepare_data")
        with tempfile.TemporaryDirectory() as directory:
            broken = Path(directory) / "broken.jsonl"
            broken.write_text("{not json\n")
            for path, message in ((Path(directory) / "missing.jsonl", "no such input file"), (broken, "broken.jsonl")):
                with self.subTest(path=path.name), contextlib.redirect_stderr(io.StringIO()) as stderr:
                    with self.assertRaises(SystemExit) as raised:
                        prepare_data.main(["--input", str(path), "--output", str(Path(directory) / "out.jsonl"),
                                           "--source", "project-synthetic", "--license", "CC0-1.0"])
                    self.assertEqual(raised.exception.code, 2)
                    self.assertIn(message, stderr.getvalue())


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

    def test_training_report_survives_a_warning_after_it(self):
        quality = runner("kaggle_500m_quality_run")
        log = '{"step": 1}\n{\n  "steps": 1200\n}\nException ignored in: <function _remove at 0x1>\n'
        self.assertEqual(quality._final_training_report(log), {"steps": 1200})

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

    def test_50m_runner_keeps_training_log(self):
        # A captured subprocess hides the traceback when training fails, so the 50M runner streams to a log
        # through the 500M helpers, and its bundle must ship them.
        prepare = runner("prepare_kaggle")
        source = (ROOT / "scripts" / "kaggle_quality_run.py").read_text()
        self.assertIn("from kaggle_500m_quality_run import _final_training_report, _run_logged", source)
        self.assertNotIn("capture_output", source)
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            prepare.prepare(Path(directory), "someone", "slug", "kaggle_quality_run.py")
            manifest = set(json.loads((Path(directory) / "source-manifest.json").read_text()))
        self.assertIn("scripts/kaggle_500m_quality_run.py", manifest)


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
