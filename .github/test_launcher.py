"""Exercise launcher routing without importing PyTorch or starting a server."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


class LauncherTests(unittest.TestCase):
    def run_launcher(self, arguments, imports_ok, cwd=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("launch-studio.command", "pyproject.toml"):
                shutil.copy(Path(__file__).resolve().parents[1] / name, root)
            python = root / ".venv-ui-py313/bin/python"
            python.parent.mkdir(parents=True)
            # imports_ok answers only the dependency check; the PyTorch import check after an install passes.
            python.write_text(
                f"#!{sys.executable}\n"
                "import json, sys\n"
                "with open('calls.jsonl', 'a') as log:\n"
                "    log.write(json.dumps(sys.argv[1:]) + '\\n')\n"
                f"sys.exit(0 if {imports_ok!r} or sys.argv[1] == '-m' or 'find_spec' not in sys.argv[2] else 1)\n"
            )
            python.chmod(0o755)
            uv = root / "bin/uv"
            uv.parent.mkdir()
            uv.write_text(
                f"#!{sys.executable}\n"
                "import json, sys\n"
                "with open('calls.jsonl', 'a') as log:\n"
                "    log.write(json.dumps(['uv', *sys.argv[1:]]) + '\\n')\n"
            )
            uv.chmod(0o755)
            result = subprocess.run(
                ["/bin/bash", str(root / "launch-studio.command"), *arguments],
                capture_output=True, text=True, timeout=10, cwd=cwd,
                env={**os.environ, "PATH": f"{uv.parent}{os.pathsep}{os.environ.get('PATH', '')}"},
            )
            calls = [json.loads(line) for line in (root / "calls.jsonl").read_text().splitlines()]
            self.assertEqual(result.returncode, 0, result.stderr)
            return calls

    def test_sources_only_bypasses_broken_model_dependencies(self):
        arguments = ["--port", "8767", "--sources-only"]
        self.assertEqual(self.run_launcher(arguments, False),
                         [["-m", "cognition_slm.server", "--open", *arguments]])

    def test_model_mode_keeps_dependency_checks_and_arguments(self):
        arguments = ["--checkpoint", "/tmp/model with spaces.pt"]
        calls = self.run_launcher(arguments, True)
        self.assertEqual([call[0] for call in calls], ["-c", "-c", "-m"])
        self.assertEqual(calls[-1], ["-m", "cognition_slm.server", "--open", *arguments])

    def test_lora_adapter_installs_the_pinned_lora_extra(self):
        # Pinned to the transformers, peft and accelerate that trained and scored the published adapters.
        lora = ["transformers==5.0.0", "peft==0.19.1", "accelerate==1.13.0"]
        for arguments, needed, package in ((["--lora-adapter", "/tmp/adapter"], lora, ".[lora]"),
                                           (["--lora-adapter=/tmp/adapter"], lora, ".[lora]"),
                                           (["--checkpoint", "/tmp/model.pt"], [], ".")):
            with self.subTest(arguments=arguments):
                calls = self.run_launcher(arguments, False)
                self.assertEqual(calls[0][0], "-c")
                self.assertEqual(calls[0][2:], ["torch", "numpy", *needed])
                self.assertEqual(calls[1], ["uv", "pip", "install", "--python", ".venv-ui-py313/bin/python",
                                            "-e", package])
                self.assertEqual(calls[-1], ["-m", "cognition_slm.server", "--open", *arguments])
        # Installed pins skip the install, and a search-only launch never checks for them.
        self.assertEqual([call[0] for call in self.run_launcher(["--lora-adapter", "/tmp/adapter"], True)],
                         ["-c", "-c", "-m"])
        self.assertEqual(self.run_launcher(["--sources-only", "--lora-adapter", "/tmp/adapter"], False),
                         [["-m", "cognition_slm.server", "--open", "--sources-only", "--lora-adapter", "/tmp/adapter"]])

    def test_relative_model_paths_resolve_from_the_callers_folder(self):
        # The launcher moves into the project folder, so a path typed relative to where it ran is rewritten.
        arguments = ["--sources-only", "--lora-adapter", "adapter", "--checkpoint=model.pt", "--port", "8767"]
        with tempfile.TemporaryDirectory() as caller:
            calls = self.run_launcher(arguments, False, cwd=caller)
            here = str(Path(caller).resolve())
        self.assertEqual(calls, [["-m", "cognition_slm.server", "--open", "--sources-only", "--lora-adapter",
                                  f"{here}/adapter", f"--checkpoint={here}/model.pt", "--port", "8767"]])
        # No arguments at all must still start the server under set -u.
        self.assertEqual(self.run_launcher([], True)[-1], ["-m", "cognition_slm.server", "--open"])


if __name__ == "__main__":
    unittest.main()
