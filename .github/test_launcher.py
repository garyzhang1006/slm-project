"""Exercise launcher routing without importing PyTorch or starting a server."""

import json
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
            shutil.copy(Path(__file__).resolve().parents[1] / "launch-studio.command", root)
            python = root / ".venv-ui-py313/bin/python"
            python.parent.mkdir(parents=True)
            python.write_text(
                f"#!{sys.executable}\n"
                "import json, sys\n"
                "with open('calls.jsonl', 'a') as log:\n"
                "    log.write(json.dumps(sys.argv[1:]) + '\\n')\n"
                f"sys.exit(0 if {imports_ok!r} or sys.argv[1] == '-m' else 1)\n"
            )
            python.chmod(0o755)
            result = subprocess.run(
                ["/bin/bash", str(root / "launch-studio.command"), *arguments],
                capture_output=True, text=True, timeout=10, cwd=cwd,
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
