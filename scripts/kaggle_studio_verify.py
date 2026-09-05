"""Verify the default Studio server with existing 500M weights on Kaggle."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import Request, urlopen


def main() -> None:
    if not Path("/kaggle/working").is_dir():
        raise RuntimeError("This verification requires Kaggle; no local model execution is permitted")
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    os.environ["PYTHONPATH"] = str(root / "src")
    os.environ["OMP_NUM_THREADS"] = "2"
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Enable a Kaggle GPU before verification")
    manifest = json.loads((root / "source-manifest.json").read_text())
    for name, expected in manifest.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Source hash mismatch: {name}")
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], check=True)
    matches = list(Path("/kaggle/input").rglob("slm-500m-language-quality.pt"))
    if len(matches) != 1:
        raise RuntimeError(f"Attach one completed 500M quality kernel; found {len(matches)} checkpoints")
    checkpoint = matches[0]
    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    target = root / "artifacts/slm-500m-language-quality.pt"
    target.parent.mkdir(exist_ok=True)
    target.symlink_to(checkpoint)
    base = "http://127.0.0.1:8766"
    with (root / "studio.log").open("w") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "cognition_slm.server", "--device", "cuda"],
            stdout=log, stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 240
            while True:
                if process.poll() is not None:
                    raise RuntimeError("Studio exited during startup; see studio.log")
                try:
                    with urlopen(base + "/api/status", timeout=5) as response:
                        status = json.load(response)
                except URLError:
                    status = {"state": "loading"}
                if status["state"] == "error":
                    raise RuntimeError(status["error"])
                if status["state"] == "ready":
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError("Studio startup exceeded 240 seconds")
                time.sleep(1)
            assert status["model"]["parameters"] == 499_524_075, status
            assert status["model"]["context_window"] == 2048, status
            with urlopen(base, timeout=5) as response:
                assert b"<html" in response.read().lower()
            request = Request(base + "/api/generate", data=json.dumps({
                "prompt": "hi", "task_type": "language_generation",
                "temperature": 0, "top_k": 0, "max_new_tokens": 64,
            }).encode(), headers={"Content-Type": "application/json"})
            with urlopen(request, timeout=180) as response:
                generation = json.load(response)
            assert generation["text"].strip(), generation
            report = {"status": status, "generation": generation,
                      "checkpoint_sha256": digest.hexdigest(), "tests": "passed",
                      "source_manifest": manifest, "gpu": torch.cuda.get_device_name(0),
                      "scope": "Default server startup and HTTP inference; not a capability benchmark"}
            (root / "studio_verification_500m.json").write_text(json.dumps(report, indent=2) + "\n")
            print("STUDIO_VERIFICATION_COMPLETE", json.dumps(report), flush=True)
        finally:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            target.unlink()


if __name__ == "__main__":
    main()
