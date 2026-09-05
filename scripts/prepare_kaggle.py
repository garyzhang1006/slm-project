"""Package reviewed source into a private, self-contained Kaggle GPU script."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def prepare(output: Path, owner: str, slug: str, runner: str) -> None:
    files = [ROOT / name for name in ("pyproject.toml", "README.md", "LICENSE")]
    for folder, pattern in (("src/cognition_slm", "*.py"), ("tests", "*.py"), ("data", "*.json*")):
        files.extend(sorted((ROOT / folder).glob(pattern)))
    for filename in ("index.html", "style.css", "app.js"):
        files.append(ROOT / "src/cognition_slm/web" / filename)
    files.append(ROOT / "scripts" / runner)
    # The regression suite imports the curriculum generator for every runner.
    files.append(ROOT / "scripts" / "build_curriculum_data.py")
    manifest = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT))
        archive.writestr("source-manifest.json", json.dumps(manifest, indent=2))
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    output.mkdir(parents=True, exist_ok=True)
    # Only explicitly selected project files enter the payload, never local credentials or weights.
    script = (
        "import base64, io, os, runpy, zipfile\n"
        "from pathlib import Path\n"
        "root = Path('/kaggle/working/slm-project')\n"
        "root.mkdir(parents=True, exist_ok=True)\n"
        f"payload = {payload!r}\n"
        "with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))) as archive:\n"
        "    archive.extractall(root)\n"
        "os.chdir(root)\n"
        f"runpy.run_path(str(root / 'scripts' / {runner!r}), run_name='__main__')\n"
    )
    (output / "run.py").write_text(script)
    (output / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{owner}/{slug}", "title": slug,
        "code_file": "run.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_internet": False,
        "dataset_sources": [], "competition_sources": [], "kernel_sources": [],
    }, indent=2) + "\n")
    print(json.dumps({"output": str(output), "source_files": len(files), "kernel": f"{owner}/{slug}"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--slug", default="slm-2048-verification")
    parser.add_argument(
        "--runner",
        choices=("kaggle_run.py", "kaggle_quality_run.py", "kaggle_500m_quality_run.py", "kaggle_studio_verify.py"),
        default="kaggle_run.py",
        help="Kaggle entrypoint; quality runner trains a Studio checkpoint.",
    )
    args = parser.parse_args()
    prepare(args.out, args.owner, args.slug, args.runner)
