"""Package one compute stage into a private, self-contained Kaggle kernel directory."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compute.stages import OWNER, STAGES, stage_attaches, stage_slug  # noqa: E402


def source_files(root: Path, runner: str) -> list[Path]:
    """Explicit project files for the payload; local weights, credentials and caches never match."""
    files = [root / "pyproject.toml"]
    files.extend(sorted((root / "src/cognition_slm").glob("*.py")))
    for filename in ("index.html", "style.css", "app.js"):
        files.append(root / "src/cognition_slm/web" / filename)
    files.append(root / "scripts/score_holdout.py")
    files.extend(sorted((root / "data").glob("*.json*")))
    files.extend(sorted((root / "compute").glob("*.py")))
    missing = [str(path.relative_to(root)) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot package, missing files: {', '.join(missing)}")
    if root / runner not in files:
        raise FileNotFoundError(f"Stage runner {runner} is not in compute/; write it before packaging")
    return files


def run_script(payload: str, runner: str, argv: list[str]) -> str:
    return (
        "import base64, io, os, runpy, sys, zipfile\n"
        "from pathlib import Path\n"
        "root = Path('/kaggle/working/slm-project')\n"
        "root.mkdir(parents=True, exist_ok=True)\n"
        f"payload = {payload!r}\n"
        "with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))) as archive:\n"
        "    archive.extractall(root)\n"
        "os.chdir(root)\n"
        "sys.path[:0] = [str(root), str(root / 'src')]\n"
        "os.environ['PYTHONPATH'] = str(root / 'src')\n"
        f"sys.argv = [str(root / {runner!r})] + {argv!r}\n"
        f"runpy.run_path(str(root / {runner!r}), run_name='__main__')\n"
    )


def kernel_metadata(stage: str, owner: str, session: int | None, pretrain_session: int | None) -> dict:
    spec = STAGES[stage]
    slug = stage_slug(stage, session)
    metadata = {
        "id": f"{owner}/{slug}", "title": slug,
        "code_file": "run.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": spec["gpu"],
        "enable_internet": spec["internet"],
        "dataset_sources": [], "competition_sources": [],
        "kernel_sources": [f"{owner}/{name}" for name in stage_attaches(stage, session, pretrain_session)],
    }
    if spec["gpu"]:
        metadata["machine_shape"] = "NvidiaTeslaT4"
    return metadata


def prepare(stage: str, output: Path, owner: str = OWNER, session: int | None = None,
            pretrain_session: int | None = None, root: Path = ROOT) -> dict:
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; choose one of {', '.join(STAGES)}")
    # Validates the session before any file work, so a bad flag writes nothing.
    metadata = kernel_metadata(stage, owner, session, pretrain_session)
    runner = STAGES[stage]["runner"]
    # source_files globs these folders, so a run.py or JSON written into one would ship in the next payload.
    for folder in ("compute", "data", "src/cognition_slm"):
        # macOS disks ignore case, so Compute/ there is compute/, and resolve() keeps the case as typed.
        if Path(str(output.resolve()).casefold()).is_relative_to(str((root / folder).resolve()).casefold()):
            raise ValueError(f"--out {output} is inside {folder}/, which is packaged; "
                             "choose a folder outside it, such as /tmp/slm-kernel")
    files = source_files(root, runner)
    manifest = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(root))
        archive.writestr("source-manifest.json", json.dumps(manifest, indent=2))
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    argv = list(STAGES[stage].get("args", [])) + ([] if session is None else ["--session", str(session)])
    output.mkdir(parents=True, exist_ok=True)
    (output / "run.py").write_text(run_script(payload, runner, argv))
    (output / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    summary = {"output": str(output), "source_files": len(files), "kernel": metadata["id"],
               "kernel_sources": metadata["kernel_sources"], "gpu": metadata["enable_gpu"]}
    print(json.dumps(summary))
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=tuple(STAGES), required=True)
    parser.add_argument("--session", type=int, help="Pretrain session number k (pretrain only)")
    parser.add_argument("--pretrain-session", type=int,
                        help="sft only: which pretrain session kernel to attach (default: estimated last session)")
    parser.add_argument("--owner", default=OWNER)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.pretrain_session is not None and args.stage != "sft":
        parser.error("--pretrain-session only applies to --stage sft")
    try:
        prepare(args.stage, args.out, args.owner, args.session, args.pretrain_session)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
