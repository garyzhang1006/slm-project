"""Stage corpus: stream English pretraining text into train and held-out JSONL files, on Kaggle only.

Each output line is {"text": str, "source": str}, the format cognition_slm.data.load_pretrain_text
reads. Byte counts are UTF-8 bytes of the text field, which equal token counts under the byte
tokenizer (plus one BOS and one EOS per document).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import unicodedata

# Revisions and licenses checked against https://huggingface.co/api/datasets/<id> on 2026-09-25.
# The fineweb-edu revision is the one scripts/kaggle_long_run.py already pins.
SOURCES = {
    "HuggingFaceFW/fineweb-edu": {"config": "sample-10BT",
                                  "revision": "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
                                  "license": "odc-by"},
    "roneneldan/TinyStories": {"config": None,
                               "revision": "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64",
                               "license": "cdla-sharing-1.0"},
}
# TinyStories teaches plain narrative English; most bytes still come from broad educational web text.
DEFAULT_SHARES = {"HuggingFaceFW/fineweb-edu": 0.85, "roneneldan/TinyStories": 0.15}
DEFAULT_TARGET_BYTES = 1_500_000_000
EVAL_MODULUS = 200  # one document in 200 (0.5%) goes to the held-out split
DEFAULT_MAX_EVAL_BYTES = 16_000_000
MIN_DOCUMENT_BYTES = 200
MAX_DOCUMENT_BYTES = 100_000
MIN_LANGUAGE_SCORE = 0.8
OUT_DIR = Path("/kaggle/working/corpus")
HOLDOUT_PATH = Path("data/simple_questions_holdout.json")

COMMON_WORDS = frozenset(
    "the of and to a in is that it for was on are as with be at by this have from or an "
    "they which you one had not but what all were when we there can he she his her their".split())
_NON_WORD = re.compile(r"[^0-9a-z]+")
_SENTENCE_END = re.compile(r"(?<=[.?!])\s+")


def normalize_overlap(text: str) -> str:
    """Casefold, drop punctuation and collapse whitespace, for holdout overlap checks."""
    return " ".join(_NON_WORD.sub(" ", unicodedata.normalize("NFKC", text).casefold()).split())


def holdout_stems(prompts: list[str], min_words: int = 4) -> list[str]:
    """Distinctive sentences of the holdout prompts, normalized.

    A sentence shared by two or more prompts ("Reply with the number.") is an answer-format
    instruction rather than holdout content, so it is dropped; otherwise every trained row with
    that instruction would count as overlap. Sentences under min_words ("What is it?") are too
    common in ordinary text to mark overlap.
    """
    counts: dict[str, int] = {}
    per_prompt = []
    for prompt in prompts:
        sentences = {normalize_overlap(part) for part in _SENTENCE_END.split(prompt.strip())}
        sentences.add(normalize_overlap(prompt))
        sentences.discard("")
        per_prompt.append(sentences)
        for sentence in sentences:
            counts[sentence] = counts.get(sentence, 0) + 1
    stems = {sentence for sentences in per_prompt for sentence in sentences
             if counts[sentence] == 1 and len(sentence.split()) >= min_words}
    return sorted(stems)


def overlaps_holdout(text: str, stems: list[str]) -> bool:
    padded = f" {normalize_overlap(text)} "
    return any(f" {stem} " in padded for stem in stems)


def load_holdout_stems(path: Path = HOLDOUT_PATH) -> list[str]:
    rows = json.loads(Path(path).read_text())["rows"]
    return holdout_stems([row["prompt"] for row in rows])


def digest(path: Path) -> str:
    # Defined here, not imported from scripts/kaggle_english_run.py, because compute/package.py
    # bundles only scripts/score_holdout.py from scripts/.
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def document_digest(text: str) -> bytes:
    """16-byte key of the whitespace- and case-normalized text; 16 bytes keeps a large seen-set small."""
    return hashlib.sha256(" ".join(text.casefold().split()).encode("utf-8")).digest()[:16]


def is_eval_document(digest: bytes, modulus: int = EVAL_MODULUS) -> bool:
    return int.from_bytes(digest[:8], "big") % modulus == 0


def contains_secret(text: str) -> bool:
    from cognition_slm.audit import SECRET_PATTERNS

    return any(pattern.search(text) for pattern in SECRET_PATTERNS)


def looks_english(text: str, sample_chars: int = 5000) -> bool:
    """Mostly ASCII letters and enough common English function words."""
    sample = text[:sample_chars]
    letters = [char for char in sample if char.isalpha()]
    if len(letters) < 50:
        return False
    if sum(char.isascii() for char in letters) / len(letters) < 0.97:
        return False
    words = [word for word in _NON_WORD.split(sample.casefold()) if word]
    return sum(word in COMMON_WORDS for word in words) / len(words) >= 0.15


def clean_text(text) -> str | None:
    if not isinstance(text, str):
        return None
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    # U+FFFD marks bytes that failed to decode upstream; such documents are usually garbled.
    if "�" in text or any(ord(char) < 32 and char not in "\n\t" for char in text):
        return None
    return text


def check_document(raw: dict, source: str, stems: list[str]) -> tuple[str | None, str]:
    """Return (clean text, "ok") or (None, rejection reason)."""
    if source == "HuggingFaceFW/fineweb-edu":
        if raw.get("language", "en") != "en" or float(raw.get("language_score") or 0) < MIN_LANGUAGE_SCORE:
            return None, "language_label"
    text = clean_text(raw.get("text"))
    if text is None:
        return None, "invalid_text"
    size = len(text.encode("utf-8"))
    if size < MIN_DOCUMENT_BYTES:
        return None, "too_short"
    if size > MAX_DOCUMENT_BYTES:
        return None, "too_long"
    if not looks_english(text):
        return None, "not_english"
    if contains_secret(text):
        return None, "secret_pattern"
    if overlaps_holdout(text, stems):
        return None, "holdout_overlap"
    return text, "ok"


def next_source(train_bytes: dict[str, int], shares: dict[str, float]) -> str:
    """The source furthest below its byte share, so the mixture tracks shares as it grows."""
    total = sum(train_bytes.get(name, 0) for name in shares) + 1
    return max(shares, key=lambda name: (shares[name] - train_bytes.get(name, 0) / total, name))


def resilient_rows(open_stream, retries: int = 5, wait_seconds: float = 30.0):
    """Yield rows from open_stream(skip), reopening past the consumed rows after a network error."""
    consumed, failures = 0, 0
    while True:
        try:
            for row in open_stream(consumed):
                consumed += 1
                yield row
            return
        # Hub streaming surfaces requests, aiohttp and fsspec errors with no common base class.
        except Exception as exc:
            failures += 1
            if failures > retries:
                raise RuntimeError(f"Stream failed {failures} times after {consumed} rows: {exc}") from exc
            print(f"Stream error after {consumed} rows ({exc}); reopening in {wait_seconds}s", flush=True)
            time.sleep(wait_seconds)


class CorpusWriter:
    """Deduplicates, splits by document hash and writes JSONL until the train byte target."""

    def __init__(self, out_dir: Path, target_bytes: int, max_eval_bytes: int,
                 eval_modulus: int = EVAL_MODULUS):
        self.target_bytes, self.max_eval_bytes, self.eval_modulus = target_bytes, max_eval_bytes, eval_modulus
        self.seen: set[bytes] = set()
        self.bytes = {"train": 0, "eval": 0}
        self.documents = {"train": 0, "eval": 0}
        self.per_source: dict[str, dict] = {}
        out_dir.mkdir(parents=True, exist_ok=True)
        self.paths = {"train": out_dir / "pretrain_train.jsonl", "eval": out_dir / "pretrain_eval.jsonl"}
        self.handles = {split: path.open("w", encoding="utf-8") for split, path in self.paths.items()}

    @property
    def done(self) -> bool:
        return self.bytes["train"] >= self.target_bytes

    def stats(self, source: str) -> dict:
        return self.per_source.setdefault(source, {
            "scanned": 0, "train_documents": 0, "eval_documents": 0,
            "train_bytes": 0, "eval_bytes": 0, "rejected": {}})

    def add(self, raw: dict, source: str, stems: list[str]) -> str:
        stats = self.stats(source)
        stats["scanned"] += 1
        text, reason = check_document(raw, source, stems)
        if text is not None:
            digest = document_digest(text)
            if digest in self.seen:
                text, reason = None, "duplicate"
            else:
                self.seen.add(digest)
                split = "eval" if is_eval_document(digest, self.eval_modulus) else "train"
                size = len(text.encode("utf-8"))
                # Surplus held-out documents are dropped, never moved into train.
                if split == "eval" and self.bytes["eval"] + size > self.max_eval_bytes:
                    text, reason = None, "eval_full"
        if text is None:
            stats["rejected"][reason] = stats["rejected"].get(reason, 0) + 1
            return reason
        self.handles[split].write(json.dumps({"text": text, "source": source}, ensure_ascii=False) + "\n")
        self.bytes[split] += size
        self.documents[split] += 1
        stats[f"{split}_documents"] += 1
        stats[f"{split}_bytes"] += size
        return split

    def close(self) -> None:
        for handle in self.handles.values():
            handle.close()


def open_source(source: str):
    from datasets import load_dataset

    spec = SOURCES[source]
    kwargs = {"name": spec["config"]} if spec["config"] else {}

    def open_stream(skip: int):
        stream = load_dataset(source, revision=spec["revision"], split="train", streaming=True, **kwargs)
        return stream.skip(skip) if skip else stream

    return resilient_rows(open_stream)


def build_corpus(out_dir: Path, target_bytes: int, max_eval_bytes: int, shares: dict[str, float],
                 stems: list[str], log_every: int = 20_000) -> dict:
    writer = CorpusWriter(out_dir, target_bytes, max_eval_bytes)
    streams = {source: open_source(source) for source in shares}
    active = dict(shares)
    started = time.monotonic()
    scanned = 0
    try:
        while active and not writer.done:
            source = next_source({name: writer.stats(name)["train_bytes"] for name in active}, active)
            raw = next(streams[source], None)
            if raw is None:
                print(f"{source} exhausted; continuing with the remaining sources", flush=True)
                active.pop(source)
                continue
            writer.add(raw, source, stems)
            scanned += 1
            if scanned % log_every == 0:
                print(f"scanned={scanned} train_bytes={writer.bytes['train']} "
                      f"eval_bytes={writer.bytes['eval']} elapsed={time.monotonic() - started:.0f}s",
                      flush=True)
    finally:
        writer.close()
    if writer.bytes["train"] < target_bytes:
        raise RuntimeError(f"Sources ran out at {writer.bytes['train']} of {target_bytes} train bytes")
    return {"target_bytes": target_bytes, "shares": shares, "documents": writer.documents,
            "bytes": writer.bytes, "per_source": writer.per_source,
            "elapsed_seconds": round(time.monotonic() - started, 1), "paths": writer.paths}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-bytes", type=int, default=DEFAULT_TARGET_BYTES,
                        help="UTF-8 text bytes to write to pretrain_train.jsonl")
    parser.add_argument("--max-eval-bytes", type=int, default=DEFAULT_MAX_EVAL_BYTES)
    parser.add_argument("--tinystories-share", type=float,
                        default=DEFAULT_SHARES["roneneldan/TinyStories"])
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)
    if args.target_bytes < 1 or args.max_eval_bytes < 1:
        parser.error("--target-bytes and --max-eval-bytes must be positive")
    if not 0.0 <= args.tinystories_share < 1.0:
        parser.error("--tinystories-share must be in [0, 1)")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    if not Path("/kaggle/working").is_dir():
        raise RuntimeError("The corpus stage streams gigabytes of data; run it on Kaggle, never locally")
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    sys.path[:0] = [str(root), str(root / "src")]
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    manifest_path = root / "source-manifest.json"
    if manifest_path.exists():
        for name, expected in json.loads(manifest_path.read_text()).items():
            if digest(root / name) != expected:
                raise RuntimeError(f"Source hash mismatch: {name}")
    shares = {"HuggingFaceFW/fineweb-edu": 1.0 - args.tinystories_share,
              "roneneldan/TinyStories": args.tinystories_share}
    shares = {name: share for name, share in shares.items() if share > 0}
    stems = load_holdout_stems(root / HOLDOUT_PATH)
    result = build_corpus(args.out_dir, args.target_bytes, args.max_eval_bytes, shares, stems)
    paths = result.pop("paths")
    report = dict(result, files={split: {"path": str(path), "sha256": digest(path)}
                                 for split, path in paths.items()},
                  sources={name: dict(SOURCES[name], id=name) for name in shares},
                  filters={"min_document_bytes": MIN_DOCUMENT_BYTES, "max_document_bytes": MAX_DOCUMENT_BYTES,
                           "fineweb_min_language_score": MIN_LANGUAGE_SCORE,
                           "english_heuristic": "97% ASCII letters and 15% common English words",
                           "dedupe": "sha256 of casefolded, whitespace-collapsed text",
                           "secrets": "cognition_slm.audit.SECRET_PATTERNS",
                           "holdout_overlap": "documents containing a simple_questions_holdout sentence"},
                  eval_rule=f"sha256 document key modulo {EVAL_MODULUS} == 0, capped at max_eval_bytes",
                  byte_unit="UTF-8 bytes of the text field; the byte tokenizer adds BOS and EOS per document")
    write_json(args.out_dir / "corpus_manifest.json", report)
    print(json.dumps({"documents": report["documents"], "bytes": report["bytes"]}), flush=True)
    print("CORPUS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
