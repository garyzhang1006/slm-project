"""Stage sft_data: build short-answer instruction data on Kaggle CPU.

Sources: databricks-dolly-15k rows with short responses, first-turn English pairs from
OpenAssistant/oasst1, and the project-authored rows in compute/short_facts.py. Every row whose
prompt or answer overlaps data/simple_questions_holdout.json is dropped, so the holdout stays a
fair test. Writes /kaggle/working/sft/sft_train.jsonl, sft_eval.jsonl and sft_manifest.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    # Run directly as a script, the repository root is not on sys.path yet.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compute import short_facts  # noqa: E402
from compute.stage1_corpus import (HOLDOUT_PATH, contains_secret, digest, holdout_stems,  # noqa: E402
                                   normalize_overlap, overlaps_holdout, write_json)

# Revisions and licenses checked against https://huggingface.co/api/datasets/<id> on 2026-09-25.
# The dolly revision matches scripts/kaggle_english_run.py SOURCES.
SOURCES = {
    "databricks/databricks-dolly-15k": {"revision": "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a",
                                        "license": "CC-BY-SA-3.0"},
    "OpenAssistant/oasst1": {"revision": "fdf72ae0827c1cda404aff25b6603abec9e3399b",
                             "license": "Apache-2.0"},
    short_facts.SOURCE: {"revision": "this repository", "license": short_facts.LICENSE},
}
MAX_RECORD_BYTES = 1024  # serialized prompt template plus answer, well inside a 2048-byte block
MAX_DOLLY_RESPONSE_CHARS = 400
MAX_OASST_PROMPT_CHARS = 400
MAX_OASST_ANSWER_CHARS = 600
MAX_OASST_TOXICITY = 0.2
EVAL_MODULUS = 50  # one group in 50 (2%) is held out
MAX_EVAL_ROWS = 1024
OUT_DIR = Path("/kaggle/working/sft")


def sft_record(record_id: str, prompt: str, answer: str, source: str, license_name: str,
               max_bytes: int = MAX_RECORD_BYTES) -> dict | None:
    """A record valid under cognition_slm.data.validate_record that fits max_bytes, else None."""
    from cognition_slm.data import DataValidationError, format_training_text, validate_record

    raw = {"id": record_id, "prompt": prompt.strip(), "answer": answer.strip(),
           "task_type": "language_generation", "confidence": 0.9, "error_category": "none",
           "source": source, "license": license_name}
    try:
        example = validate_record(raw)
    except DataValidationError:
        return None
    # +2 for the BOS and EOS the byte tokenizer adds.
    if len(format_training_text(example).encode("utf-8")) + 2 > max_bytes:
        return None
    return example.to_dict()


def dolly_rows(raws) -> list[tuple[str, dict]]:
    """(group, record) for Dolly rows whose response is short."""
    rows = []
    for index, raw in enumerate(raws):
        instruction, context, response = (raw.get(key) or "" for key in ("instruction", "context", "response"))
        if not all(isinstance(value, str) for value in (instruction, context, response)):
            continue
        if not instruction.strip() or not response.strip() or len(response) > MAX_DOLLY_RESPONSE_CHARS:
            continue
        prompt = instruction.strip() + (f"\n\nContext:\n{context.strip()}" if context.strip() else "")
        record = sft_record(f"dolly-{index}", prompt, response, "databricks/databricks-dolly-15k",
                            SOURCES["databricks/databricks-dolly-15k"]["license"])
        if record:
            rows.append((f"dolly:{index}", record))
    return rows


def oasst_pairs(raws) -> list[tuple[str, dict]]:
    """(group, record) for English first-turn prompts and their best-ranked English reply."""
    messages = [raw for raw in raws if not raw.get("deleted")]
    roots = {raw["message_id"]: raw for raw in messages
             if raw.get("role") == "prompter" and raw.get("parent_id") is None
             and raw.get("lang") == "en" and raw.get("review_result") is not False}
    best: dict[str, dict] = {}
    for raw in messages:
        parent = raw.get("parent_id")
        if raw.get("role") != "assistant" or parent not in roots or raw.get("lang") != "en":
            continue
        toxicity = (raw.get("detoxify") or {}).get("toxicity")
        if toxicity is not None and toxicity > MAX_OASST_TOXICITY:
            continue
        rank = raw.get("rank")
        # Unranked replies sort after every ranked one.
        key = (rank is None, rank if rank is not None else 0, raw["message_id"])
        if parent not in best or key < best[parent][0]:
            best[parent] = (key, raw)
    rows = []
    for parent, (_, reply) in sorted(best.items()):
        prompt, answer = roots[parent]["text"], reply["text"]
        if len(prompt) > MAX_OASST_PROMPT_CHARS or len(answer) > MAX_OASST_ANSWER_CHARS:
            continue
        record = sft_record(f"oasst-{reply['message_id']}", prompt, answer, "OpenAssistant/oasst1",
                            SOURCES["OpenAssistant/oasst1"]["license"])
        if record:
            rows.append((f"oasst:{parent}", record))
    return rows


def short_fact_records() -> list[tuple[str, dict]]:
    rows = []
    for row in short_facts.short_fact_rows():
        key = hashlib.sha256(row["prompt"].encode("utf-8")).hexdigest()[:16]
        record = sft_record(f"short-{key}", row["prompt"], row["answer"], short_facts.SOURCE,
                            short_facts.LICENSE)
        if record:
            rows.append((f"short:{row['group']}", record))
    return rows


def holdout_conflict(record: dict, stems: list[str], holdout_prompts: list[str]) -> bool:
    """True when the prompt or answer contains a holdout sentence, or the prompt sits inside one."""
    if overlaps_holdout(record["prompt"], stems) or overlaps_holdout(record["answer"], stems):
        return True
    prompt = normalize_overlap(record["prompt"])
    return len(prompt.split()) >= 3 and any(f" {prompt} " in f" {question} " for question in holdout_prompts)


def filter_rows(rows: list[tuple[str, dict]], stems: list[str], holdout_prompts: list[str],
                dropped: dict[str, int]) -> list[tuple[str, dict]]:
    """Drop holdout overlap, secrets and repeated prompts, counting each reason in dropped."""
    from cognition_slm.audit import _prompt_key

    kept, seen = [], set()
    normalized_holdout = [normalize_overlap(prompt) for prompt in holdout_prompts]
    for group, record in rows:
        if holdout_conflict(record, stems, normalized_holdout):
            reason = "holdout_overlap"
        elif contains_secret(record["prompt"]) or contains_secret(record["answer"]):
            reason = "secret_pattern"
        elif _prompt_key(record["prompt"]) in seen:
            # The audit keys prompts this way; one row per key keeps train and eval disjoint.
            reason = "duplicate_prompt"
        else:
            seen.add(_prompt_key(record["prompt"]))
            kept.append((group, record))
            continue
        dropped[reason] = dropped.get(reason, 0) + 1
    return kept


def _hash(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def split_rows(rows: list[tuple[str, dict]], eval_modulus: int = EVAL_MODULUS,
               max_eval: int = MAX_EVAL_ROWS) -> tuple[list[dict], list[dict]]:
    """Hold out whole groups by hash, so no phrasing of an eval fact is trained on.

    A held-out group that would push eval past max_eval rows goes to train whole. Train order
    is a fixed hash order, which mixes sources.
    """
    groups: dict[str, list[dict]] = {}
    for group, record in rows:
        groups.setdefault(group, []).append(record)
    train, evaluation = [], []
    for group in sorted(groups, key=lambda name: (_hash(name), name)):
        members = groups[group]
        if _hash(group) % eval_modulus == 0 and len(evaluation) + len(members) <= max_eval:
            evaluation.extend(members)
        else:
            train.extend(members)
    train.sort(key=lambda record: _hash(record["id"]))
    return train, evaluation


def build_sft(dolly_raws, oasst_raws, holdout_prompts: list[str]) -> tuple[list[dict], list[dict], dict]:
    """Pure assembly used by main and tests: records in, (train, eval, stats) out."""
    stems = holdout_stems(holdout_prompts)
    by_source = {"databricks/databricks-dolly-15k": dolly_rows(dolly_raws),
                 short_facts.SOURCE: short_fact_records()}
    if oasst_raws is not None:
        by_source["OpenAssistant/oasst1"] = oasst_pairs(oasst_raws)
    stats = {"candidates": {name: len(rows) for name, rows in by_source.items()}, "dropped": {}}
    # Project rows go first, so a prompt duplicated in a public source keeps the short answer.
    ordered = by_source[short_facts.SOURCE] + [row for name, rows in by_source.items()
                                                if name != short_facts.SOURCE for row in rows]
    kept = filter_rows(ordered, stems, holdout_prompts, stats["dropped"])
    train, evaluation = split_rows(kept)
    stats["train"], stats["eval"] = {}, {}
    for split, records in (("train", train), ("eval", evaluation)):
        for record in records:
            stats[split][record["source"]] = stats[split].get(record["source"], 0) + 1
    return train, evaluation, stats


def _load_rows(source: str, fields: tuple[str, ...]) -> list[dict]:
    from datasets import load_dataset

    stream = load_dataset(source, revision=SOURCES[source]["revision"], split="train", streaming=True)
    return [{field: raw.get(field) for field in fields} for raw in stream]


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--no-oasst", action="store_true", help="Skip OpenAssistant/oasst1")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if not Path("/kaggle/working").is_dir():
        raise RuntimeError("The SFT data stage downloads datasets; run it on Kaggle, never locally")
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    sys.path[:0] = [str(root), str(root / "src")]
    os.environ.update(PYTHONPATH=str(root / "src"), PYTHONUNBUFFERED="1")

    manifest_path = root / "source-manifest.json"
    if manifest_path.exists():
        for name, expected in json.loads(manifest_path.read_text()).items():
            if digest(root / name) != expected:
                raise RuntimeError(f"Source hash mismatch: {name}")
    holdout_prompts = [row["prompt"] for row in json.loads((root / HOLDOUT_PATH).read_text())["rows"]]
    dolly = _load_rows("databricks/databricks-dolly-15k", ("instruction", "context", "response"))
    oasst = None if args.no_oasst else _load_rows(
        "OpenAssistant/oasst1", ("message_id", "parent_id", "text", "role", "lang", "review_result",
                                 "deleted", "rank", "detoxify"))
    train, evaluation, stats = build_sft(dolly, oasst, holdout_prompts)
    if len(train) < 5_000 or len(evaluation) < 64:
        raise RuntimeError(f"Only {len(train)} train and {len(evaluation)} eval rows; expected at least 5000 and 64")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"train": args.out_dir / "sft_train.jsonl", "eval": args.out_dir / "sft_eval.jsonl"}
    write_jsonl(train, paths["train"])
    write_jsonl(evaluation, paths["eval"])
    subprocess.run([sys.executable, "-m", "cognition_slm.audit", "--train", str(paths["train"]),
                    "--eval", str(paths["eval"])], check=True)
    used = [name for name in SOURCES if name != "OpenAssistant/oasst1" or oasst is not None]
    write_json(args.out_dir / "sft_manifest.json", dict(
        stats, files={split: {"path": str(path), "rows": len(rows), "sha256": digest(path)}
                      for (split, path), rows in zip(paths.items(), (train, evaluation))},
        sources={name: SOURCES[name] for name in used},
        filters={"max_record_bytes": MAX_RECORD_BYTES, "max_dolly_response_chars": MAX_DOLLY_RESPONSE_CHARS,
                 "max_oasst_prompt_chars": MAX_OASST_PROMPT_CHARS,
                 "max_oasst_answer_chars": MAX_OASST_ANSWER_CHARS, "max_oasst_toxicity": MAX_OASST_TOXICITY,
                 "holdout": "rows whose prompt or answer contains a holdout sentence, or whose prompt "
                            "sits inside a holdout prompt, are dropped"},
        eval_rule=f"sha256 of the row group modulo {EVAL_MODULUS} == 0, at most {MAX_EVAL_ROWS} rows"))
    print(json.dumps({"train": len(train), "eval": len(evaluation), "dropped": stats["dropped"]}), flush=True)
    print("SFT_DATA_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
