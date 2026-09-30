"""Stage distill_data: the LoRA-tuned SmolLM2 writes short answers for the slm-160m SFT, on a Kaggle GPU.

Stage 3 keeps only Dolly rows whose human answer fits in 400 characters, so thousands of good open_qa and
general_qa questions go unused because their answers run to paragraphs a 160M byte model cannot imitate.
This stage asks the adapter (18 of 22 exact on the holdout) for a short answer to each of those questions
and writes them in the stage 3 record format. Stage 4 merges them into sft_train when this kernel is attached.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compute.lora_baseline import (GENERATION_BATCH_SIZE, base_model, encode,  # noqa: E402
                                   ensure_dependencies, find_input, left_pad)
from compute.stage1_corpus import HOLDOUT_PATH, contains_secret, digest, normalize_overlap, write_json  # noqa: E402
from compute.short_facts import SENTENCE_CUE  # noqa: E402
from compute.stage3_sft_data import (EVERYDAY_EVAL_PATH, MAX_DOLLY_RESPONSE_CHARS, SOURCES,  # noqa: E402
                                     _load_rows, holdout_conflict, holdout_stems, one_sentence_answer,
                                     sft_record, write_jsonl)

DOLLY = "databricks/databricks-dolly-15k"
# Categories that ask a question a short answer can settle; brainstorming and creative writing do not.
CATEGORIES = frozenset({"open_qa", "general_qa", "classification"})
MAX_PROMPT_CHARS = 300
MAX_ANSWER_CHARS = 400
# A period after these words, or after a single letter such as the J of "J. K." or the C of "D.C.", ends no sentence.
ABBREVIATIONS = frozenset({"mr", "mrs", "ms", "dr", "st", "jr", "sr", "prof", "vs", "etc", "e.g", "i.e", "approx"})
OUT_DIR = Path("/kaggle/working/distill")
SOURCE = "distilled:SmolLM2-360M-Instruct+LoRA<-databricks/databricks-dolly-15k"


def candidate_prompts(raws, limit: int) -> list[tuple[int, str]]:
    """(dolly index, instruction) for context-free questions whose human answer stage 3 dropped as too long."""
    chosen = []
    for index, raw in enumerate(raws):
        instruction, context, response, category = (raw.get(key) or "" for key in
                                                    ("instruction", "context", "response", "category"))
        if not all(isinstance(value, str) for value in (instruction, context, response, category)):
            continue
        if (category in CATEGORIES and not context.strip() and instruction.strip()
                and len(instruction) <= MAX_PROMPT_CHARS and len(response) > MAX_DOLLY_RESPONSE_CHARS):
            chosen.append((index, instruction.strip()))
    # Hash order, not file order, so a limit samples every category instead of the first few thousand rows.
    chosen.sort(key=lambda item: hashlib.sha256(item[1].encode("utf-8")).hexdigest())
    return chosen[:limit]


def trim_answer(text: str, limit: int = MAX_ANSWER_CHARS) -> str:
    """First paragraph, cut back to a sentence end when it runs past limit; '' when nothing usable is left."""
    paragraph = text.strip().split("\n\n", 1)[0].strip()
    if len(paragraph) <= limit:
        # A lead-in such as "Here are some examples:" or an opener such as "Sure!" before more paragraphs
        # only introduces the answer, so kept alone it is no answer.
        if paragraph.endswith(":") or ("\n\n" in text.strip() and len(paragraph.split()) < 3):
            return ""
        return paragraph
    cut = paragraph[:limit]
    for end in range(len(cut) - 2, 0, -1):
        if cut[end] not in ".!?" or cut[end + 1] != " ":
            continue
        word = cut[:end].rsplit(None, 1)[-1].lstrip("(\"'").lower()
        if cut[end] == "." and (word in ABBREVIATIONS or len(word.rsplit(".", 1)[-1]) == 1 and word[-1:].isalpha()):
            continue
        # Kept alone, an opener such as "Sure!" or "Great question!" is a fragment of the answer, not an answer.
        return cut[:end + 1] if len(cut[:end].split()) >= 3 else ""
    return ""


def mostly_ascii(text: str) -> bool:
    """English-script check that still passes letter-free answers such as "13"."""
    letters = [character for character in text if character.isalpha()]
    return not letters or sum(character.isascii() for character in letters) / len(letters) >= 0.97


def build_records(prompts: list[tuple[int, str]], answers: list[str], stems: list[str],
                  holdout_prompts: list[str], dropped: dict[str, int],
                  finished: list[bool] | None = None) -> list[dict]:
    """Turn generated answers into stage 3 records, counting every rejection reason in dropped.

    finished[i] is False when max_new_tokens, not the end token, stopped answer i."""
    normalized = [normalize_overlap(prompt) for prompt in holdout_prompts]
    records = []
    for (index, prompt), raw, done in zip(prompts, answers, finished or [True] * len(answers)):
        # Stopped inside its first paragraph, the answer may end mid-sentence, such as "It is the 1".
        if not done and "\n\n" not in raw.strip():
            dropped["cut_off"] = dropped.get("cut_off", 0) + 1
            continue
        answer = trim_answer(raw)
        # The same rule as stage 3's Dolly rows, so no bare question trains to a full sentence.
        if answer and one_sentence_answer(prompt, answer):
            prompt += SENTENCE_CUE
        record = sft_record(f"distill-{index}", prompt, answer, SOURCE, SOURCES[DOLLY]["license"]) if answer else None
        if record is None:
            reason = "empty_or_invalid"
        elif not mostly_ascii(answer):
            reason = "not_english"
        elif holdout_conflict(record, stems, normalized):
            reason = "holdout_overlap"
        elif contains_secret(prompt) or contains_secret(answer):
            reason = "secret_pattern"
        else:
            records.append(record)
            continue
        dropped[reason] = dropped.get(reason, 0) + 1
    return records


def generate(torch, model, tokenizer, prompts: list[str], max_new_tokens: int) -> tuple[list[str], list[bool]]:
    """Greedy full answers (not first lines), batched and left-padded like lora_baseline.generate_answers,
    and for each whether the model ended it with the end token rather than running into max_new_tokens."""
    model.train(False)
    answers, finished = [], []
    with torch.no_grad():
        for start in range(0, len(prompts), GENERATION_BATCH_SIZE):
            ids, mask = left_pad([encode(tokenizer, prompt) for prompt in prompts[start:start + GENERATION_BATCH_SIZE]],
                                 tokenizer.pad_token_id)
            ids = torch.tensor(ids, device=model.device)
            output = model.generate(input_ids=ids, attention_mask=torch.tensor(mask, device=model.device),
                                    do_sample=False, max_new_tokens=max_new_tokens,
                                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            new = output[:, ids.shape[1]:]
            answers += [tokenizer.decode(row, skip_special_tokens=True) for row in new]
            finished += (new == tokenizer.eos_token_id).any(dim=1).tolist()
            if start // GENERATION_BATCH_SIZE % 25 == 0:
                print(json.dumps({"generated": len(answers), "of": len(prompts)}), flush=True)
    return answers, finished


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-prompts", type=int, default=6000)
    parser.add_argument("--max-new-tokens", type=int, default=112)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)
    if args.max_prompts < 1 or args.max_new_tokens < 1:
        parser.error("--max-prompts and --max-new-tokens must be positive")
    return args


def main(argv: list[str] | None = None) -> None:
    if not Path("/kaggle/working").is_dir():
        raise RuntimeError("Distillation runs the LoRA model; run it on Kaggle, never locally")
    args = parse_args(argv)
    os.chdir(ROOT)
    sys.path[:0] = [str(ROOT / "src")]
    os.environ.update(PYTHONPATH=str(ROOT / "src"), PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
    started = time.monotonic()
    holdout_prompts = [row["prompt"] for path in (HOLDOUT_PATH, EVERYDAY_EVAL_PATH)
                       for row in json.loads((ROOT / path).read_text())["rows"]]
    stems = holdout_stems(holdout_prompts)
    normalized = [normalize_overlap(prompt) for prompt in holdout_prompts]
    raws = _load_rows(DOLLY, ("instruction", "context", "response", "category"))
    # Screen prompts before generating, so GPU time is never spent on rows that would be dropped anyway.
    prompts = [(index, prompt) for index, prompt in candidate_prompts(raws, len(raws))
               if not holdout_conflict({"prompt": prompt, "answer": ""}, stems, normalized)][:args.max_prompts]
    if len(prompts) < 500:
        raise RuntimeError(f"Only {len(prompts)} candidate prompts; expected several thousand from Dolly")

    versions = ensure_dependencies()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for distillation; enable the T4 accelerator")
    adapter = find_input("adapter_config.json").parent
    model_id, model_revision = base_model(adapter)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=model_revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # fp32, as in training: fp16 overflowed SmolLM2 activations.
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=model_revision, torch_dtype=torch.float32)
    model = PeftModel.from_pretrained(model.to("cuda"), str(adapter))
    answers, finished = generate(torch, model, tokenizer, [prompt for _, prompt in prompts], args.max_new_tokens)
    dropped: dict[str, int] = {}
    records = build_records(prompts, answers, stems, holdout_prompts, dropped, finished)
    path = args.out_dir / "distill_train.jsonl"
    write_jsonl(records, path)
    write_json(args.out_dir / "distill_manifest.json", {
        "rows": len(records), "prompts": len(prompts), "dropped": dropped, "sha256": digest(path),
        "teacher": {"model_id": model_id, "model_revision": model_revision, "adapter": str(adapter),
                    "adapter_sha256": digest(adapter / "adapter_model.safetensors")
                    if (adapter / "adapter_model.safetensors").exists() else None},
        "prompt_source": {DOLLY: SOURCES[DOLLY]}, "versions": versions,
        "filters": {"categories": sorted(CATEGORIES), "max_prompt_chars": MAX_PROMPT_CHARS,
                    "max_answer_chars": MAX_ANSWER_CHARS,
                    "human_answer_longer_than": MAX_DOLLY_RESPONSE_CHARS},
        "seconds": round(time.monotonic() - started, 1)})
    print("DISTILL_COMPLETE", json.dumps({"rows": len(records), "dropped": dropped}), flush=True)


if __name__ == "__main__":
    main()
