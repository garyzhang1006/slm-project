"""Stage distill_data: the LoRA-tuned SmolLM2 writes short answers for the slm-160m SFT, on a Kaggle GPU.

Stage 3 keeps only Dolly rows whose human answer fits in 400 characters, so thousands of good open_qa and
general_qa questions go unused because their answers run to paragraphs a 160M byte model cannot imitate.
This stage asks the attached adapter, the 1.7B one (208 of 252 everyday questions exact), for a short answer to
each of those questions and writes them in the stage 3 record format. Stage 4 merges them into sft_train when
this kernel is attached.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compute.lora_baseline import (GENERATION_BATCH_SIZE, base_model, encode,  # noqa: E402
                                   ensure_dependencies, find_input, left_pad)
from compute.stage1_corpus import HOLDOUT_PATH, contains_secret, digest, normalize_overlap, write_json  # noqa: E402
from compute.short_facts import SENTENCE_CUE  # noqa: E402
from compute.stages import DISTILL_FILTERS_VERSION  # noqa: E402
from compute.stage3_sft_data import (EVERYDAY_EVAL_PATH, MAX_DOLLY_RESPONSE_CHARS, SOURCES,  # noqa: E402
                                     _load_rows, abbreviated, holdout_conflict, holdout_stems,
                                     one_sentence_answer, sft_record, write_jsonl)

DOLLY = "databricks/databricks-dolly-15k"
# Categories that ask a question a short answer can settle; brainstorming and creative writing do not.
CATEGORIES = frozenset({"open_qa", "general_qa", "classification"})
MAX_PROMPT_CHARS = 300
MAX_ANSWER_CHARS = 400
# A numbered list starts a line or follows a colon, as in "ways: 1. Rest 2. Eat"; "on May 1. ... on June 2." is none.
NUMBERED_LIST = re.compile(r"(?:^|[\n:])\s*1[.)]\s.*?\s2[.)]\s", re.S)
# A bullet list starts a line with "- ", "* " or a bullet sign.
BULLETED_LIST = re.compile(r"(?:^|\n)[ \t]*[-*\u2022][ \t]")
# Greetings that open a reply without answering it, as the first paragraph before the answer.
OPENERS = frozenset({"sure", "certainly", "of course", "okay", "ok", "absolutely", "alright", "all right",
                     "great question", "good question", "sure thing", "happy to help"})
# One sentence that only offers help, the part after the opener in "Sure, I'd be happy to help you with that.".
HELP_OFFER = re.compile(r"[^.!?]*\bhelp\b[^.!?]*[.!]?", re.I)
# A list marker such as "- " or "2. " at the start of a line, set aside when comparing lines.
LIST_MARKER = re.compile(r"^[ \t]*(?:[-*\u2022]|\d+[.)])\s+", re.M)
# An item number after a space and before the item's text on the same line, as in "Try these: 1. Rest 2. Eat",
# read as one only inside a numbered list so that "Team A scored 3. Team A scored 4." keeps its numbers, and never
# before a line break, where "3." ends an item such as "1. Team A scored 3.".
INLINE_NUMBER = re.compile(r"(?<=\s)\d{1,2}[.)][ \t]+(?=\S)")
OUT_DIR = Path("/kaggle/working/distill")


def source_label(model_id: str) -> str:
    """Record source naming the teacher whose adapter wrote the answers, such as
    "distilled:SmolLM2-1.7B-Instruct+LoRA<-databricks/databricks-dolly-15k"."""
    return f"distilled:{model_id.rsplit('/', 1)[-1]}+LoRA<-{DOLLY}"


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
    paragraph, _, rest = text.strip().partition("\n\n")
    paragraph = paragraph.strip()
    # A list with a blank line between items would keep only its first item, which is no whole answer.
    if LIST_MARKER.match(paragraph) and LIST_MARKER.match(rest.lstrip("\n")):
        return ""
    if len(paragraph) <= limit:
        # A lead-in such as "Here are some examples:" introduces the answer, and an exclamation such as "Sure!" or
        # "That's a great question!" or an opener such as "Certainly." before more paragraphs greets it; kept
        # alone, neither is an answer.
        greeting = paragraph.endswith("!") or paragraph.rstrip(".!").lower() in OPENERS
        if paragraph.endswith(":") or ("\n\n" in text.strip() and greeting):
            return ""
        # An opener followed only by an offer to help answers nothing, even as the whole reply.
        opener, comma, offer = paragraph.partition(",")
        if comma and opener.strip().lower() in OPENERS and HELP_OFFER.fullmatch(offer.strip()):
            return ""
        return paragraph
    cut = paragraph[:limit]
    # A list cut short is no whole answer, so the cut may only fall before the list starts.
    starts = [found.start() for found in (NUMBERED_LIST.search(paragraph), BULLETED_LIST.search(paragraph)) if found]
    if starts and min(starts) < limit:
        cut = paragraph[:min(starts) + 1]
    # The character after the cut still counts, so a sentence that ends right at the limit, or at a line break, is kept.
    for end in range(len(cut) - 1, 0, -1):
        if cut[end] not in ".!?" or not paragraph[end + 1].isspace():
            continue
        if cut[end] == "." and abbreviated(cut[:end].rsplit(None, 1)[-1]):
            continue
        # Kept alone, an opener such as "Sure!" or "Great question!" is a fragment of the answer, not an answer.
        return cut[:end + 1] if len(cut[:end].split()) >= 3 else ""
    return ""


def repetitive(text: str) -> bool:
    """True when two lines or sentences of two or more words say the same thing once list markers are set aside,
    as the teacher writes when it loops, such as "1. The Blue Moon 2. The Blue Moon"."""
    if NUMBERED_LIST.search(text):
        text = INLINE_NUMBER.sub("\n", text)
    parts = []
    for line in LIST_MARKER.sub("", text).splitlines():
        start = 0
        for end in re.finditer(r"[.!?](?=\s|$)", line):
            words = line[start:end.start()].split()
            if end.group() == "." and words and abbreviated(words[-1]):
                continue
            parts.append(line[start:end.start()])
            start = end.end()
        parts.append(line[start:])
    parts = [" ".join(part.lower().split()) for part in parts]
    parts = [part for part in parts if len(part.split()) >= 2]
    return len(parts) != len(set(parts))


def mostly_ascii(text: str) -> bool:
    """English-script check that still passes letter-free answers such as "13"."""
    letters = [character for character in text if character.isalpha()]
    return not letters or sum(character.isascii() for character in letters) / len(letters) >= 0.97


def build_records(prompts: list[tuple[int, str]], answers: list[str], stems: list[str],
                  holdout_prompts: list[str], dropped: dict[str, int],
                  finished: list[bool] | None = None, *, source: str) -> list[dict]:
    """Turn generated answers into stage 3 records, counting every rejection reason in dropped.

    finished[i] is False when max_new_tokens, not the end token, stopped answer i."""
    normalized = [normalize_overlap(prompt) for prompt in holdout_prompts]
    records = []
    for (index, prompt), raw, done in zip(prompts, answers, finished or [True] * len(answers)):
        # Stopped inside its first paragraph, the answer may end mid-sentence, such as "It is the 1", and
        # trim_answer keeps a first paragraph of up to MAX_ANSWER_CHARS whole, so that one drops. Every other cut-off
        # answer is safe to trim. A longer single paragraph comes back only up to a sentence end at index
        # MAX_ANSWER_CHARS - 1 or earlier that the model followed with whitespace, so the cut-off end, which lies past
        # MAX_ANSWER_CHARS, never survives. With a blank line in it, only the first paragraph can come back, and the
        # model finished that paragraph before writing the blank line.
        if not done and "\n\n" not in raw.strip() and len(raw.strip()) <= MAX_ANSWER_CHARS:
            dropped["cut_off"] = dropped.get("cut_off", 0) + 1
            continue
        answer = trim_answer(raw)
        # The same rule as stage 3's Dolly rows, so no bare question trains to a full sentence.
        if answer and one_sentence_answer(prompt, answer):
            prompt += SENTENCE_CUE
        record = sft_record(f"distill-{index}", prompt, answer, source, SOURCES[DOLLY]["license"]) if answer else None
        if record is None:
            reason = "empty_or_invalid"
        elif not mostly_ascii(answer):
            reason = "not_english"
        # A 160M student trained on a looping answer would learn the loop.
        elif repetitive(answer):
            reason = "repetitive"
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
    records = build_records(prompts, answers, stems, holdout_prompts, dropped, finished,
                            source=source_label(model_id))
    path = args.out_dir / "distill_train.jsonl"
    write_jsonl(records, path)
    write_json(args.out_dir / "distill_manifest.json", {
        "rows": len(records), "prompts": len(prompts), "dropped": dropped, "sha256": digest(path),
        "filters_version": DISTILL_FILTERS_VERSION,
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
