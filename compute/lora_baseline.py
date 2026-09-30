"""LoRA fast path: fine-tune an open instruct model on sft_train.jsonl, exclusively on Kaggle."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

# Verified through https://huggingface.co/api/models/HuggingFaceTB/SmolLM2-360M-Instruct on 2026-09-25:
# license apache-2.0, not gated, LlamaForCausalLM, ships model.safetensors and a tokenizer whose
# config has a ChatML chat_template, eos_token and pad_token both <|im_end|>, and bos_token <|im_start|>.
MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
MODEL_REVISION = "a10cc1512eabd3dde888204e902eca88bddb4951"
MODEL_LICENSE = "apache-2.0"
# --model choices. The 1.7B revision was read from https://huggingface.co/api/models/HuggingFaceTB/
# SmolLM2-1.7B-Instruct on 2026-09-26 (license apache-2.0, same ChatML tokenizer family as the 360M model).
MODELS = {
    "360m": {"model_id": MODEL_ID, "model_revision": MODEL_REVISION},
    "1.7b": {"model_id": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
             "model_revision": "31b70e2e869a7173562077fd711b654946d38674"},
}
# fp32 1.7B weights take about 6.8 GB of the T4's 16 GB, so it trains on smaller micro-batches with
# gradient checkpointing, and one epoch keeps it inside the --max-seconds budget.
MODEL_DEFAULTS = {
    "360m": {"epochs": 2.0, "batch_size": 8, "gradient_accumulation_steps": 2, "gradient_checkpointing": False},
    "1.7b": {"epochs": 1.0, "batch_size": 4, "gradient_accumulation_steps": 4, "gradient_checkpointing": True},
}
# Written next to the adapter so lora_eval, distill_data and Studio load the base it was trained on.
BASE_MODEL_FILE = "base_model.json"
# Standard projection names of the transformers Llama implementation that SmolLM2 uses.
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
SYSTEM_PROMPT = "You are a helpful assistant. Answer in plain English with a short, direct reply."
# The custom checkpoint answered 3 of the 24 holdout questions correctly before this pipeline existed.
PREVIOUS_CUSTOM_CORRECT = 3
# Worded unlike any training prompt, so the answers show phrasing the adapter has not seen.
ENGLISH_PROBES = (
    "What color is the sky on a sunny day with no clouds?",
    "Write one sentence about a dog.",
    "Tell me the opposite of hot.",
    "What word means more than one mouse?",
    "Which day of the week follows Monday?",
    "Can you tell me what my name is?",
)
IGNORE_INDEX = -100
HOLDOUT_FILE = "data/simple_questions_holdout.json"
EVERYDAY_FILE = "data/everyday_eval.json"
# Kernel outputs that supply each file find_input looks for, so a missing-file error names the right one.
INPUT_SOURCES = {
    "sft_train.jsonl": "the slm-sft-data kernel output",
    "sft_eval.jsonl": "the slm-sft-data kernel output",
    "adapter_config.json": "the slm-lora-baseline or slm-lora-1b7 kernel output",
}
# 16 prompts of under 100 tokens plus 64 new tokens fit a 16 GB T4 next to the fp32 360M model.
GENERATION_BATCH_SIZE = 16


def base_model(adapter: Path) -> tuple[str, str]:
    """(model_id, revision) an adapter was trained on; adapters saved before --model existed are 360M."""
    path = adapter / BASE_MODEL_FILE
    if not path.exists():
        return MODEL_ID, MODEL_REVISION
    base = json.loads(path.read_text())
    return base["model_id"], base["model_revision"]


def normalize(text: str) -> str:
    """Casefold and keep word and number tokens only, so punctuation or spacing cannot hide a holdout copy."""
    return " ".join(re.findall(r"[^\W_]+", text.casefold().replace("'", "").replace("\u2019", "")))


def holdout_keys(holdout_rows: list[dict]) -> tuple[set[str], set[str]]:
    """Whole holdout prompts, plus each question sentence alone (so "What is 4 plus 9?" is caught
    even without the "Reply with the number." suffix)."""
    prompts = {normalize(row["prompt"]) for row in holdout_rows}
    questions = {normalize(part) for row in holdout_rows for part in re.findall(r"[^.?!:]*\?", row["prompt"])}
    return prompts, {key for key in prompts | questions if len(key.split()) >= 4}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def find_input(name: str, root: Path = Path("/kaggle/input")) -> Path:
    """Locate exactly one attached file; ambiguity would silently train on the wrong data."""
    matches = sorted(root.rglob(name)) if root.is_dir() else []
    if len(matches) != 1:
        source = INPUT_SOURCES.get(name, "the kernel that writes it")
        raise RuntimeError(f"Expected exactly one {name} under {root}, found {len(matches)}: "
                           f"{[str(path) for path in matches]}; attach {source} (and only one copy)")
    return matches[0]


def load_sft_rows(path: Path) -> list[dict]:
    """Read prompt/answer JSONL rows written by compute/stage3_sft_data.py."""
    rows = []
    # str.splitlines() would also break at the raw U+2028 that json.dumps(ensure_ascii=False) leaves in a row.
    for number, line in enumerate(path.read_text(encoding="utf-8").split("\n"), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not all(isinstance(row.get(field), str) and row[field].strip() for field in ("prompt", "answer")):
            raise ValueError(f"{path}:{number}: prompt and answer must be non-empty strings")
        rows.append(row)
    if not rows:
        raise ValueError(f"{path} holds no rows")
    return rows


def screened_rows(root: Path) -> list[dict]:
    """Rows of both eval files. Stage 3 screens against both, and so does this trainer, so a stale or
    hand-built sft_train cannot leak everyday_eval questions into the adapter."""
    return [row for name in (HOLDOUT_FILE, EVERYDAY_FILE) for row in json.loads((root / name).read_text())["rows"]]


def drop_holdout_overlap(rows: list[dict], holdout_rows: list[dict]) -> tuple[list[dict], int]:
    """Defensive re-check of stage 3's filter, so holdout questions never become training targets."""
    prompts, fragments = holdout_keys(holdout_rows)

    def overlaps(row: dict) -> bool:
        prompt, answer = normalize(row["prompt"]), normalize(row["answer"])
        padded = f" {prompt} "
        return (prompt in prompts or answer in prompts
                or any(f" {fragment} " in padded for fragment in fragments))

    kept = [row for row in rows if not overlaps(row)]
    return kept, len(rows) - len(kept)


def build_messages(prompt: str) -> list[dict]:
    # An explicit system turn keeps prompting identical whatever default the chat template carries.
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]


def mask_prompt(prompt_ids: list[int], answer_ids: list[int], max_length: int) -> tuple[list[int], list[int]] | None:
    """Join prompt and answer, computing loss on answer tokens only; None when no answer token fits."""
    ids = (prompt_ids + answer_ids)[:max_length]
    labels = ([IGNORE_INDEX] * len(prompt_ids) + answer_ids)[:max_length]
    if all(label == IGNORE_INDEX for label in labels):
        return None
    return ids, labels


def collate(examples: list[tuple[list[int], list[int]]], pad_id: int) -> dict[str, list[list[int]]]:
    """Right-pad a batch; padded positions get no attention and no loss."""
    width = max(len(ids) for ids, _ in examples)
    batch = {"input_ids": [], "labels": [], "attention_mask": []}
    for ids, labels in examples:
        padding = width - len(ids)
        batch["input_ids"].append(ids + [pad_id] * padding)
        batch["labels"].append(labels + [IGNORE_INDEX] * padding)
        batch["attention_mask"].append([1] * len(ids) + [0] * padding)
    return batch


def window_weights(window: list[list[tuple[list[int], list[int]]]]) -> list[float]:
    """Each micro-batch's share of the answer tokens in one accumulation window.

    The model's loss is a mean over its own micro-batch, so scaling by these shares makes the update match one
    combined batch, the same per-token average evaluation_loss reports.
    """
    # The causal loss shifts labels left by one, so a first-position label never counts.
    counts = [sum(label != IGNORE_INDEX for _, labels in rows for label in labels[1:]) for rows in window]
    return [count / sum(counts) for count in counts]


def planned_optimizer_steps(rows: int, batch_size: int, accumulation: int, epochs: float, max_steps: int) -> int:
    per_epoch = math.ceil(rows / (batch_size * accumulation))
    return max(1, min(max_steps, math.ceil(per_epoch * epochs)))


def learning_rate_at(step: int, total: int, warmup: int, peak: float) -> float:
    """Linear warmup then linear decay toward zero at the final optimizer step."""
    if warmup and step < warmup:
        return peak * (step + 1) / warmup
    return peak * max(0.0, (total - step) / max(1, total - warmup))


def first_line(text: str) -> str:
    """Evaluation stops at the first newline, matching compute/stage5_evaluate.py."""
    return text.strip().split("\n", 1)[0].strip()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODELS), default="360m",
                        help="SmolLM2 base to adapt; unset training flags take this model's defaults")
    parser.add_argument("--epochs", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--max-seconds", type=float, default=3 * 3600,
                        help="Stop training early and still evaluate once this wall-clock budget is spent")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--eval-rows", type=int, default=512, help="sft_eval rows used for held-out loss")
    parser.add_argument("--eval-every", type=int, default=250,
                        help="Optimizer steps between held-out loss checks; the best adapter is kept")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args(argv)
    for name, value in MODEL_DEFAULTS[args.model].items():
        if getattr(args, name, None) is None:
            setattr(args, name, value)
    args.model_id, args.model_revision = MODELS[args.model]["model_id"], MODELS[args.model]["model_revision"]
    if min(args.batch_size, args.gradient_accumulation_steps, args.max_steps, args.max_length) < 1:
        parser.error("batch size, accumulation, max steps and max length must be positive")
    if args.eval_every < 0:
        parser.error("--eval-every must be 0 (off) or positive")
    return args


# The versions field of both 2026-09 lora_report.json files. An unpinned install takes whatever release is
# newest on the day, so a rerun could train or score with different library code than the reported adapters.
PINNED_VERSIONS = {"transformers": "5.0.0", "peft": "0.19.1", "accelerate": "1.13.0"}


def ensure_dependencies() -> dict:
    """Kaggle images ship transformers; missing packages are installed because this stage has internet on."""
    missing = [name for name in PINNED_VERSIONS if importlib.util.find_spec(name) is None]
    if missing:
        pins = [f"{name}=={PINNED_VERSIONS[name]}" for name in missing]
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", *pins], check=True)
    # peft raises ImportError while wrapping layers when an old torchao is installed (Kaggle ships 0.10.0,
    # peft wants >0.16.0). This stage never quantizes, so removing torchao is safer than upgrading it
    # against the image's torch build.
    if importlib.util.find_spec("torchao") is not None:
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "--quiet", "--yes", "torchao"], check=True)
    from importlib.metadata import version
    return {name: version(name) for name in ("torch", "transformers", "peft", "accelerate")}


def encode(tokenizer, prompt: str, answer: str | None = None):
    """Prompt ids alone, or (prompt ids, answer ids ending in EOS) when an answer is given."""
    # Render the template to text and tokenize it here: apply_chat_template(tokenize=True) returns a list in
    # older transformers but a BatchEncoding in newer ones, and list() of that yields its key names.
    if tokenizer.chat_template:
        text = tokenizer.apply_chat_template(build_messages(prompt), add_generation_prompt=True, tokenize=False)
    else:
        text = f"{SYSTEM_PROMPT}\nQuestion: {prompt}\nAnswer:"
    prompt_ids = list(tokenizer(text, add_special_tokens=False).input_ids)
    if answer is None:
        return prompt_ids
    text = answer.strip() if tokenizer.chat_template else " " + answer.strip()
    return prompt_ids, tokenizer(text, add_special_tokens=False).input_ids + [tokenizer.eos_token_id]


def left_pad(sequences: list[list[int]], pad_id: int) -> tuple[list[list[int]], list[list[int]]]:
    """Left-pad prompts for batched decoder-only generation, so every row's new tokens start at one column."""
    width = max(len(sequence) for sequence in sequences)
    ids = [[pad_id] * (width - len(sequence)) + sequence for sequence in sequences]
    mask = [[0] * (width - len(sequence)) + [1] * len(sequence) for sequence in sequences]
    return ids, mask


def generate_answers(torch, model, tokenizer, prompts: list[str], max_new_tokens: int,
                     batch_size: int = GENERATION_BATCH_SIZE) -> list[str]:
    """Greedy answers, batch_size prompts per generate call; one prompt per call left the T4 mostly idle."""
    model.train(False)
    answers = []
    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            ids, mask = left_pad([encode(tokenizer, prompt) for prompt in prompts[start:start + batch_size]],
                                 tokenizer.pad_token_id)
            ids = torch.tensor(ids, device=model.device)
            output = model.generate(input_ids=ids, attention_mask=torch.tensor(mask, device=model.device),
                                    do_sample=False, max_new_tokens=max_new_tokens,
                                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            answers += [first_line(tokenizer.decode(row, skip_special_tokens=True)) for row in output[:, ids.shape[1]:]]
    return answers


def evaluate_holdout(torch, model, tokenizer, holdout_rows: list[dict], max_new_tokens: int, score) -> dict:
    answers = generate_answers(torch, model, tokenizer, [row["prompt"] for row in holdout_rows], max_new_tokens)
    predictions = [{"id": row["id"], "answer": answer} for row, answer in zip(holdout_rows, answers)]
    probes = generate_answers(torch, model, tokenizer, list(ENGLISH_PROBES), max_new_tokens)
    return {"predictions": [{**row, "answer": answer} for row, answer in zip(holdout_rows, answers)],
            "scores": score(holdout_rows, predictions),
            "english_probes": [{"prompt": prompt, "answer": answer} for prompt, answer in zip(ENGLISH_PROBES, probes)]}


class BestAdapter:
    """CPU copy of the trainable (LoRA) weights with the lowest held-out loss seen so far."""

    def __init__(self) -> None:
        self.loss, self.step, self.weights = None, None, {}

    def offer(self, loss: float | None, step: int, named_parameters) -> bool:
        if loss is None or not math.isfinite(loss) or (self.loss is not None and loss >= self.loss):
            return False
        self.loss, self.step = loss, step
        self.weights = {name: parameter.detach().to("cpu", copy=True)
                        for name, parameter in named_parameters if parameter.requires_grad}
        return True

    def restore(self, named_parameters) -> None:
        for name, parameter in named_parameters:
            if name in self.weights:
                parameter.data.copy_(self.weights[name].to(parameter.device))


def evaluation_loss(torch, model, examples: list, batch_size: int, pad_id: int) -> float | None:
    """Mean answer-token cross entropy in nats over the encoded sft_eval rows."""
    if not examples:
        return None
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch = {key: torch.tensor(value, device=model.device)
                     for key, value in collate(examples[start:start + batch_size], pad_id).items()}
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
            labels = batch["labels"][:, 1:]
            losses = torch.nn.functional.cross_entropy(
                logits[:, :-1].float().reshape(-1, logits.shape[-1]), labels.reshape(-1),
                ignore_index=IGNORE_INDEX, reduction="sum")
            total += losses.item()
            count += int((labels != IGNORE_INDEX).sum().item())
    model.train()
    return total / max(1, count)


def main(argv: list[str] | None = None) -> None:
    if not Path("/kaggle/working").is_dir():
        raise RuntimeError("Run the LoRA baseline on Kaggle; local model execution is prohibited")
    args = parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    sys.path[:0] = [str(root), str(root / "src"), str(root / "scripts")]
    os.environ.update(PYTHONPATH=str(root / "src"), PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
    manifest_path = root / "source-manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for name, expected in manifest.items():
        if digest(root / name) != expected:
            raise RuntimeError(f"Source hash mismatch: {name}")
    from score_holdout import score_predictions

    versions = ensure_dependencies()
    import random
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the LoRA baseline; enable the T4 accelerator")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    holdout_path = root / HOLDOUT_FILE
    holdout = json.loads(holdout_path.read_text())["rows"]
    screened = screened_rows(root)
    train_path, eval_path = find_input("sft_train.jsonl"), find_input("sft_eval.jsonl")
    train_rows, dropped_train = drop_holdout_overlap(load_sft_rows(train_path), screened)
    eval_rows, dropped_eval = drop_holdout_overlap(load_sft_rows(eval_path), screened)
    artifacts = root / "artifacts"
    artifacts.mkdir(exist_ok=True)
    destination = root / "lora_report.json"
    report = {
        "status": "baseline", "model_id": args.model_id, "model_revision": args.model_revision,
        "model_license": MODEL_LICENSE, "versions": versions, "arguments": vars(args),
        "gpu": torch.cuda.get_device_name(0), "source_manifest": manifest,
        "data": {"train": {"path": str(train_path), "sha256": digest(train_path), "rows": len(train_rows),
                           "dropped_holdout_overlap": dropped_train},
                 "eval": {"path": str(eval_path), "sha256": digest(eval_path), "rows": len(eval_rows),
                          "dropped_holdout_overlap": dropped_eval}},
        "holdout_sha256": digest(holdout_path), "everyday_eval_sha256": digest(root / EVERYDAY_FILE),
        "pass_gate": (f"Holdout exact matches above the custom checkpoint's {PREVIOUS_CUSTOM_CORRECT}/24; "
                      "unknown-category rows still need manual review"),
    }
    write_json(destination, report)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=args.model_revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Plain fp32 throughout. fp16 autocast overflowed SmolLM2 activations: the untouched base model's
    # eval loss was already NaN on the first Kaggle run, and the adapter came out unusable.
    # The T4 has no native bf16, and 360M fp32 parameters still fit its 16 GB.
    model = AutoModelForCausalLM.from_pretrained(args.model_id, revision=args.model_revision,
                                                 torch_dtype=torch.float32)
    model.to("cuda")
    report["baseline"] = evaluate_holdout(torch, model, tokenizer, holdout, args.max_new_tokens, score_predictions)
    report["status"] = "training"
    write_json(destination, report)
    print("baseline holdout", json.dumps(report["baseline"]["scores"]["total"]), flush=True)

    config = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                        target_modules=list(TARGET_MODULES), task_type="CAUSAL_LM")
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, config)
    model.config.use_cache = False
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    report["parameters"] = {"trainable": sum(p.numel() for p in trainable),
                            "total": sum(p.numel() for p in model.parameters())}
    examples, eval_examples = [], []
    for rows, target in ((train_rows, examples), (eval_rows[:args.eval_rows], eval_examples)):
        for row in rows:
            encoded = mask_prompt(*encode(tokenizer, row["prompt"], row["answer"]), args.max_length)
            if encoded:
                target.append(encoded)
    if not examples:
        raise RuntimeError(f"No training rows fit in --max-length {args.max_length}")
    total_steps = planned_optimizer_steps(len(examples), args.batch_size, args.gradient_accumulation_steps,
                                          args.epochs, args.max_steps)
    warmup = int(total_steps * args.warmup_ratio)
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.0)
    report["training"] = {"planned_steps": total_steps, "warmup_steps": warmup, "encoded_train": len(examples),
                          "encoded_eval": len(eval_examples), "log": [],
                          "initial_eval_loss": evaluation_loss(torch, model, eval_examples, args.batch_size,
                                                               tokenizer.pad_token_id)}
    write_json(destination, report)
    initial = report["training"]["initial_eval_loss"]
    if initial is not None and not math.isfinite(initial):
        raise RuntimeError(f"Base model eval loss is {initial} before any training; fix numerics before "
                           "spending GPU time on the adapter")
    best = BestAdapter()
    started, step, micro, order = time.monotonic(), 0, 0, []
    stopped_reason = "planned_steps"
    model.train()
    while step < total_steps:
        window = []
        for _ in range(args.gradient_accumulation_steps):
            if not order:
                order = list(range(len(examples)))
                random.shuffle(order)
            indices, order = order[:args.batch_size], order[args.batch_size:]
            window.append([examples[index] for index in indices])
        window_loss = 0.0
        for rows, weight in zip(window, window_weights(window)):
            batch = {key: torch.tensor(value, device="cuda")
                     for key, value in collate(rows, tokenizer.pad_token_id).items()}
            loss = model(**batch).loss * weight
            if not torch.isfinite(loss):
                # Stop at once: a non-finite loss corrupts the adapter and would waste the rest of the GPU session.
                report.update(status="failed_nonfinite_loss", failed_at_step=step)
                write_json(destination, report)
                raise RuntimeError(f"Non-finite training loss at optimizer step {step} (micro-batch {micro}); "
                                   f"see {destination}")
            loss.backward()
            window_loss += loss.item()
            micro += 1
        for group in optimizer.param_groups:
            group["lr"] = learning_rate_at(step, total_steps, warmup, args.learning_rate)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        if step % 50 == 0 or step == total_steps:
            entry = {"step": step, "loss": window_loss,
                     "learning_rate": optimizer.param_groups[0]["lr"],
                     "elapsed_seconds": round(time.monotonic() - started, 1)}
            report["training"]["log"].append(entry)
            print(json.dumps(entry), flush=True)
            write_json(destination, report)
        if args.eval_every and step % args.eval_every == 0 and step < total_steps:
            held_out = evaluation_loss(torch, model, eval_examples, args.batch_size, tokenizer.pad_token_id)
            kept = best.offer(held_out, step, model.named_parameters())
            report["training"].setdefault("eval_log", []).append({"step": step, "eval_loss": held_out, "best": kept})
            print(json.dumps({"step": step, "eval_loss": held_out, "best": kept}), flush=True)
            write_json(destination, report)
        if time.monotonic() - started > args.max_seconds:
            stopped_reason = "max_seconds"
            break
    final_loss = evaluation_loss(torch, model, eval_examples, args.batch_size, tokenizer.pad_token_id)
    best.offer(final_loss, step, model.named_parameters())
    # A late loss spike (or overfitting in the last epoch) should not decide which adapter ships.
    restored = best.step is not None and best.step != step
    if restored:
        best.restore(model.named_parameters())
    report["training"].update(
        completed_steps=step, stopped_reason=stopped_reason, seconds=round(time.monotonic() - started, 1),
        final_eval_loss=final_loss, best_step=best.step, best_eval_loss=best.loss, restored_best=restored)
    adapter = artifacts / "lora-adapter"
    model.save_pretrained(adapter)
    tokenizer.save_pretrained(adapter)
    base = {"model_id": args.model_id, "model_revision": args.model_revision}
    write_json(adapter / BASE_MODEL_FILE, base)
    weights = adapter / "adapter_model.safetensors"
    # run_pipeline compares this hash with lora_eval and distill_data reports to spot runs on an older adapter.
    report.update(status="evaluating", adapter=str(adapter),
                  adapter_sha256=digest(weights) if weights.exists() else None)
    write_json(destination, report)
    model.config.use_cache = True
    report["final"] = evaluate_holdout(torch, model, tokenizer, holdout, args.max_new_tokens, score_predictions)
    report["passes_gate"] = report["final"]["scores"]["total"]["exact"] > PREVIOUS_CUSTOM_CORRECT
    # Plain transformers weights with the adapter folded in, so Studio can serve it without peft.
    # Last, because merge_and_unload rewrites the base layers in place.
    merged = artifacts / "lora-merged"
    model.merge_and_unload().save_pretrained(merged)
    tokenizer.save_pretrained(merged)
    write_json(merged / BASE_MODEL_FILE, base)
    report["merged"] = str(merged)
    report["status"] = "complete_pending_manual_review"
    write_json(destination, report)
    print("final holdout", json.dumps(report["final"]["scores"]["total"]), report["status"], flush=True)


if __name__ == "__main__":
    main()
