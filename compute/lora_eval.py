"""Score the base SmolLM2 and the trained LoRA adapter on data/everyday_eval.json, exclusively on Kaggle.

The 24-question holdout shares templates with older short_facts training rows, so its score can flatter
the adapter. The 252 everyday questions are filtered out of the SFT data by stage 3 and give the fairer number.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compute.lora_baseline import (MODEL_ID, MODEL_REVISION, digest, ensure_dependencies,  # noqa: E402
                                   find_input, generate_answers, write_json)

EVAL_FILES = ("data/everyday_eval.json", "data/simple_questions_holdout.json")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args(argv)
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    return args


def adapter_dir(root: Path = Path("/kaggle/input")) -> Path:
    """The attached slm-lora-baseline output must hold exactly one saved adapter."""
    config = find_input("adapter_config.json", root)
    return config.parent


def score_model(torch, model, tokenizer, rows: list[dict], max_new_tokens: int, score) -> dict:
    started = time.monotonic()
    answers = generate_answers(torch, model, tokenizer, [row["prompt"] for row in rows], max_new_tokens)
    predictions = [{"id": row["id"], "answer": answer} for row, answer in zip(rows, answers)]
    return {"predictions": [{**row, "answer": answer} for row, answer in zip(rows, answers)],
            "scores": score(rows, predictions), "seconds": round(time.monotonic() - started, 1)}


def compare(base: dict, adapter: dict) -> dict:
    """Per-category exact-match counts side by side, for the one-line summary."""
    categories = sorted(set(base["scores"]["categories"]) | set(adapter["scores"]["categories"]))
    empty = {"exact": 0, "scored": 0}
    return {name: {"base": base["scores"]["categories"].get(name, empty)["exact"],
                   "adapter": adapter["scores"]["categories"].get(name, empty)["exact"],
                   "scored": adapter["scores"]["categories"].get(name, empty)["scored"]}
            for name in categories}


def main(argv: list[str] | None = None) -> None:
    if not Path("/kaggle/working").is_dir():
        raise RuntimeError("Run the LoRA eval on Kaggle; local model execution is prohibited")
    args = parse_args(argv)
    os.chdir(ROOT)
    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
    os.environ.update(PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
    from score_holdout import REPORT_KEYS, score_predictions

    versions = ensure_dependencies()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the LoRA eval; enable the T4 accelerator")
    adapter = adapter_dir()
    destination = ROOT / "lora_eval_report.json"
    sets = {REPORT_KEYS[Path(name).name]: json.loads((ROOT / name).read_text())["rows"] for name in EVAL_FILES}
    report = {"status": "evaluating", "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
              "versions": versions, "adapter": str(adapter),
              "adapter_sha256": digest(adapter / "adapter_model.safetensors")
              if (adapter / "adapter_model.safetensors").exists() else None,
              "eval_sha256": {name: digest(ROOT / name) for name in EVAL_FILES}, "arguments": vars(args),
              "gpu": torch.cuda.get_device_name(0), "base": {}, "lora": {}}
    write_json(destination, report)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # fp32 to match training: fp16 overflowed SmolLM2 activations in the first LoRA run.
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION, torch_dtype=torch.float32)
    model.to("cuda")
    for key, rows in sets.items():
        report["base"][key] = score_model(torch, model, tokenizer, rows, args.max_new_tokens, score_predictions)
        write_json(destination, report)
        print("base", key, json.dumps(report["base"][key]["scores"]["total"]), flush=True)
    model = PeftModel.from_pretrained(model, str(adapter))
    for key, rows in sets.items():
        report["lora"][key] = score_model(torch, model, tokenizer, rows, args.max_new_tokens, score_predictions)
        write_json(destination, report)
        print("lora", key, json.dumps(report["lora"][key]["scores"]["total"]), flush=True)
    # A top-level everyday_eval list lets `scripts/score_holdout.py --holdout data/everyday_eval.json`
    # re-score this report after manual review.
    report["everyday_eval"] = [{"id": row["id"], "answer": row["answer"]}
                               for row in report["lora"]["everyday_eval"]["predictions"]]
    report["by_category"] = compare(report["base"]["everyday_eval"], report["lora"]["everyday_eval"])
    report["status"] = "complete_pending_manual_review"
    write_json(destination, report)
    print("LORA_EVAL_COMPLETE", json.dumps(report["by_category"]), flush=True)


if __name__ == "__main__":
    main()
