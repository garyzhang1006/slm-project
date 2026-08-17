"""Autoregressive generation from a saved checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .config import ModelConfig
from .data import format_prompt, validate_record
from .model import CognitionSLM
from .tokenizer import ByteTokenizer


@torch.no_grad()
def generate_ids(
    model: CognitionSLM,
    input_ids: torch.Tensor,
    tokenizer: ByteTokenizer,
    *,
    max_new_tokens: int = 96,
    temperature: float = 0.8,
    top_k: int = 40,
) -> torch.Tensor:
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    model.eval()
    generated = input_ids
    for _ in range(max_new_tokens):
        context = generated[:, -model.config.block_size :]
        attention_mask = torch.ones_like(context)
        output = model(context, attention_mask=attention_mask)
        next_logits = output.logits[:, -1, :]
        next_logits[:, tokenizer.pad_id] = float("-inf")
        next_logits[:, tokenizer.bos_id] = float("-inf")
        if temperature == 0:
            next_token = next_logits.argmax(dim=-1, keepdim=True)
        else:
            next_logits = next_logits / temperature
            if top_k > 0:
                k = min(top_k, next_logits.size(-1))
                values, _ = torch.topk(next_logits, k)
                cutoff = values[:, [-1]]
                next_logits = next_logits.masked_fill(next_logits < cutoff, float("-inf"))
            probabilities = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
        generated = torch.cat([generated, next_token], dim=1)
        if bool((next_token == tokenizer.eos_id).all()):
            break
    return generated


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[CognitionSLM, ByteTokenizer]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must be a dictionary of tensors and plain metadata")
    required = {"model_config", "model_state_dict"}
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"checkpoint missing required fields: {sorted(missing)}")
    config = ModelConfig.from_dict(checkpoint["model_config"])
    model = CognitionSLM(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return model, ByteTokenizer(vocab_size=config.vocab_size)


def generate_text(
    model: CognitionSLM,
    tokenizer: ByteTokenizer,
    prompt: str,
    *,
    task_type: str = "code_generation",
    max_new_tokens: int = 96,
    temperature: float = 0.8,
    top_k: int = 40,
) -> str:
    record = validate_record(
        {
            "id": "generation",
            "prompt": prompt,
            "answer": "placeholder",
            "task_type": task_type,
            "confidence": 0.5,
            "error_category": "none",
            "source": "runtime",
            "license": "runtime",
        }
    )
    prompt_ids = tokenizer.encode(format_prompt(record), add_eos=False)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=next(model.parameters()).device)
    generated = generate_ids(
        model,
        input_ids,
        tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )
    return tokenizer.decode(generated[0, input_ids.size(1) :].tolist()).strip()


def _device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--task-type", default="code_generation")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = _device(args.device)
    model, tokenizer = load_checkpoint(args.checkpoint, device)
    print(
        generate_text(
            model,
            tokenizer,
            args.prompt,
            task_type=args.task_type,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
    )


if __name__ == "__main__":
    main()
