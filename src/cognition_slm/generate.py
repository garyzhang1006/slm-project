"""Autoregressive generation from a saved checkpoint."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from .checkpoint import load_checkpoint_payload
from .code_eval import CODE_TASK_TYPES, python_syntax_valid
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
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    try:
        valid_temperature = (
            type(temperature) in (int, float)
            and temperature >= 0
            and math.isfinite(temperature)
        )
    except OverflowError:
        valid_temperature = False
    if not valid_temperature:
        raise ValueError("temperature must be a finite, non-negative number")
    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    model.eval()
    generated = input_ids
    finished = torch.zeros((input_ids.size(0), 1), dtype=torch.bool, device=input_ids.device)
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
        next_token = next_token.masked_fill(finished, tokenizer.eos_id)
        finished = finished | next_token.eq(tokenizer.eos_id)
        generated = torch.cat([generated, next_token], dim=1)
        if bool(finished.all()):
            break
    return generated


@torch.no_grad()
def score_generated_ids(
    model: CognitionSLM, generated_ids: torch.Tensor, prompt_length: int
) -> float:
    """Return mean log probability of generated tokens, using rolling context."""
    if generated_ids.ndim != 2 or generated_ids.size(0) != 1:
        raise ValueError("generated_ids must have shape (1, sequence_length)")
    if not 0 < prompt_length < generated_ids.size(1):
        raise ValueError("prompt_length must leave at least one generated token")
    log_probability = torch.zeros((), device=generated_ids.device)
    generated_count = generated_ids.size(1) - prompt_length
    for position in range(prompt_length, generated_ids.size(1)):
        context = generated_ids[:, max(0, position - model.config.block_size) : position]
        output = model(context, attention_mask=torch.ones_like(context))
        next_log_probabilities = torch.log_softmax(output.logits[:, -1, :], dim=-1)
        log_probability = log_probability + next_log_probabilities.gather(
            1, generated_ids[:, position : position + 1]
        ).squeeze()
    return float((log_probability / generated_count).item())


def rank_candidate_indices(
    texts: list[str], model_scores: list[float], task_type: str, syntax_bonus: float
) -> int:
    if not texts or len(texts) != len(model_scores):
        raise ValueError("texts and model_scores must be non-empty and have equal length")
    if syntax_bonus < 0:
        raise ValueError("syntax_bonus must be non-negative")
    ranking_scores = []
    for text, model_score in zip(texts, model_scores):
        bonus = syntax_bonus if task_type in CODE_TASK_TYPES and python_syntax_valid(text) else 0.0
        ranking_scores.append(model_score + bonus)
    return max(range(len(ranking_scores)), key=ranking_scores.__getitem__)


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[CognitionSLM, ByteTokenizer]:
    checkpoint, config = load_checkpoint_payload(torch, path, inference_only=True)
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
    num_candidates: int = 1,
    syntax_bonus: float = 0.5,
) -> str:
    if num_candidates < 1:
        raise ValueError("num_candidates must be positive")
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
    if len(prompt_ids) > model.config.block_size:
        raise ValueError(
            f"formatted prompt has {len(prompt_ids)} byte tokens, exceeding "
            f"block_size {model.config.block_size}; shorten the prompt"
        )
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=next(model.parameters()).device)
    candidates = [
        generate_ids(
            model,
            input_ids,
            tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        for _ in range(num_candidates)
    ]
    if num_candidates == 1:
        selected = candidates[0]
    else:
        texts = [tokenizer.decode(item[0, input_ids.size(1) :].tolist()).strip() for item in candidates]
        scores = [score_generated_ids(model, item, input_ids.size(1)) for item in candidates]
        selected = candidates[rank_candidate_indices(texts, scores, task_type, syntax_bonus)]
    return tokenizer.decode(selected[0, input_ids.size(1) :].tolist()).strip()


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
    parser.add_argument("--num-candidates", type=int, default=1)
    parser.add_argument("--syntax-bonus", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.num_candidates < 1 or args.syntax_bonus < 0:
        parser.error("--num-candidates must be positive and --syntax-bonus must be non-negative")
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
            num_candidates=args.num_candidates,
            syntax_bonus=args.syntax_bonus,
        )
    )


if __name__ == "__main__":
    main()
