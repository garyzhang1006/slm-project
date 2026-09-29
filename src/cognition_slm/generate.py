"""Autoregressive generation from a saved checkpoint."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from .checkpoint import load_checkpoint_payload
from .code_eval import CODE_TASK_TYPES, python_syntax_valid
from .config import TASK_TYPES
from .data import format_prompt, validate_record
from .model import CognitionSLM, KVCache
from .tokenizer import ByteTokenizer

# Below this, logits / temperature can overflow float32 to inf and softmax returns NaN.
GREEDY_TEMPERATURE_FLOOR = 1e-5


def _next_logits(
    model: CognitionSLM, context: torch.Tensor, cache: KVCache | None, *, use_cache: bool,
) -> tuple[torch.Tensor, KVCache | None]:
    if use_cache and isinstance(model, CognitionSLM):
        if cache is not None and cache[0][0].size(2) < model.config.block_size:
            logits, cache = model.forward_inference(context[:, -1:], cache, last_only=True)
        else:
            # Dropping old keys alone changes hidden states and position indices.
            logits, cache = model.forward_inference(context, last_only=True)
        return logits[:, -1, :], cache
    output = model(context, attention_mask=torch.ones_like(context))
    return output.logits[:, -1, :], None


def _finite_number(value: object) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _stop_token_ids(tokenizer: ByteTokenizer, stop_sequences: list[str] | tuple[str, ...] | None) -> list[list[int]]:
    if stop_sequences is None:
        return []
    if isinstance(stop_sequences, str) or not isinstance(stop_sequences, (list, tuple)):
        raise ValueError("stop_sequences must be a list of non-empty strings")
    if not all(isinstance(item, str) and item for item in stop_sequences):
        raise ValueError("stop_sequences must be a list of non-empty strings")
    return [tokenizer.encode(item, add_bos=False, add_eos=False) for item in stop_sequences]


def strip_stop_sequence(text: str, stop_sequences: list[str] | tuple[str, ...] | None) -> str:
    """Remove the stop sequence that ended generation, if any, from the decoded text."""
    # Longest first: with stops "b" and "ab", text "xab" matched both, and the earliest
    # occurrence ("ab") is the one to remove.
    for item in sorted(stop_sequences or (), key=len, reverse=True):
        if text.endswith(item):
            return text[: -len(item)]
    return text


@torch.no_grad()
def generate_ids(
    model: CognitionSLM,
    input_ids: torch.Tensor,
    tokenizer: ByteTokenizer,
    *,
    max_new_tokens: int = 96,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    stop_sequences: list[str] | tuple[str, ...] | None = None,
    use_cache: bool = True,
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
    if not _finite_number(top_p) or not 0 < top_p <= 1:
        raise ValueError("top_p must be a finite number in (0, 1]")
    if not _finite_number(repetition_penalty) or repetition_penalty <= 0:
        raise ValueError("repetition_penalty must be a finite, positive number")
    stop_ids = _stop_token_ids(tokenizer, stop_sequences)
    model.eval()
    generated = input_ids
    prompt_length = input_ids.size(1)
    finished = torch.zeros((input_ids.size(0), 1), dtype=torch.bool, device=input_ids.device)
    cache = None
    for _ in range(max_new_tokens):
        context = generated[:, -model.config.block_size :]
        next_logits, cache = _next_logits(model, context, cache, use_cache=use_cache)
        next_logits[:, tokenizer.pad_id] = float("-inf")
        next_logits[:, tokenizer.bos_id] = float("-inf")
        if repetition_penalty != 1.0 and generated.size(1) > prompt_length:
            # CTRL-style: shrink positive logits and grow negative ones for tokens this
            # call already emitted; the prompt is exempt so answers may reuse its words.
            previous = generated[:, prompt_length:]
            scores = next_logits.gather(1, previous)
            scores = torch.where(scores > 0, scores / repetition_penalty, scores * repetition_penalty)
            next_logits = next_logits.scatter(1, previous, scores)
        if temperature < GREEDY_TEMPERATURE_FLOOR:
            next_token = next_logits.argmax(dim=-1, keepdim=True)
        else:
            next_logits = next_logits / temperature
            if top_k > 0:
                k = min(top_k, next_logits.size(-1))
                values, _ = torch.topk(next_logits, k)
                cutoff = values[:, [-1]]
                next_logits = next_logits.masked_fill(next_logits < cutoff, float("-inf"))
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, dim=-1, descending=True)
                sorted_probabilities = torch.softmax(sorted_logits, dim=-1)
                # Drop a token once the mass before it already covers top_p; the most
                # likely token always survives because its preceding mass is zero.
                sorted_remove = sorted_probabilities.cumsum(dim=-1) - sorted_probabilities > top_p
                remove = sorted_remove.scatter(1, sorted_indices, sorted_remove)
                next_logits = next_logits.masked_fill(remove, float("-inf"))
            probabilities = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
        next_token = next_token.masked_fill(finished, tokenizer.eos_id)
        finished = finished | next_token.eq(tokenizer.eos_id)
        generated = torch.cat([generated, next_token], dim=1)
        if stop_ids:
            new_ids = generated[:, prompt_length:].tolist()
            for row, row_ids in enumerate(new_ids):
                if not finished[row, 0] and any(
                    len(row_ids) >= len(ids) and row_ids[-len(ids) :] == ids for ids in stop_ids
                ):
                    finished[row, 0] = True
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
    model.eval()
    cache = None
    log_probability = torch.zeros((), device=generated_ids.device)
    generated_count = generated_ids.size(1) - prompt_length
    for position in range(prompt_length, generated_ids.size(1)):
        context = generated_ids[:, max(0, position - model.config.block_size) : position]
        next_logits, cache = _next_logits(model, context, cache, use_cache=True)
        next_log_probabilities = torch.log_softmax(next_logits, dim=-1)
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
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    stop_sequences: list[str] | tuple[str, ...] | None = None,
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
    if len(prompt_ids) >= model.config.block_size:
        raise ValueError(
            f"formatted prompt has {len(prompt_ids)} byte tokens, leaving no room to generate "
            f"within block_size {model.config.block_size}; shorten the prompt"
        )
    # Past block_size the rolling window drops BOS and the task header, a context
    # training never produced, so stop at the window like the server's budget check.
    max_new_tokens = min(max_new_tokens, model.config.block_size - len(prompt_ids))
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=next(model.parameters()).device)
    candidates = [
        generate_ids(
            model,
            input_ids,
            tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stop_sequences=stop_sequences,
        )
        for _ in range(num_candidates)
    ]
    texts = [
        strip_stop_sequence(tokenizer.decode(item[0, input_ids.size(1) :].tolist()), stop_sequences).strip()
        for item in candidates
    ]
    if num_candidates == 1:
        return texts[0]
    scores = [score_generated_ids(model, item, input_ids.size(1)) for item in candidates]
    return texts[rank_candidate_indices(texts, scores, task_type, syntax_bonus)]


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
    parser.add_argument("--task-type", default="language_generation", choices=TASK_TYPES)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--num-candidates", type=int, default=1)
    parser.add_argument("--syntax-bonus", type=float, default=0.5)
    parser.add_argument("--top-p", type=float, default=1.0, help="Nucleus sampling mass; 1.0 disables it.")
    parser.add_argument("--repetition-penalty", type=float, default=1.0,
                        help="CTRL-style penalty on already generated tokens; 1.0 disables it.")
    parser.add_argument("--stop", action="append", default=None, metavar="TEXT",
                        help="Stop when the output ends with TEXT; repeat for several sequences.")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.num_candidates < 1 or args.syntax_bonus < 0:
        parser.error("--num-candidates must be positive and --syntax-bonus must be non-negative")
    if not 0 < args.top_p <= 1 or not 0 < args.repetition_penalty < math.inf:
        parser.error("--top-p must be in (0, 1] and --repetition-penalty must be positive")
    if args.stop is not None and not all(args.stop):
        parser.error("--stop values must be non-empty")
    # Checked before the checkpoint loads, which can take a while, rather than inside generation.
    if not 0 <= args.temperature < math.inf or args.top_k < 0 or args.max_new_tokens < 1:
        parser.error("--temperature must be finite and non-negative, --top-k non-negative and --max-new-tokens positive")
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
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            stop_sequences=args.stop,
        )
    )


if __name__ == "__main__":
    main()
