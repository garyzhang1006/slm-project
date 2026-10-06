"""Studio runtime for SmolLM2-Instruct (360M or 1.7B) plus a LoRA adapter trained by compute/lora_baseline.py."""

from __future__ import annotations

import json
import time
from pathlib import Path

from .server import ModelRuntime, validate_request

# Must match compute/lora_baseline.py: the adapter was trained against this exact base revision and prompt.
BASE_MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
BASE_MODEL_REVISION = "a10cc1512eabd3dde888204e902eca88bddb4951"
SYSTEM_PROMPT = "You are a helpful assistant. Answer in plain English with a short, direct reply."
# compute/lora_baseline.py writes this next to the adapter; older adapters without it are 360M.
BASE_MODEL_FILE = "base_model.json"
# Below this temperature, sampling is replaced with greedy decoding. generate.py has its own, lower floor
# (GREEDY_TEMPERATURE_FLOOR) for the in-house model; the two runtimes do not share a cutoff.
GREEDY_BELOW = 1e-3


class StopInAnswer:
    """Stopping criterion that looks only at generated tokens. transformers' stop_strings also matches text
    that starts in the prompt, which ends in a newline, so a stop such as "\nThe" fired on the first token."""

    def __init__(self, tokenizer, prompt_length: int, stops: list[str]) -> None:
        self.tokenizer, self.prompt_length, self.stops = tokenizer, prompt_length, stops

    def __call__(self, input_ids, scores, **kwargs):
        import torch

        text = self.tokenizer.decode(input_ids[0, self.prompt_length:].tolist(), skip_special_tokens=True)
        return torch.full((input_ids.shape[0],), any(stop in text for stop in self.stops), dtype=torch.bool,
                          device=input_ids.device)


class LoraRuntime(ModelRuntime):
    """Same status, locking and request contract as ModelRuntime, backed by transformers and peft."""

    # The chat template never uses format_prompt's tags, so a question may name them.
    allows_template_tags = True

    def __init__(self, adapter: Path, device: str = "cpu") -> None:
        super().__init__(adapter, device)
        # lora tells the page to start on the greedy Steady preset before the model has finished loading.
        self.metadata = {"name": "SmolLM2-360M + LoRA", "checkpoint": adapter.name, "lora": True}

    def load(self) -> None:
        try:
            merged = not (self.checkpoint / "adapter_config.json").is_file()
            if merged and not (self.checkpoint / "config.json").is_file():
                raise FileNotFoundError(
                    f"No adapter_config.json or config.json in {self.checkpoint}. Download artifacts/lora-adapter "
                    "or artifacts/lora-merged from the slm-lora-baseline Kaggle output and pass that folder."
                )
            model_id, revision = BASE_MODEL_ID, BASE_MODEL_REVISION
            if (self.checkpoint / BASE_MODEL_FILE).is_file():
                recorded = json.loads((self.checkpoint / BASE_MODEL_FILE).read_text())
                model_id, revision = recorded["model_id"], recorded["model_revision"]
            # Merged folders record their base too, so Studio names the model that is actually served. Named before
            # the imports and downloads, so the loading and error states show it too.
            self.metadata["name"] = f"{model_id.rsplit('/', 1)[-1]} + LoRA"
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            from .generate import _device

            device = _device(self.device)
            tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
            # fp32: the first Kaggle run showed fp16 overflowing SmolLM2 activations.
            if merged:
                # lora-merged already has the adapter folded into the weights, so peft is not needed.
                base = model = AutoModelForCausalLM.from_pretrained(self.checkpoint, torch_dtype=torch.float32)
            else:
                from peft import PeftModel

                base = AutoModelForCausalLM.from_pretrained(model_id, revision=revision,
                                                            torch_dtype=torch.float32)
                model = PeftModel.from_pretrained(base, str(self.checkpoint))
            model.to(device)
            model.train(False)
            self.model, self.tokenizer = model, tokenizer
            self.metadata.update(
                parameters=sum(parameter.numel() for parameter in model.parameters()),
                context_window=base.config.max_position_embeddings,
                device=str(device), architecture="llama+lora (merged)" if merged else "llama+lora",
            )
            self.state = "ready"
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.state = "error"

    def prompt_ids(self, prompt: str) -> list[int]:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        # Render to text, then tokenize: apply_chat_template(tokenize=True) changed return type across versions.
        text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return list(self.tokenizer(text, add_special_tokens=False).input_ids)

    def generate(self, request: dict) -> dict:
        import torch

        options, record = validate_request(request, allow_template_tags=self.allows_template_tags)
        # The published adapter scores decoded greedily (compute/lora_baseline.py generate_answers), so a request
        # without a temperature gets greedy here rather than validate_request's 0.3 for the in-house model.
        if "temperature" not in request:
            options["temperature"] = 0
        # The tokenizer reads a literal <|im_end|> in the question as the real marker, which ends the user turn early.
        marker = next((token for token in self.tokenizer.all_special_tokens if token in record.prompt), None)
        if marker:
            raise ValueError(f"The question contains {marker}, which this model reads as a chat control marker. "
                             "Remove it and try again.")
        prompt_ids = self.prompt_ids(record.prompt)
        window = self.metadata.get("context_window") or 0
        if window and len(prompt_ids) + options["max_new_tokens"] > window:
            raise ValueError(
                f"Prompt uses {len(prompt_ids)} tokens; requested output uses {options['max_new_tokens']}. "
                f"Total exceeds context window {window}. Shorten the prompt or reduce output length."
            )
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=next(self.model.parameters()).device)
        sample = options["temperature"] >= GREEDY_BELOW
        settings = {"max_new_tokens": options["max_new_tokens"], "do_sample": sample,
                    "repetition_penalty": options["repetition_penalty"],
                    "pad_token_id": self.tokenizer.pad_token_id, "eos_token_id": self.tokenizer.eos_token_id}
        if sample:
            settings.update(temperature=options["temperature"], top_k=options["top_k"], top_p=options["top_p"])
        if options["stop_sequences"]:
            settings["stopping_criteria"] = [StopInAnswer(self.tokenizer, len(prompt_ids), options["stop_sequences"])]
        started = time.perf_counter()
        with torch.no_grad():
            output = self.model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), **settings)
        new_ids = output[0, len(prompt_ids):].tolist()
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        finish_reason = "eos" if new_ids and new_ids[-1] == self.tokenizer.eos_token_id else "length"
        if finish_reason == "length":
            # A byte-level token can end partway through a character, which SmolLM2's tokenizer decodes as U+FFFD.
            text = text.rstrip("\ufffd")
        if finish_reason == "length" and options["stop_sequences"]:
            # The final token can run past the stop string, so cut at its first occurrence.
            found = [index for index in (text.find(item) for item in options["stop_sequences"]) if index >= 0]
            if found:
                text, finish_reason = text[: min(found)], "stop"
        return {
            "text": text,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "prompt_tokens": len(prompt_ids),
            "generated_tokens": len(new_ids),
            "finish_reason": finish_reason,
        }
