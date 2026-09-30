"""Local browser workbench for a saved Cognition SLM checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from .config import TASK_TYPES
from .data import format_prompt, validate_record
from .grounding import MAX_PROMPT_BYTES, MAX_SOURCE_BYTES, source_excerpts

DEFAULT_CHECKPOINT = Path("artifacts/slm-500m-language-quality.pt")
DEFAULT_PARAMETERS = 499_524_075
# The largest search body: 12,000 source and 2,000 question bytes can each grow sixfold when JSON
# escapes control characters as \u00XX, plus the object's own keys and quotes.
MAX_BODY_BYTES = 6 * (MAX_SOURCE_BYTES + MAX_PROMPT_BYTES) + 64


def default_checkpoint() -> Path:
    """Select 500M explicitly; missing weights must never select a smaller model."""
    return DEFAULT_CHECKPOINT


WEB_ROOT = Path(__file__).with_name("web")
STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}


class ModelRuntime:
    """Keep one model in memory and serialize inference requests."""

    # format_prompt wraps the question in template tags, so a question may not contain them.
    allows_template_tags = False

    def __init__(self, checkpoint: Path, device: str = "cpu", expected_parameters: int | None = None) -> None:
        self.checkpoint = checkpoint
        self.expected_parameters = expected_parameters
        self.device = device
        self.state = "loading"
        self.error = None
        self.model = None
        self.tokenizer = None
        self.metadata = {"name": "Cognition SLM", "checkpoint": checkpoint.name}
        self.lock = threading.Lock()

    def load(self) -> None:
        try:
            if not self.checkpoint.is_file():
                raise FileNotFoundError(
                    f"Checkpoint not found: {self.checkpoint}. Restart with --checkpoint PATH."
                )
            # Import and load off the serving thread so the workbench opens immediately.
            import torch

            from .checkpoint import load_checkpoint_payload
            from .generate import _device
            from .model import CognitionSLM
            from .tokenizer import ByteTokenizer

            device = _device(self.device)
            if device.type == "cpu":
                torch.set_num_threads(min(4, torch.get_num_threads()))
            payload, config = load_checkpoint_payload(torch, self.checkpoint, inference_only=True)
            # Training checkpoints include Adam state unused by Studio.
            weights = payload["model_state_dict"]
            metadata = payload.get("metadata", {})
            del payload
            model = CognitionSLM(config)
            parameters = sum(parameter.numel() for parameter in model.parameters())
            if self.expected_parameters is not None and parameters != self.expected_parameters:
                raise ValueError(
                    f"Expected {self.expected_parameters:,} parameters; checkpoint has {parameters:,}. "
                    "Download the 500M checkpoint or select another model with --checkpoint PATH."
                )
            model.load_state_dict(weights)
            del weights
            model.to(device).eval()
            self.model = model
            self.tokenizer = ByteTokenizer(vocab_size=config.vocab_size)
            self.metadata.update(
                parameters=parameters,
                context_window=config.block_size,
                device=str(device),
                architecture=config.architecture,
            )
            step = metadata.get("step") if isinstance(metadata, dict) else None
            if isinstance(step, int):
                self.metadata["training_steps"] = step
            self.state = "ready"
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.state = "error"

    def status(self) -> dict:
        result = {
            "state": self.state,
            "busy": self.lock.locked(),
            "model": dict(self.metadata),
            "task_types": list(TASK_TYPES),
        }
        if self.error:
            result["error"] = self.error
        return result

    def generate(self, request: dict) -> dict:
        import torch

        from .generate import generate_ids, strip_stop_sequence

        options, record = validate_request(request, allow_template_tags=self.allows_template_tags)
        prompt_ids = self.tokenizer.encode(format_prompt(record), add_eos=False)
        budget = len(prompt_ids) + options["max_new_tokens"]
        if budget > self.model.config.block_size:
            raise ValueError(
                f"Formatted prompt uses {len(prompt_ids)} byte tokens; requested output uses "
                f"{options['max_new_tokens']}. Total {budget} exceeds context window "
                f"{self.model.config.block_size}. Shorten the prompt or reduce output length."
            )
        input_ids = torch.tensor(
            [prompt_ids], dtype=torch.long, device=next(self.model.parameters()).device
        )
        started = time.perf_counter()
        output = generate_ids(self.model, input_ids, self.tokenizer, **options)
        new_ids = output[0, len(prompt_ids) :].tolist()
        text = self.tokenizer.decode(new_ids)
        finish_reason = "eos" if new_ids and new_ids[-1] == self.tokenizer.eos_id else "length"
        if finish_reason == "length" and options["stop_sequences"]:
            stripped = strip_stop_sequence(text, options["stop_sequences"])
            if stripped != text:
                text, finish_reason = stripped, "stop"
        return {
            "text": text,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "prompt_tokens": len(prompt_ids),
            "generated_tokens": len(new_ids),
            "finish_reason": finish_reason,
        }


def validate_request(request: dict, allow_template_tags: bool = False) -> tuple[dict, object]:
    if not isinstance(request, dict):
        raise ValueError("Expected a JSON object.")
    allowed = {
        "prompt", "task_type", "max_new_tokens", "temperature", "top_k",
        "top_p", "repetition_penalty", "stop_sequences",
    }
    if set(request) - allowed:
        raise ValueError("Unknown request fields: " + ", ".join(sorted(set(request) - allowed)))
    # Defaults favor short factual answers; they match the Studio controls in index.html, except that
    # Studio also sends a newline stop sequence for language generation and the API sends none.
    options = {
        "max_new_tokens": request.get("max_new_tokens", 64),
        "temperature": request.get("temperature", 0.3),
        "top_k": request.get("top_k", 40),
        "top_p": request.get("top_p", 0.9),
        "repetition_penalty": request.get("repetition_penalty", 1.0),
        "stop_sequences": request.get("stop_sequences"),
    }
    for key, lower, upper in (("max_new_tokens", 1, 512), ("top_k", 0, 259)):
        value = options[key]
        if type(value) is not int or not lower <= value <= upper:
            raise ValueError(f"{key} must be an integer between {lower} and {upper}.")
    temperature = options["temperature"]
    if (
        type(temperature) not in (int, float)
        or not 0 <= temperature <= 2
        or not math.isfinite(temperature)
    ):
        raise ValueError("temperature must be a finite number between 0 and 2.")
    top_p = options["top_p"]
    if type(top_p) not in (int, float) or not 0 < top_p <= 1:
        raise ValueError("top_p must be a number greater than 0 and at most 1.")
    penalty = options["repetition_penalty"]
    if type(penalty) not in (int, float) or not 1 <= penalty <= 2:
        raise ValueError("repetition_penalty must be a number between 1 and 2.")
    stop_sequences = options["stop_sequences"]
    if stop_sequences is not None and (
        type(stop_sequences) is not list
        or not 1 <= len(stop_sequences) <= 4
        or not all(type(item) is str and 1 <= len(item.encode("utf-8")) <= 64 for item in stop_sequences)
    ):
        raise ValueError("stop_sequences must be a list of 1 to 4 strings, each 1 to 64 UTF-8 bytes.")
    record = validate_record({
        "id": "workbench", "prompt": request.get("prompt"), "answer": "placeholder",
        "task_type": request.get("task_type", "language_generation"), "confidence": 0.5,
        "error_category": "none", "source": "runtime", "license": "runtime",
    }, allow_template_tags=allow_template_tags)
    return options, record


def reject_lone_surrogates(request: object) -> None:
    # JSON can escape half of a surrogate pair, such as \ud800, and json.loads keeps it as text that
    # UTF-8 can't encode, so later byte counts would fail with a bare codec error.
    if not isinstance(request, dict):
        return
    for key, value in request.items():
        for item in value if isinstance(value, list) else [value]:
            if isinstance(item, str):
                try:
                    item.encode("utf-8")
                except UnicodeEncodeError:
                    raise ValueError(
                        f"{key} contains an unpaired surrogate escape such as \\ud800, which is not valid text. "
                        "Remove it or send the whole character."
                    ) from None


class WorkbenchServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], runtime: ModelRuntime) -> None:
        self.runtime = runtime
        super().__init__(address, WorkbenchHandler)


class WorkbenchHandler(BaseHTTPRequestHandler):
    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(15)

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, status: int, data: dict) -> None:
        self._send(status, json.dumps(data).encode(), "application/json; charset=utf-8")

    def _local_request(self) -> bool:
        port = self.server.server_address[1]
        hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        if port == 80:
            hosts.update({"127.0.0.1", "localhost"})
        host = self.headers.get("Host")
        origin = self.headers.get("Origin")
        if host not in hosts or (origin is not None and origin != f"http://{host}"):
            self._json(403, {"error": "Only same-origin localhost requests are allowed."})
            return False
        return True

    def do_GET(self) -> None:
        if not self._local_request():
            return
        path = urlsplit(self.path).path
        if path == "/api/status":
            self._json(200, self.server.runtime.status())
        elif path in STATIC_FILES:
            filename, content_type = STATIC_FILES[path]
            try:
                self._send(200, (WEB_ROOT / filename).read_bytes(), content_type)
            except FileNotFoundError:
                self._json(404, {"error": "Workbench asset missing. Reinstall the project."})
        else:
            self._json(404, {"error": "Not found."})

    def do_POST(self) -> None:
        if not self._local_request():
            return
        if self.path not in {"/api/generate", "/api/grounded"}:
            self._json(404, {"error": "Not found."})
            return
        if self.headers.get_content_type() != "application/json":
            self._json(415, {"error": "Content-Type must be application/json."})
            return
        try:
            if self.headers.get("Transfer-Encoding"):
                raise ValueError("Transfer-Encoding is not supported.")
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_BODY_BYTES:
                self._json(413, {"error": f"Request body must contain 1 to {MAX_BODY_BYTES} bytes."})
                return
            body = self.rfile.read(length)
            if len(body) != length:
                raise ValueError("Incomplete request body.")
            request = json.loads(body)
            reject_lone_surrogates(request)
            if self.path == "/api/grounded":
                self._json(200, source_excerpts(request))
                return
            validate_request(request, allow_template_tags=self.server.runtime.allows_template_tags)
        except (ValueError, UnicodeError, TimeoutError, RecursionError) as exc:
            self._json(400, {"error": str(exc)})
            return
        runtime = self.server.runtime
        if runtime.state != "ready":
            self._json(503, {"error": runtime.error or "Model is still loading. Try again shortly."})
            return
        if not runtime.lock.acquire(blocking=False):
            self._json(409, {"error": "Model is generating. Wait for the current response."})
            return
        try:
            result = runtime.generate(request)
            status = 200
        except (ValueError, UnicodeError) as exc:
            status, result = 400, {"error": str(exc)}
        except Exception as exc:
            status, result = 500, {"error": f"Generation failed: {type(exc).__name__}: {exc}"}
        finally:
            runtime.lock.release()
        self._json(status, result)


def main() -> None:
    # launch-studio.command resolves only the full flag spellings against the caller's folder.
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda", "auto"), default="cpu")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--sources-only", action="store_true", help="Serve reference excerpts without loading or running model weights.")
    parser.add_argument("--open", action="store_true", help="Open Studio in the default browser once the server is listening.")
    parser.add_argument("--lora-adapter", type=Path, default=None,
                        help="Serve a SmolLM2 LoRA adapter folder on the base model its base_model.json names, "
                             "SmolLM2-360M-Instruct when it names none (needs transformers and peft), "
                             "or a lora-merged folder (transformers only).")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.lora_adapter is not None:
        if args.checkpoint is not None:
            parser.error("--lora-adapter and --checkpoint select different models; pass one of them")
        if not args.sources_only and not any((args.lora_adapter / name).is_file()
                                             for name in ("adapter_config.json", "config.json")):
            parser.error(f"No adapter_config.json or config.json in {args.lora_adapter}. "
                         "Pass the downloaded artifacts/lora-adapter or artifacts/lora-merged folder.")
        from .lora_runtime import LoraRuntime

        runtime = LoraRuntime(args.lora_adapter, args.device)
    else:
        checkpoint = args.checkpoint or default_checkpoint()
        if not args.sources_only and not checkpoint.is_file():
            parser.error(f"Checkpoint not found: {checkpoint}. Download the Kaggle weights to this path or use --checkpoint PATH.")
        runtime = ModelRuntime(checkpoint, args.device, expected_parameters=DEFAULT_PARAMETERS if args.checkpoint is None else None)
    try:
        server = WorkbenchServer(("127.0.0.1", args.port), runtime)
    except OSError as exc:
        other = args.port + 1 if args.port < 65535 else args.port - 1
        parser.exit(1, f"Cannot start slm studio: {exc}. Try a different port, for example --port {other}.\n")
    if args.sources_only:
        runtime.state = "disabled"
        runtime.error = "Model disabled in --sources-only mode. Source excerpts remain available."
    else:
        threading.Thread(target=runtime.load, daemon=True).start()
    url = f"http://127.0.0.1:{args.port}"
    print(f"slm studio is running at {url}", flush=True)
    # Only after binding succeeded, so a port clash never opens a dead page.
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
