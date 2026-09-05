"""Local browser workbench for a saved Cognition SLM checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from .config import TASK_TYPES
from .data import format_prompt, validate_record

DEFAULT_CHECKPOINT = Path("artifacts/slm-500m-language-quality.pt")
DEFAULT_PARAMETERS = 499_524_075


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
            payload, config = load_checkpoint_payload(torch, self.checkpoint)
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

        from .generate import generate_ids

        options, record = validate_request(request)
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
        return {
            "text": self.tokenizer.decode(new_ids),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "prompt_tokens": len(prompt_ids),
            "generated_tokens": len(new_ids),
            "finish_reason": "eos" if new_ids and new_ids[-1] == self.tokenizer.eos_id else "length",
        }


def validate_request(request: dict) -> tuple[dict, object]:
    if not isinstance(request, dict):
        raise ValueError("Expected a JSON object.")
    allowed = {"prompt", "task_type", "max_new_tokens", "temperature", "top_k"}
    if set(request) - allowed:
        raise ValueError("Unknown request fields: " + ", ".join(sorted(set(request) - allowed)))
    options = {
        "max_new_tokens": request.get("max_new_tokens", 128),
        "temperature": request.get("temperature", 0.8),
        "top_k": request.get("top_k", 40),
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
    record = validate_record({
        "id": "workbench", "prompt": request.get("prompt"), "answer": "placeholder",
        "task_type": request.get("task_type", "language_generation"), "confidence": 0.5,
        "error_category": "none", "source": "runtime", "license": "runtime",
    })
    return options, record


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
        if self.path != "/api/generate":
            self._json(404, {"error": "Not found."})
            return
        if self.headers.get_content_type() != "application/json":
            self._json(415, {"error": "Content-Type must be application/json."})
            return
        try:
            if self.headers.get("Transfer-Encoding"):
                raise ValueError("Transfer-Encoding is not supported.")
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 16_384:
                self._json(413, {"error": "Request body must contain 1 to 16384 bytes."})
                return
            body = self.rfile.read(length)
            if len(body) != length:
                raise ValueError("Incomplete request body.")
            request = json.loads(body)
            validate_request(request)
        except (ValueError, UnicodeError, TimeoutError) as exc:
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda", "auto"), default="cpu")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    checkpoint = args.checkpoint or default_checkpoint()
    if not checkpoint.is_file():
        parser.error(f"Checkpoint not found: {checkpoint}. Download the Kaggle weights to this path or use --checkpoint PATH.")
    runtime = ModelRuntime(checkpoint, args.device, expected_parameters=DEFAULT_PARAMETERS if args.checkpoint is None else None)
    try:
        server = WorkbenchServer(("127.0.0.1", args.port), runtime)
    except OSError as exc:
        parser.exit(1, f"Cannot start workbench: {exc}. Try a different --port.\n")
    threading.Thread(target=runtime.load, daemon=True).start()
    print(f"Cognition workbench: http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
