import http.client
import json
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from cognition_slm.server import DEFAULT_PARAMETERS, ModelRuntime, WorkbenchServer, default_checkpoint, main, validate_request


class RequestValidationTests(unittest.TestCase):
    def test_default_is_500m_even_without_weights(self):
        with patch.object(Path, "is_file", return_value=False):
            self.assertEqual(default_checkpoint(), Path("artifacts/slm-500m-language-quality.pt"))

    def test_default_cli_requires_500m(self):
        with patch("sys.argv", ["studio"]), patch.object(Path, "is_file", return_value=True), \
             patch("cognition_slm.server.ModelRuntime") as runtime, \
             patch("cognition_slm.server.WorkbenchServer") as server, \
             patch("cognition_slm.server.threading.Thread"):
            server.return_value.serve_forever.side_effect = KeyboardInterrupt
            main()
            runtime.assert_called_once_with(default_checkpoint(), "cpu", expected_parameters=DEFAULT_PARAMETERS)

    def test_sources_only_never_loads_weights(self):
        with patch("sys.argv", ["studio", "--sources-only"]), patch.object(Path, "is_file", return_value=False), \
             patch("cognition_slm.server.ModelRuntime") as runtime, \
             patch("cognition_slm.server.WorkbenchServer") as server, \
             patch("cognition_slm.server.threading.Thread") as thread:
            server.return_value.serve_forever.side_effect = KeyboardInterrupt
            main()
            runtime.return_value.load.assert_not_called()
            thread.assert_not_called()

    def test_wrong_size_checkpoint_rejected(self):
        import torch
        import tempfile
        from cognition_slm.config import ModelConfig
        from cognition_slm.model import CognitionSLM

        config = ModelConfig(n_layer=1, n_head=2, n_embd=16, block_size=32)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "small.pt"
            torch.save({"model_config": config.to_dict(), "model_state_dict": CognitionSLM(config).state_dict()}, checkpoint)
            runtime = ModelRuntime(checkpoint, expected_parameters=DEFAULT_PARAMETERS)
            runtime.load()
            self.assertEqual(runtime.state, "error")
            self.assertIn("Expected 499,524,075 parameters", runtime.error)

    def test_invalid_sampling_options(self):
        for key, value in (
            ("temperature", float("nan")), ("temperature", float("inf")),
            ("temperature", 10 ** 500), ("temperature", True),
            ("max_new_tokens", 0), ("max_new_tokens", 513),
            ("max_new_tokens", 1.5), ("top_k", -1), ("top_k", 260),
        ):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_request({"prompt": "hello", key: value})

    def test_defaults_favor_short_factual_answers(self):
        options, _ = validate_request({"prompt": "hello"})
        self.assertEqual(options, {"max_new_tokens": 64, "temperature": 0.3, "top_k": 40, "top_p": 0.9,
                                   "repetition_penalty": 1.0, "stop_sequences": None})

    def test_decoding_fields_accepted(self):
        request = {"prompt": "hello", "top_p": 1, "repetition_penalty": 1.2, "stop_sequences": ["\n", "###"]}
        options, _ = validate_request(request)
        self.assertEqual((options["top_p"], options["repetition_penalty"], options["stop_sequences"]),
                         (1, 1.2, ["\n", "###"]))

    def test_invalid_decoding_fields(self):
        for key, value in (
            ("top_p", 0), ("top_p", 1.01), ("top_p", float("nan")), ("top_p", True), ("top_p", "0.9"),
            ("repetition_penalty", 0.9), ("repetition_penalty", 2.5),
            ("repetition_penalty", float("inf")), ("repetition_penalty", False),
            ("stop_sequences", "\n"), ("stop_sequences", []), ("stop_sequences", [""]),
            ("stop_sequences", [1]), ("stop_sequences", ["a"] * 5), ("stop_sequences", ["x" * 65]),
        ):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_request({"prompt": "hello", key: value})

    def test_invalid_prompt_and_unknown_fields(self):
        for request in ([], {"prompt": " "}, {"prompt": "x", "task_type": "chat"},
                        {"prompt": "x", "checkpoint": "elsewhere"}):
            with self.subTest(request=request), self.assertRaises(ValueError):
                validate_request(request)

    def test_missing_checkpoint_is_visible(self):
        runtime = ModelRuntime(Path("/nonexistent/cognition-checkpoint.pt"))
        runtime.load()
        self.assertEqual(runtime.status()["state"], "error")
        self.assertIn("--checkpoint PATH", runtime.status()["error"])

    def test_formatted_prompt_and_output_budget_rejected(self):
        from cognition_slm.tokenizer import ByteTokenizer

        runtime = ModelRuntime(Path("unused"))
        runtime.tokenizer = ByteTokenizer()
        runtime.model = SimpleNamespace(config=SimpleNamespace(block_size=100))
        with self.assertRaisesRegex(ValueError, "exceeds context window"):
            runtime.generate({"prompt": "hello", "max_new_tokens": 100})

    def test_generation_returns_exact_counts_and_eos(self):
        import torch
        from cognition_slm.tokenizer import ByteTokenizer

        runtime = ModelRuntime(Path("unused"))
        runtime.tokenizer = ByteTokenizer()
        runtime.model = SimpleNamespace(
            config=SimpleNamespace(block_size=2048),
            parameters=lambda: iter([torch.empty(0)]),
        )
        def output(model, ids, tokenizer, **options):
            return torch.cat([ids, torch.tensor([[100, tokenizer.eos_id]])], dim=1)

        with patch("cognition_slm.generate.generate_ids", side_effect=output):
            result = runtime.generate({"prompt": "hello", "max_new_tokens": 2})
        self.assertEqual(result["text"], "a")
        self.assertEqual(result["generated_tokens"], 2)
        self.assertEqual(result["finish_reason"], "eos")
        self.assertGreater(result["prompt_tokens"], len("hello"))

    def _runtime_emitting(self, text):
        import torch
        from cognition_slm.tokenizer import ByteTokenizer

        runtime = ModelRuntime(Path("unused"))
        runtime.tokenizer = ByteTokenizer()
        runtime.model = SimpleNamespace(
            config=SimpleNamespace(block_size=2048),
            parameters=lambda: iter([torch.empty(0)]),
        )
        emitted = runtime.tokenizer.encode(text, add_bos=False, add_eos=False)

        def output(model, ids, tokenizer, **options):
            return torch.cat([ids, torch.tensor([emitted])], dim=1)

        return runtime, output

    def test_stop_sequence_is_stripped_and_reported(self):
        runtime, output = self._runtime_emitting("Paris.\n")
        with patch("cognition_slm.generate.generate_ids", side_effect=output) as generate:
            result = runtime.generate({"prompt": "Capital of France?", "stop_sequences": ["\n"]})
        self.assertEqual(result["text"], "Paris.")
        self.assertEqual(result["finish_reason"], "stop")
        self.assertEqual(result["generated_tokens"], 7)
        self.assertEqual(generate.call_args.kwargs["stop_sequences"], ["\n"])
        self.assertEqual(generate.call_args.kwargs["top_p"], 0.9)

    def test_length_finish_keeps_text_without_stop_match(self):
        runtime, output = self._runtime_emitting("Paris")
        with patch("cognition_slm.generate.generate_ids", side_effect=output):
            result = runtime.generate({"prompt": "Capital of France?", "max_new_tokens": 5,
                                       "stop_sequences": ["\n"]})
        self.assertEqual((result["text"], result["finish_reason"]), ("Paris", "length"))


class StudioAssetTests(unittest.TestCase):
    web = Path(__file__).resolve().parents[1] / "src" / "cognition_slm" / "web"

    def test_decoding_controls_match_server_limits(self):
        html = (self.web / "index.html").read_text()
        for markup in ('id="top-p" type="range" min="0.05" max="1" step="0.05" value="0.9"',
                       'id="repetition-penalty" type="range" min="1" max="2" step="0.05" value="1"',
                       '<textarea id="stop-sequences" rows="2" spellcheck="false" aria-describedby="stop-hint">\\n</textarea>'):
            self.assertIn(markup, html)
        # Default Studio payload for language generation must pass server validation.
        options, _ = validate_request({"prompt": "hello", "top_p": 0.9, "repetition_penalty": 1,
                                       "stop_sequences": ["\n"]})
        self.assertEqual(options["stop_sequences"], ["\n"])

    def test_app_sends_validated_field_names(self):
        script = (self.web / "app.js").read_text()
        for snippet in ('top_p: Number($("top-p").value)',
                        'repetition_penalty: Number($("repetition-penalty").value)',
                        "stop_sequences: stops.length ? stops : undefined",
                        "config.stop_sequences.length <= 4",
                        "encoder.encode(item).length <= 64",
                        "!validK || !validStops ||",
                        '!validStops ? "Stop sequences: use at most 4 entries',
                        'response.finish_reason === "stop" ? "A stop sequence ended the response',
                        '"repetition-penalty", "stop-sequences"]) $(id).addEventListener("input", syncComposer)'):
            self.assertIn(snippet, script)
        # Source-excerpt requests carry only the prompt and reference text.
        self.assertIn("grounded ? { prompt: run.prompt, source_text: run.source_text }", script)
        self.assertIn('$("settings-open").disabled = grounded', script)


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.runtime = ModelRuntime(Path("unused"))
        self.runtime.state = "ready"
        self.runtime.generate = Mock(return_value={"text": "test", "generated_tokens": 4})
        self.server = WorkbenchServer(("127.0.0.1", 0), self.runtime)
        self.worker = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.worker.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.worker.join()

    def request(self, method="POST", path="/api/generate", body=None, headers=None):
        connection = http.client.HTTPConnection(*self.server.server_address, timeout=5)
        try:
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    def generate(self, headers=None):
        return self.request(body=json.dumps({"prompt": "hello"}),
                            headers=headers or {"Content-Type": "application/json"})

    def test_success_and_status(self):
        self.assertEqual(self.generate()[0], 200)
        status, payload = self.request("GET", "/api/status")
        self.assertEqual(status, 200)
        self.assertEqual(payload["state"], "ready")
        self.assertFalse(payload["busy"])

    def test_new_fields_accepted_over_http(self):
        body = json.dumps({"prompt": "hello", "top_p": 0.5, "repetition_penalty": 1.1,
                           "stop_sequences": ["\n"]})
        status, _ = self.request(body=body, headers={"Content-Type": "application/json"})
        self.assertEqual(status, 200)
        self.runtime.generate.assert_called_once()
        status, payload = self.request(body=json.dumps({"prompt": "hello", "top_p": 2}),
                                       headers={"Content-Type": "application/json"})
        self.assertEqual(status, 400)
        self.assertIn("top_p", payload["error"])

    def test_hostile_origin_and_host(self):
        for header in ({"Origin": "https://evil.example"}, {"Host": "evil.example"}):
            with self.subTest(header=header):
                status, _ = self.generate({"Content-Type": "application/json", **header})
                self.assertEqual(status, 403)
        self.runtime.generate.assert_not_called()

    def test_busy_returns_conflict(self):
        with self.runtime.lock:
            self.assertEqual(self.generate()[0], 409)
            self.assertTrue(self.request("GET", "/api/status")[1]["busy"])
        self.runtime.generate.assert_not_called()

    def test_loading_returns_unavailable(self):
        self.runtime.state = "loading"
        self.assertEqual(self.generate()[0], 503)

    def test_grounded_works_without_ready_model(self):
        self.runtime.state = "disabled"
        with self.runtime.lock:
            status, result = self.request(path="/api/grounded", body=json.dumps({
                "prompt": "What is the launch date?", "source_text": "The launch date is Friday."
            }), headers={"Content-Type": "application/json"})
        self.assertEqual(status, 200)
        self.assertEqual(result["text"], "[S1] The launch date is Friday.")
        self.runtime.generate.assert_not_called()

    def test_grounded_invalid_input(self):
        status, _ = self.request(path="/api/grounded", body='{"prompt":"hi"}',
                                 headers={"Content-Type": "application/json"})
        self.assertEqual(status, 400)

    def test_generation_error_releases_lock(self):
        self.runtime.generate.side_effect = RuntimeError("inference failure")
        self.assertEqual(self.generate()[0], 500)
        self.assertFalse(self.runtime.lock.locked())

    def test_body_and_content_type_limits(self):
        self.assertEqual(self.request(body="{}")[0], 415)
        headers = {"Content-Type": "application/json"}
        self.assertEqual(self.request(body="invalid", headers=headers)[0], 400)
        self.assertEqual(self.request(body="x" * 16_385, headers=headers)[0], 413)
        self.runtime.generate.assert_not_called()

    def test_unknown_and_traversal_paths(self):
        for path in ("/api/unknown", "/../server.py", "/%2e%2e/server.py"):
            self.assertEqual(self.request("GET", path)[0], 404)


if __name__ == "__main__":
    unittest.main()
