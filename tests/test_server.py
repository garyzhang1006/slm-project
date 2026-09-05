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
