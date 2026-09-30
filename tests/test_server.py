import http.client
import json
import re
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from cognition_slm.grounding import MAX_PROMPT_BYTES, MAX_SOURCE_BYTES
from cognition_slm.server import DEFAULT_PARAMETERS, MAX_BODY_BYTES, ModelRuntime, WorkbenchServer, default_checkpoint, main, validate_request


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

    def test_open_flag_opens_the_browser_after_binding(self):
        for argv, opened in ((["studio", "--sources-only", "--open", "--port", "8770"], True),
                             (["studio", "--sources-only", "--port", "8770"], False)):
            with self.subTest(argv=argv), patch("sys.argv", argv), \
                 patch("cognition_slm.server.ModelRuntime"), \
                 patch("cognition_slm.server.WorkbenchServer") as server, \
                 patch("cognition_slm.server.webbrowser.open") as browser:
                server.return_value.serve_forever.side_effect = KeyboardInterrupt
                main()
                self.assertEqual(browser.call_args_list, [(("http://127.0.0.1:8770",),)] if opened else [])

    def test_abbreviated_flags_are_refused(self):
        # launch-studio.command rewrites only the full --checkpoint and --lora-adapter spellings to the
        # caller's folder, so an abbreviation would load a path relative to the project folder instead.
        for argv in (["studio", "--sources"], ["studio", "--sources-only", "--check", "model.pt"],
                     ["studio", "--sources-only", "--lora", "adapter"]):
            with self.subTest(argv=argv), patch("sys.argv", argv), \
                 patch("cognition_slm.server.ModelRuntime") as runtime, \
                 patch("cognition_slm.lora_runtime.LoraRuntime") as lora, \
                 patch("cognition_slm.server.WorkbenchServer") as server, \
                 patch("sys.stderr"), self.assertRaises(SystemExit):
                server.return_value.serve_forever.side_effect = KeyboardInterrupt
                main()
            runtime.assert_not_called()
            lora.assert_not_called()

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

    def test_page_limits_match_the_server(self):
        html = (self.web / "index.html").read_text()
        script = (self.web / "app.js").read_text()
        limits = re.search(r"const LIMITS = \{ source: (\d+), question: (\d+) \};", script)
        self.assertEqual((int(limits.group(1)), int(limits.group(2))), (MAX_SOURCE_BYTES, MAX_PROMPT_BYTES))
        # The page's top-k check, its message and the input box all stop at the largest value the server takes.
        top_k = int(re.search(r"config\.top_k <= (\d+);", script).group(1))
        self.assertIn(f"Top K must be a whole number from 0 to {top_k}.", script)
        self.assertIn(f'id="top-k" type="number" min="0" max="{top_k}"', html)
        max_tokens = int(re.search(r'id="max-tokens" type="range" min="\d+" max="(\d+)"', html).group(1))
        for key, largest in (("top_k", top_k), ("max_new_tokens", max_tokens)):
            with self.subTest(key=key):
                self.assertEqual(validate_request({"prompt": "hello", key: largest})[0][key], largest)
                with self.assertRaises(ValueError):
                    validate_request({"prompt": "hello", key: largest + 1})

    def test_defaults_match_balanced_preset_markup_and_server(self):
        html = (self.web / "index.html").read_text()
        script = (self.web / "app.js").read_text()
        defaults = dict(re.findall(r'"?([a-z-]+)"?: "([^"]*)"', re.search(r"const DEFAULTS = \{([^}]*)\}", script).group(1)))
        balanced = {key: float(value) for key, value in
                    re.findall(r'"?([a-z-]+)"?: ([\d.]+)', re.search(r"^  balanced: \{([^}]*)\}", script, re.M).group(1))}
        # Reset answer settings must land on Balanced, the preset the page starts with.
        self.assertIn('id="preset-balanced" value="balanced" checked', html)
        self.assertEqual({key: float(defaults[key]) for key in balanced}, balanced)
        for element_id in ("max-tokens", "temperature", "top-k", "top-p", "repetition-penalty"):
            with self.subTest(element_id=element_id):
                self.assertRegex(html, rf'id="{element_id}" [^>]*value="{re.escape(defaults[element_id])}"')
        self.assertIn(f'<select id="task-type" aria-describedby="task-hint">\n            <option value="{defaults["task-type"]}">', html)
        # A request that leaves the fields out gets the same values as the page's defaults.
        options, record = validate_request({"prompt": "hello"})
        fields = {"max-tokens": "max_new_tokens", "temperature": "temperature", "top-k": "top_k",
                  "top-p": "top_p", "repetition-penalty": "repetition_penalty"}
        self.assertEqual({field: float(defaults[element_id]) for element_id, field in fields.items()},
                         {field: options[field] for field in fields.values()})
        self.assertEqual(record.task_type, defaults["task-type"])

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

    def test_byte_token_limits_apply_only_to_byte_level_models(self):
        script = (self.web / "app.js").read_text()
        # promptTokens counts bytes, which overcounts a BPE model's tokens, so LoRA models are checked by the server.
        self.assertIn("return customModel() && Boolean(context) && promptTokens(prompt) + config.max_new_tokens > context;", script)
        self.assertIn("[count, context && customModel() ? Math.max(context - config.max_new_tokens, 0) : 0,", script)

    def test_try_again_checks_the_context_window_like_send(self):
        script = (self.web / "app.js").read_text()
        # Try again can follow a raise in Answer length, so it must stop where Send stops, with the same message.
        self.assertIn('const overflow = !grounded && overflows($("prompt").value, config);', script)
        self.assertIn("!run.grounded && overflows(run.prompt, settings()) ? notify(TOO_LONG, true) : ask(run, true)", script)
        self.assertIn("overflow ? TOO_LONG", script)

    def test_notices_clear_when_their_cause_changes(self):
        script = (self.web / "app.js").read_text()
        clear = '$("source-text").addEventListener("input", () => { state.notice = null; });'
        # Registered before syncComposer, so the redraw after a paste no longer shows the old notice.
        self.assertLess(script.index(clear), script.index('for (const id of ["prompt", "source-text"'))
        self.assertIn("if (phase() !== before) state.notice = null;", script)

    def test_unreadable_answer_is_an_error(self):
        script = (self.web / "app.js").read_text()
        # A 200 whose body can't be parsed must not be drawn and saved as an empty answer.
        self.assertIn("const result = await response.json().catch(() => null);", script)
        self.assertIn("if (!result) throw new Error(", script)

    def test_question_is_counted_as_it_will_be_sent(self):
        # A question of control characters alone would count as text, enable Send and go out empty.
        script = (self.web / "app.js").read_text()
        self.assertIn(r'const cleanPrompt = (text) => text.replace(/[\x00-\x08\x0B\x0C\x0E-\x1F]/g, " ").trim();', script)
        self.assertIn("const prompt = cleanPrompt(text);", script)
        self.assertIn('const questionBytes = bytes(cleanPrompt($("prompt").value));', script)
        self.assertIn('runs.push({ prompt: cleanPrompt($("prompt").value), grounded,', script)

    def test_status_poll_cannot_undo_the_busy_flag_an_answer_cleared(self):
        # A poll answered while this page's request held the model arrives after it with busy still true.
        script = (self.web / "app.js").read_text()
        poll = script[script.index("async function pollStatus()"):script.index("function renderStatus()")]
        self.assertIn("const answered = state.answered;", poll)
        self.assertIn("if (state.answered !== answered && status?.busy) status.busy = false;", poll)
        ask = script[script.index("async function ask("):script.index('$("prompt-form").addEventListener')]
        self.assertIn("state.answered += 1;", ask)

    def test_failed_thread_save_drops_the_stale_copy_and_says_so(self):
        script = (self.web / "app.js").read_text()
        # A full sessionStorage keeps the last copy that fit, which a refresh would restore without a word.
        self.assertIn("setItem(key, JSON.stringify(value)); return true; } catch { return false; }", script)
        save = script[script.index("function saveThread()"):script.index("function restoreThread()")]
        self.assertIn('if (writeStore("sessionStorage", "studio-thread", {', save)
        self.assertIn('window.sessionStorage.removeItem("studio-thread")', save)
        self.assertIn("if (!state.unsaved && runs.length) {", save)

    def test_download_escapes_comment_openers_outside_code_fences(self):
        script = (self.web / "app.js").read_text()
        # A line opening <!--, <?, <style and the like with no closing marker hides the rest of the file in a
        # CommonMark viewer, in answers and in quoted passages alike.
        opener = r"const HTML_BLOCK_OPENER = /^( {0,3})(<(?:!--|\?|![A-Za-z]|!\[CDATA\[|(?:script|pre|style|textarea)(?=[\s>]|$)))/i;"
        self.assertIn(opener, script)
        self.assertIn("return open ? line : escapeHtmlBlock(line);", script)
        self.assertIn('source.text.split("\\n").map(escapeHtmlBlock).join("\\n> ")', script)
        self.assertIn("const [plain, dangling] = markdownAnswer(text);", script)
        self.assertIn("answer.code ? [fence, text, fence] : dangling ? [plain, dangling] : [plain]", script)

    def test_unsent_example_leaves_focus_in_the_question_box(self):
        script = (self.web / "app.js").read_text()
        # Filling in the example hides its button, so focus must move on when nothing is sent.
        handler = script[script.index('$("source-example").addEventListener'):]
        self.assertIn('else $("prompt").focus();', handler[:handler.index("});")])

    def test_script_ids_exist_in_page(self):
        html = (self.web / "index.html").read_text()
        script = (self.web / "app.js").read_text()
        # The dialogs are also looked up as $(name) and $(`${name}-open`).
        ids = set(re.findall(r'\$\("([a-z-]+)"\)', script)) | {"settings", "settings-open", "about", "about-open"}
        # Template lookups: $(`theme-${theme}`) at startup, $(`keys-${value}`) and label[for="preset-${preset}"].
        themes = re.findall(r'"([a-z]+)"', re.search(r"const THEMES = \[([^\]]*)\]", script).group(1))
        presets = re.findall(r"^  ([a-z]+): \{ temperature", script, re.M)
        self.assertTrue(themes and presets)
        ids |= {f"theme-{name}" for name in themes} | {f"preset-{name}" for name in presets} | {"keys-on", "keys-off"}
        for element_id in sorted(ids):
            self.assertIn(f'id="{element_id}"', html)
        # Every answer draws icons cloned by name from the page's template.
        for name in sorted(set(re.findall(r'icon(?:Button)?\("([a-z-]+)"', script))):
            self.assertIn(f'data-icon="{name}"', html)

    def test_page_loads_only_same_origin_assets(self):
        html = (self.web / "index.html").read_text()
        # The server's CSP blocks inline scripts and styles, so every asset must be a same-origin file.
        self.assertNotIn("<style", html)
        self.assertNotIn(" style=", html)
        self.assertEqual(re.findall(r"<script[^>]*>", html), ['<script src="/app.js" defer>'])
        self.assertNotRegex(html, r'(src|href)="https?://')


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

    def test_grounded_accepts_the_largest_escaped_body(self):
        # Every byte a control character: the page allows it, and JSON escapes each one to six bytes.
        body = json.dumps({"prompt": "\x01" * 2000, "source_text": "\x01" * 12000})
        status, _ = self.request(path="/api/grounded", body=body, headers={"Content-Type": "application/json"})
        self.assertNotEqual(status, 413)

    def test_lone_surrogate_is_a_clear_bad_request(self):
        # json.dumps escapes the half pair as \ud800, and json.loads on the server keeps it as a lone surrogate.
        for path, request, field in (
            ("/api/generate", {"prompt": "a\ud800"}, "prompt"),
            ("/api/grounded", {"prompt": "a\ud800", "source_text": "text"}, "prompt"),
            ("/api/grounded", {"prompt": "text", "source_text": "a\ud800"}, "source_text"),
        ):
            with self.subTest(path=path, field=field):
                status, payload = self.request(path=path, body=json.dumps(request),
                                               headers={"Content-Type": "application/json"})
                self.assertEqual(status, 400)
                self.assertIn(f"{field} contains an unpaired surrogate", payload["error"])
                self.assertNotIn("codec", payload["error"])
        self.runtime.generate.assert_not_called()

    def test_grounded_invalid_input(self):
        status, _ = self.request(path="/api/grounded", body='{"prompt":"hi"}',
                                 headers={"Content-Type": "application/json"})
        self.assertEqual(status, 400)

    def test_generation_error_releases_lock(self):
        self.runtime.generate.side_effect = RuntimeError("inference failure")
        self.assertEqual(self.generate()[0], 500)
        self.assertFalse(self.runtime.lock.locked())

    def test_runtime_value_error_is_a_bad_request(self):
        # Only the loaded model knows its context window, so this check runs after request validation.
        self.runtime.generate.side_effect = ValueError("Total 2100 exceeds context window 2048. Shorten the prompt or reduce output length.")
        status, payload = self.generate()
        self.assertEqual(status, 400)
        self.assertIn("exceeds context window 2048", payload["error"])
        self.assertFalse(self.runtime.lock.locked())

    def test_body_and_content_type_limits(self):
        self.assertEqual(self.request(body="{}")[0], 415)
        headers = {"Content-Type": "application/json"}
        self.assertEqual(self.request(body="invalid", headers=headers)[0], 400)
        self.assertEqual(self.request(body="x" * (MAX_BODY_BYTES + 1), headers=headers)[0], 413)
        self.runtime.generate.assert_not_called()

    def test_deeply_nested_json_is_rejected(self):
        # Before Python 3.14, json.loads raises RecursionError on deep nesting instead of ValueError.
        headers = {"Content-Type": "application/json"}
        body = "[" * 20000 + "]" * 20000
        self.assertEqual(self.request(body=body, headers=headers)[0], 400)
        parser = SimpleNamespace(loads=Mock(side_effect=RecursionError("maximum recursion depth exceeded")),
                                 dumps=json.dumps)
        with patch("cognition_slm.server.json", parser):
            self.assertEqual(self.request(body=body, headers=headers)[0], 400)
        self.runtime.generate.assert_not_called()

    def test_unknown_and_traversal_paths(self):
        for path in ("/api/unknown", "/../server.py", "/%2e%2e/server.py"):
            self.assertEqual(self.request("GET", path)[0], 404)


if __name__ == "__main__":
    unittest.main()
