import json
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from cognition_slm.lora_runtime import BASE_MODEL_REVISION, SYSTEM_PROMPT, LoraRuntime
from cognition_slm.server import main

EOS = 2
# One token that decodes to "\n  ", like a SmolLM2 newline-plus-indent token.
NEWLINE_SPACES = 1000
PIECES = {NEWLINE_SPACES: "\n  "}


class FakeTokenizer:
    """Character-level stand-in for the SmolLM2 tokenizer; token id is the character code."""
    eos_token_id = EOS
    pad_token_id = EOS
    # The SmolLM2 list, read with transformers 5.0.0.
    all_special_tokens = ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]

    def __init__(self):
        self.messages = None

    def apply_chat_template(self, messages, add_generation_prompt, tokenize):
        self.messages = messages
        # Ends in a newline, like the SmolLM2 generation prompt "<|im_start|>assistant\n".
        return "<" + "|".join(message["content"] for message in messages) + ">\n"

    def __call__(self, text, add_special_tokens):
        return SimpleNamespace(input_ids=[ord(character) for character in text])

    def decode(self, ids, skip_special_tokens):
        return "".join(PIECES.get(value, chr(value)) for value in ids if not (skip_special_tokens and value == EOS))


class FakeModel:
    def __init__(self, emitted):
        self.emitted = emitted
        self.settings = None

    def parameters(self):
        return iter([torch.empty(0)])

    def generate(self, input_ids, attention_mask, **settings):
        # Emits one token at a time and honors stopping_criteria, and stop_strings the way transformers
        # does for one-character tokens: on the tail of the whole sequence, prompt included.
        self.settings = settings
        output = input_ids
        for token in self.emitted:
            output = torch.cat([output, torch.tensor([[token]])], dim=1)
            whole = "".join(chr(value) for value in output[0].tolist())
            if any(whole.endswith(stop) for stop in settings.get("stop_strings", [])):
                break
            if any(bool(criterion(output, None).all()) for criterion in settings.get("stopping_criteria", [])):
                break
        return output


def runtime(emitted, window=4096):
    loaded = LoraRuntime(Path("adapter"))
    loaded.tokenizer = FakeTokenizer()
    loaded.model = FakeModel(emitted)
    loaded.metadata["context_window"] = window
    return loaded


class LoraRuntimeTests(unittest.TestCase):
    def test_revision_matches_training(self):
        source = (Path(__file__).resolve().parents[1] / "compute/lora_baseline.py").read_text()
        self.assertIn(BASE_MODEL_REVISION, source)
        self.assertIn(SYSTEM_PROMPT, source)

    def test_studio_lora_extra_pins_the_training_versions(self):
        root = Path(__file__).resolve().parents[1]
        pins = re.search(r"^PINNED_VERSIONS = (\{[^}]*\})", (root / "compute/lora_baseline.py").read_text(), re.M)
        extra = re.search(r"^lora = \[([^\]]*)\]", (root / "pyproject.toml").read_text(), re.M)
        self.assertEqual(sorted(re.findall(r'"([^"]+)"', extra.group(1))),
                         sorted(f"{name}=={version}" for name, version in json.loads(pins.group(1)).items()))

    def test_question_may_name_the_byte_model_template_tags(self):
        result = runtime([ord("o"), ord("k"), EOS]).generate({"prompt": "What is <answer> in a prompt?"})
        self.assertEqual(result["text"], "ok")

    def test_chat_control_markers_are_refused(self):
        # SmolLM2's tokenizer turns a literal <|im_end|> in the question into the end-of-turn id itself.
        loaded = runtime([EOS])
        with self.assertRaisesRegex(ValueError, re.escape("contains <|im_end|>")):
            loaded.generate({"prompt": "What does <|im_end|> do?"})
        self.assertIsNone(loaded.model.settings)

    def test_greedy_at_zero_temperature_and_eos_finish(self):
        loaded = runtime([ord("h"), ord("i"), EOS])
        result = loaded.generate({"prompt": "Say hi", "temperature": 0, "max_new_tokens": 8})
        self.assertEqual(result["text"], "hi")
        self.assertEqual(result["finish_reason"], "eos")
        self.assertEqual(result["generated_tokens"], 3)
        self.assertFalse(loaded.model.settings["do_sample"])
        self.assertNotIn("temperature", loaded.model.settings)
        self.assertEqual(loaded.tokenizer.messages[0], {"role": "system", "content": SYSTEM_PROMPT})
        self.assertEqual(loaded.tokenizer.messages[1], {"role": "user", "content": "Say hi"})

    def test_omitted_temperature_decodes_greedily_like_the_published_scores(self):
        # validate_request fills in 0.3 for the in-house model; the LoRA scores came from do_sample=False.
        loaded = runtime([ord("o"), ord("k"), EOS])
        self.assertEqual(loaded.generate({"prompt": "Go"})["text"], "ok")
        self.assertFalse(loaded.model.settings["do_sample"])
        self.assertNotIn("temperature", loaded.model.settings)
        # An explicit temperature, including the shared 0.3 default sent by the Balanced preset, still samples.
        loaded.generate({"prompt": "Go", "temperature": 0.3})
        self.assertTrue(loaded.model.settings["do_sample"])
        self.assertEqual(loaded.model.settings["temperature"], 0.3)

    def test_sampling_options_are_forwarded(self):
        loaded = runtime([ord("x")])
        loaded.generate({"prompt": "Go", "temperature": 0.7, "top_k": 20, "top_p": 0.8,
                         "repetition_penalty": 1.2, "max_new_tokens": 1})
        settings = loaded.model.settings
        self.assertTrue(settings["do_sample"])
        self.assertEqual((settings["temperature"], settings["top_k"], settings["top_p"]), (0.7, 20, 0.8))
        self.assertEqual(settings["repetition_penalty"], 1.2)

    def test_stop_sequence_is_passed_and_stripped(self):
        loaded = runtime([ord(character) for character in "yes\n\nmore"])
        result = loaded.generate({"prompt": "Go", "stop_sequences": ["\n\n"], "max_new_tokens": 9})
        self.assertEqual((result["text"], result["finish_reason"], result["generated_tokens"]), ("yes", "stop", 5))

    def test_stop_sequence_only_matches_the_answer(self):
        # The prompt ends in a newline, so matching the whole sequence stopped "\nThe" at the first word.
        loaded = runtime([ord(character) for character in "The sky is blue.\nThe end"])
        result = loaded.generate({"prompt": "Go", "stop_sequences": ["\nThe"], "max_new_tokens": 30})
        self.assertEqual((result["text"], result["finish_reason"]), ("The sky is blue.", "stop"))

    def test_stop_sequence_inside_the_final_token_is_cut(self):
        # A single "\n  " token runs past stop "\n"; a suffix-only strip would keep "yes\n  " and report length.
        loaded = runtime([ord("y"), ord("e"), ord("s"), NEWLINE_SPACES, ord("x")])
        result = loaded.generate({"prompt": "Go", "stop_sequences": ["\n"], "max_new_tokens": 8})
        self.assertEqual((result["text"], result["finish_reason"], result["generated_tokens"]), ("yes", "stop", 4))

    def test_a_character_cut_off_at_the_length_limit_is_dropped(self):
        # SmolLM2's tokenizer decodes 东 plus half of 京, cut off at the length limit, as 东 and U+FFFD.
        result = runtime([ord("东"), 0xFFFD]).generate({"prompt": "Go", "max_new_tokens": 2})
        self.assertEqual((result["text"], result["finish_reason"]), ("东", "length"))

    def test_context_window_is_enforced(self):
        with self.assertRaisesRegex(ValueError, "exceeds context window"):
            runtime([EOS], window=10).generate({"prompt": "hello there", "max_new_tokens": 5})

    def test_load_uses_the_recorded_base_model(self):
        loads = []

        class Loaded(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(2))
                self.config = SimpleNamespace(max_position_embeddings=8192)

        def from_pretrained(name, **options):
            loads.append((str(name), options.get("revision")))
            return Loaded()

        transformers = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda path: FakeTokenizer()),
                                       AutoModelForCausalLM=SimpleNamespace(from_pretrained=from_pretrained))
        peft = SimpleNamespace(PeftModel=SimpleNamespace(from_pretrained=lambda base, path: base))
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict("sys.modules", {"transformers": transformers, "peft": peft}):
            adapter = Path(directory)
            (adapter / "adapter_config.json").write_text("{}")
            (adapter / "base_model.json").write_text(
                '{"model_id": "HuggingFaceTB/SmolLM2-1.7B-Instruct", "model_revision": "abc"}')
            loaded = LoraRuntime(adapter)
            loaded.load()
            (adapter / "base_model.json").unlink()
            older = LoraRuntime(adapter)
            older.load()
        self.assertEqual(loaded.state, "ready", loaded.error)
        self.assertEqual(loads, [("HuggingFaceTB/SmolLM2-1.7B-Instruct", "abc"),
                                 ("HuggingFaceTB/SmolLM2-360M-Instruct", BASE_MODEL_REVISION)])
        self.assertEqual(loaded.metadata["name"], "SmolLM2-1.7B-Instruct + LoRA")

    def test_merged_folder_is_named_after_its_recorded_base(self):
        class Loaded(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(2))
                self.config = SimpleNamespace(max_position_embeddings=8192)

        loads = []
        transformers = SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda path: FakeTokenizer()),
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda name, **options: loads.append(str(name)) or Loaded()))
        with tempfile.TemporaryDirectory() as directory, patch.dict("sys.modules", {"transformers": transformers}):
            merged = Path(directory)
            (merged / "config.json").write_text("{}")
            (merged / "base_model.json").write_text(
                '{"model_id": "HuggingFaceTB/SmolLM2-1.7B-Instruct", "model_revision": "abc"}')
            loaded = LoraRuntime(merged)
            loaded.load()
        self.assertEqual(loaded.state, "ready", loaded.error)
        self.assertEqual(loads, [str(merged)])
        self.assertEqual(loaded.metadata["name"], "SmolLM2-1.7B-Instruct + LoRA")
        self.assertEqual(loaded.metadata["architecture"], "llama+lora (merged)")

    def test_recorded_base_is_named_when_loading_fails(self):
        # Studio shows the model name in the error state, so a 1.7B adapter must not read as 360M there.
        with tempfile.TemporaryDirectory() as directory, patch.dict("sys.modules", {"transformers": None}):
            adapter = Path(directory)
            (adapter / "adapter_config.json").write_text("{}")
            (adapter / "base_model.json").write_text(
                '{"model_id": "HuggingFaceTB/SmolLM2-1.7B-Instruct", "model_revision": "abc"}')
            loaded = LoraRuntime(adapter)
            loaded.load()
        self.assertEqual(loaded.state, "error")
        self.assertIn("transformers", loaded.error)
        self.assertEqual(loaded.metadata["name"], "SmolLM2-1.7B-Instruct + LoRA")

    def test_missing_adapter_is_reported(self):
        loaded = LoraRuntime(Path("/nonexistent/lora-adapter"))
        loaded.load()
        self.assertEqual(loaded.state, "error")
        self.assertIn("adapter_config.json", loaded.error)


class LoraCliTests(unittest.TestCase):
    def run_main(self, argv):
        with patch("sys.argv", ["studio", *argv]), \
             patch("cognition_slm.lora_runtime.LoraRuntime") as lora, \
             patch("cognition_slm.server.ModelRuntime") as plain, \
             patch("cognition_slm.server.WorkbenchServer") as server, \
             patch("cognition_slm.server.threading.Thread"):
            server.return_value.serve_forever.side_effect = KeyboardInterrupt
            main()
        return lora, plain

    def test_adapter_selects_lora_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "adapter_config.json").write_text("{}")
            lora, plain = self.run_main(["--lora-adapter", directory])
        lora.assert_called_once_with(Path(directory), "cpu")
        plain.assert_not_called()

    def test_merged_folder_selects_lora_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "config.json").write_text("{}")
            lora, plain = self.run_main(["--lora-adapter", directory])
        lora.assert_called_once_with(Path(directory), "cpu")
        plain.assert_not_called()

    def test_bad_adapter_arguments_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(SystemExit):
                self.run_main(["--lora-adapter", directory])
            (Path(directory) / "adapter_config.json").write_text("{}")
            with self.assertRaises(SystemExit):
                self.run_main(["--lora-adapter", directory, "--checkpoint", "model.pt"])


if __name__ == "__main__":
    unittest.main()
