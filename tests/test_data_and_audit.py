import hashlib
import json
import unittest
from pathlib import Path

from cognition_slm.audit import audit_dataset, audit_split_overlap
from cognition_slm.data import DataValidationError, encode_examples, load_jsonl, validate_record
from cognition_slm.tokenizer import ByteTokenizer


ROOT = Path(__file__).resolve().parents[1]


class DataAndAuditTests(unittest.TestCase):
    def test_demo_data_loads(self):
        train = load_jsonl(ROOT / "data" / "demo.jsonl")
        evaluation = load_jsonl(ROOT / "data" / "eval.jsonl")
        self.assertEqual(len(train), 45)
        self.assertEqual(len(evaluation), 16)
        encoded = encode_examples(train, ByteTokenizer(), block_size=256)
        self.assertTrue(
            all(
                0 <= item["pool_position"] < len(item["input_ids"])
                and item["answer_start"] < len(item["input_ids"])
                for item in encoded)
        )
        self.assertEqual(audit_split_overlap(ROOT / "data" / "demo.jsonl", ROOT / "data" / "eval.jsonl"), [])

        manifest = json.loads((ROOT / "data" / "MANIFEST.json").read_text())
        entries = {entry["path"]: entry for entry in manifest["datasets"]}
        for path, records in (("data/demo.jsonl", train), ("data/eval.jsonl", evaluation)):
            file_path = ROOT / path
            self.assertEqual(entries[path]["records"], len(records))
            self.assertEqual(
                entries[path]["sha256"],
                hashlib.sha256(file_path.read_bytes()).hexdigest(),
            )

    def test_committed_data_passes_audit(self):
        report = audit_dataset(ROOT / "data" / "demo.jsonl")
        self.assertTrue(report.ok, report.errors)
        self.assertEqual(report.warnings, [])

    def test_hidden_and_unknown_fields_are_rejected(self):
        base = {
            "id": "x",
            "prompt": "task",
            "answer": "answer",
            "task_type": "code_generation",
            "confidence": 0.5,
            "error_category": "none",
            "source": "test",
            "license": "CC0-1.0",
        }
        with self.assertRaises(DataValidationError):
            validate_record({**base, "chain_of_thought": "private"})
        with self.assertRaises(DataValidationError):
            validate_record({**base, "unexpected": "field"})

    def test_encoding_rejects_examples_without_answer_tokens(self):
        example = validate_record(
            {
                "id": "long-prompt",
                "prompt": "x" * 100,
                "answer": "return 1",
                "task_type": "code_generation",
                "confidence": 0.5,
                "error_category": "none",
                "source": "test",
                "license": "CC0-1.0",
            }
        )
        with self.assertRaisesRegex(ValueError, "no answer tokens"):
            encode_examples([example], ByteTokenizer(), block_size=16)


if __name__ == "__main__":
    unittest.main()
