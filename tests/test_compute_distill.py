import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compute import distill_data, stage4_sft  # noqa: E402

LONG = "x" * 500


class DistillTests(unittest.TestCase):
    def test_candidates_are_context_free_questions_with_long_human_answers(self):
        raws = [
            {"instruction": "Why is the sky blue?", "context": "", "response": LONG, "category": "open_qa"},
            {"instruction": "Short one?", "context": "", "response": "Yes.", "category": "open_qa"},
            {"instruction": "Summarize this.", "context": "Text", "response": LONG, "category": "general_qa"},
            {"instruction": "Write a poem.", "context": "", "response": LONG, "category": "creative_writing"},
            {"instruction": "Is a whale a fish?", "context": "", "response": LONG, "category": "classification"},
        ]
        chosen = distill_data.candidate_prompts(raws, 10)
        self.assertEqual(sorted(index for index, _ in chosen), [0, 4])
        self.assertEqual(len(distill_data.candidate_prompts(raws, 1)), 1)

    def test_trim_answer_keeps_first_paragraph_and_sentence_ends(self):
        self.assertEqual(distill_data.trim_answer(" Blue light scatters.\n\nMore detail."), "Blue light scatters.")
        long = "First sentence. " + "word " * 100
        self.assertEqual(distill_data.trim_answer(long, limit=60), "First sentence.")
        self.assertEqual(distill_data.trim_answer("word " * 100, limit=60), "")

    def test_build_records_filters_and_counts(self):
        holdout = ["What is 4 plus 9? Reply with the number."]
        stems = distill_data.holdout_stems(holdout)
        prompts = [(1, "Why do cats purr?"), (2, "What is 4 plus 9? Reply with the number."),
                   (3, "Name a color."), (4, "Say hi.")]
        answers = ["Cats purr when they are content.", "13", "", "Привет мир"]
        dropped = {}
        records = distill_data.build_records(prompts, answers, stems, holdout, dropped)
        self.assertEqual([record["id"] for record in records], ["distill-1"])
        self.assertEqual(records[0]["license"], "CC-BY-SA-3.0")
        self.assertEqual(dropped, {"holdout_overlap": 1, "empty_or_invalid": 1, "not_english": 1})

    def test_one_sentence_answers_carry_the_sentence_cue(self):
        prompts = [(1, "Which country has the most people?"), (2, "Which planet is red?")]
        records = distill_data.build_records(prompts, ["India has the most people.", "Mars"], [], [], {})
        self.assertEqual([record["prompt"] for record in records],
                         ["Which country has the most people? Answer in a full sentence.", "Which planet is red?"])

    def test_refuses_to_run_off_kaggle(self):
        if Path("/kaggle/working").is_dir():
            self.skipTest("running on Kaggle")
        with self.assertRaises(RuntimeError):
            distill_data.main([])


class MergeDistillTests(unittest.TestCase):
    def write(self, path, prompts):
        path.write_text("".join(json.dumps({"prompt": prompt, "answer": "a"}) + "\n" for prompt in prompts))
        return path

    def test_merge_skips_train_and_eval_prompts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = self.write(root / "train.jsonl", ["Why is grass green?"])
            evaluation = self.write(root / "eval.jsonl", ["Where do bees live?"])
            distill = self.write(root / "distill.jsonl", ["why is GRASS green?", "Where do bees live?",
                                                          "How do birds fly?", "How do birds fly?"])
            (root / "distill_manifest.json").write_text(json.dumps({"teacher": {"adapter_sha256": "abc"}}))
            output = root / "merged.jsonl"
            stats = stage4_sft.merge_distill(train, evaluation, [distill], output)
            self.assertEqual((stats["added"], stats["skipped_duplicate"]), (1, 3))
            # run_pipeline compares this with the current adapter to decide whether sft is stale.
            self.assertEqual(stats["teacher_adapter_sha256"], "abc")
            prompts = [json.loads(line)["prompt"] for line in output.read_text().splitlines()]
            self.assertEqual(prompts, ["Why is grass green?", "How do birds fly?"])

    def test_no_distill_attached_leaves_train_alone(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = self.write(root / "train.jsonl", ["Q?"])
            stats = stage4_sft.merge_distill(train, train, [], root / "merged.jsonl")
            self.assertEqual(stats, {"attached": False, "added": 0})
            self.assertFalse((root / "merged.jsonl").exists())
            with self.assertRaises(RuntimeError):
                stage4_sft.merge_distill(train, train, [train, train], root / "merged.jsonl")


if __name__ == "__main__":
    unittest.main()
