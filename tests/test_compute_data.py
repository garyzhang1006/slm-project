import hashlib
import json
import operator
import re
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from cognition_slm.audit import _prompt_key
from cognition_slm.data import load_pretrain_text, validate_record

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compute import short_facts  # noqa: E402
from compute import stage1_corpus as corpus  # noqa: E402
from compute import stage3_sft_data as sft  # noqa: E402

HOLDOUT_PROMPTS = [row["prompt"] for row in
                   json.loads((ROOT / "data/simple_questions_holdout.json").read_text())["rows"]]
ENGLISH = ("The river runs past the old mill and the children like to watch it in the spring. "
           "They say that it is the best place in the town to see the birds when they return. ") * 3


class HoldoutOverlapTests(unittest.TestCase):
    def test_stems_drop_shared_instructions_and_short_sentences(self):
        stems = corpus.holdout_stems(HOLDOUT_PROMPTS)
        self.assertIn("what is 4 plus 9", stems)
        self.assertNotIn("reply with the number", stems)
        self.assertNotIn("what is it", stems)

    def test_overlap_matches_whole_words_ignoring_case_and_punctuation(self):
        stems = corpus.holdout_stems(HOLDOUT_PROMPTS)
        self.assertTrue(corpus.overlaps_holdout("Quiz: WHAT is 4 plus 9?!", stems))
        self.assertFalse(corpus.overlaps_holdout("What is 4 plus 95?", stems))
        self.assertFalse(corpus.overlaps_holdout("What is 5 plus 9?", stems))


class CorpusHelperTests(unittest.TestCase):
    def test_digest_ignores_case_and_whitespace(self):
        self.assertEqual(corpus.document_digest("Hello  World\n"), corpus.document_digest("hello world"))
        self.assertNotEqual(corpus.document_digest("hello world"), corpus.document_digest("hello worlds"))

    def test_eval_split_is_about_half_a_percent(self):
        chosen = sum(corpus.is_eval_document(corpus.document_digest(f"doc {index}"))
                     for index in range(40_000))
        self.assertTrue(120 <= chosen <= 280, chosen)

    def test_english_heuristic(self):
        self.assertTrue(corpus.looks_english(ENGLISH))
        self.assertFalse(corpus.looks_english("Der Hund läuft über die Straße und bellt laut. " * 10))
        self.assertFalse(corpus.looks_english("x1 = 4; y2 = 7; z3 = x1 + y2; " * 20))

    def test_check_document_reasons(self):
        stems = corpus.holdout_stems(HOLDOUT_PROMPTS)
        fineweb, stories = "HuggingFaceFW/fineweb-edu", "roneneldan/TinyStories"
        self.assertEqual(corpus.check_document({"text": ENGLISH}, stories, stems), (ENGLISH.strip(), "ok"))
        cases = {
            "language_label": ({"text": ENGLISH, "language": "en", "language_score": 0.5}, fineweb),
            "invalid_text": ({"text": ENGLISH + "�"}, stories),
            "too_short": ({"text": "The cat sat."}, stories),
            "secret_pattern": ({"text": ENGLISH + " AKIA" + "A" * 16}, stories),
            "holdout_overlap": ({"text": ENGLISH + " How many days are in one week?"}, stories),
        }
        for reason, (raw, source) in cases.items():
            with self.subTest(reason=reason):
                self.assertEqual(corpus.check_document(raw, source, stems), (None, reason))

    def test_next_source_follows_shares(self):
        shares = {"a": 0.85, "b": 0.15}
        self.assertEqual(corpus.next_source({"a": 0, "b": 0}, shares), "a")
        self.assertEqual(corpus.next_source({"a": 900, "b": 100}, shares), "b")
        self.assertEqual(corpus.next_source({"a": 800, "b": 200}, shares), "a")

    def test_resilient_rows_resumes_after_consumed_rows(self):
        calls = []

        def open_stream(skip):
            calls.append(skip)
            for value in range(skip, 5):
                if value == 2 and len(calls) == 1:
                    raise ConnectionError("dropped")
                yield value

        self.assertEqual(list(corpus.resilient_rows(open_stream, wait_seconds=0)), [0, 1, 2, 3, 4])
        self.assertEqual(calls, [0, 2])

    def test_resilient_rows_gives_up(self):
        def broken(skip):
            raise ConnectionError("down")

        with self.assertRaisesRegex(RuntimeError, "failed 3 times"):
            list(corpus.resilient_rows(broken, retries=2, wait_seconds=0))

    def test_resilient_rows_only_counts_errors_without_progress(self):
        # A multi-hour stream sees scattered blips; each reopen that yields rows resets the retry budget.
        def flaky(skip):
            if skip >= 12:
                return
            yield from range(skip, skip + 2)
            raise ConnectionError("blip")

        self.assertEqual(list(corpus.resilient_rows(flaky, retries=2, wait_seconds=0)), list(range(12)))

    def test_writer_dedupes_caps_eval_and_writes_loadable_jsonl(self):
        stems = corpus.holdout_stems(HOLDOUT_PROMPTS)
        source = "roneneldan/TinyStories"
        with tempfile.TemporaryDirectory() as directory:
            writer = corpus.CorpusWriter(Path(directory), target_bytes=10**9, max_eval_bytes=1,
                                         eval_modulus=2)
            outcomes = [writer.add({"text": f"{ENGLISH} Story number {index}."}, source, stems)
                        for index in range(20)]
            self.assertEqual(writer.add({"text": f"{ENGLISH} STORY number 0."}, source, stems), "duplicate")
            writer.close()
            self.assertIn("eval_full", outcomes)
            self.assertEqual(writer.documents["eval"], 0)
            self.assertEqual(len(load_pretrain_text(writer.paths["train"])), outcomes.count("train"))
            first = json.loads(writer.paths["train"].read_text().splitlines()[0])
            self.assertEqual(set(first), {"text", "source"})
            stats = writer.per_source[source]
            self.assertEqual(stats["scanned"], 21)
            self.assertEqual(stats["train_bytes"], writer.bytes["train"])
            self.assertFalse(writer.done)


class ShortFactTests(unittest.TestCase):
    def test_rows_are_unique_valid_and_numerous(self):
        rows = short_facts.short_fact_rows()
        self.assertGreater(len(rows), 3000)
        self.assertEqual(len({" ".join(row["prompt"].casefold().split()) for row in rows}), len(rows))
        self.assertEqual({row["category"] for row in rows},
                         {"math", "instruction", "fact", "english", "unknown", "code"})
        self.assertEqual(len(sft.short_fact_records()), len(rows))

    def test_holdout_items_are_left_out(self):
        prompts = {row["prompt"] for row in short_facts.short_fact_rows()}
        for prompt in ("What is 4 plus 9? Reply with the number.", "What is 4 + 9?", "Add 9 and 4.",
                       "What is 3 x 4?", "Divide 18 by 3.", "Subtract 6 from 15.",
                       "What is the opposite of tall?", "What is the plural of book?",
                       "Write only the word window.", "What is 2 x 6?", "What is 6 times 2? Reply with the number.",
                       "What is double 6?", "What is 4 - 1?",
                       # The same facts in other forms: 4 plus 9 as a story, four oranges less one as a neighbor.
                       "Beth has 4 stickers and finds 9 more. How many stickers does Beth have now?",
                       "Beth has 9 stickers and finds 4 more. How many stickers does Beth have now?",
                       "Beth had 4 cookies and lost 1 of them. How many cookies are left?",
                       "What number comes just before 4?", "What number comes right after 3?"):
            self.assertNotIn(prompt, prompts)
        self.assertIn("What is 5 + 9?", prompts)
        stems = corpus.holdout_stems(HOLDOUT_PROMPTS)
        normalized = [corpus.normalize_overlap(prompt) for prompt in HOLDOUT_PROMPTS]
        self.assertFalse([record["prompt"] for _, record in sft.short_fact_records()
                          if sft.holdout_conflict(record, stems, normalized)])

    def test_everyday_eval_facts_are_left_out(self):
        rows = short_facts.short_fact_rows()
        prompts = {row["prompt"] for row in rows}
        for prompt in ("What is 72 / 8?", "Divide 20 by 4.", "What is 42 divided by 7? Reply with the number.",
                       "Multiply 5 by 6.", "Is 3 bigger than 2? Answer yes or no.", "What day comes after Friday?",
                       "Which day is right before Saturday? Reply with one word.", "What day comes before Wednesday?",
                       "What month comes after March?", "What month comes before May?",
                       "What is the last month of the year?", "Which days make up the weekend?",
                       "Yes or no: is ice cream warmer than hot tea?",
                       # The everyday rows ask which star is closest and which planet the moon travels around.
                       "What bright star do we see in the daytime sky?", "What orbits Earth and shines at night?"):
            self.assertNotIn(prompt, prompts)
        self.assertIn("What day comes after Thursday?", prompts)
        # The sums the reading passages ask stay out of stories and plain arithmetic alike.
        for prompt in ("Alice has 6 stickers and finds 2 more. How many stickers does Alice have now?",
                       "Diego had 10 cookies and lost 4 of them. How many cookies are left?",
                       "What is 6 + 2?", "Add 2 and 6.", "What is 10 - 4?", "What is 12 - 12?", "What is 2 x 4?",
                       "What is double 4?", "What is double 16?", "What number comes just before 3?"):
            self.assertNotIn(prompt, prompts)
        self.assertIn("Ivan has 6 stickers and finds 3 more. How many stickers does Ivan have now?", prompts)

    def test_stage3_screen_keeps_every_project_row(self):
        # Stage 3 drops rows sharing a sentence with either eval file; a dropped project row is wasted work.
        everyday = [row["prompt"] for row in json.loads((ROOT / "data/everyday_eval.json").read_text())["rows"]]
        screened = HOLDOUT_PROMPTS + everyday
        stems = corpus.holdout_stems(screened)
        normalized = [corpus.normalize_overlap(prompt) for prompt in screened]
        self.assertEqual([record["prompt"] for _, record in sft.short_fact_records()
                          if sft.holdout_conflict(record, stems, normalized)], [])

    def test_arithmetic_answers_are_correct(self):
        operations = {"plus": operator.add, "minus": operator.sub, "times": operator.mul,
                      "divided by": operator.floordiv}
        checked = 0
        for row in short_facts.arithmetic_rows():
            prefix = "What is "
            if not (row["prompt"].startswith(prefix) and row["prompt"].endswith("Reply with the number.")):
                continue
            expression = row["prompt"][len(prefix):row["prompt"].index("?")]
            for name, function in operations.items():
                left, separator, right = expression.partition(f" {name} ")
                if separator:
                    self.assertEqual(str(function(int(left), int(right))), row["answer"], row["prompt"])
                    checked += 1
        self.assertGreater(checked, 400)

    def test_added_topics_are_correct(self):
        self.assertGreater(len(short_facts.short_fact_rows()), 8000)
        self.assertEqual([short_facts.number_in_words(n) for n in (0, 13, 40, 42, 100)],
                         ["zero", "thirteen", "forty", "forty-two", "one hundred"])
        answers = {row["prompt"]: row["answer"] for row in short_facts.short_fact_rows()}
        self.assertEqual(answers["It is 11 o'clock now. What time will it be in 3 hours?"], "2 o'clock")
        self.assertEqual(answers["It is 2 o'clock now. What time was it 4 hours ago?"], "10 o'clock")
        self.assertEqual(answers["What is half of 34?"], "17")
        self.assertEqual(answers["Is 57 odd or even?"], "odd")
        self.assertEqual(answers["Kyiv is the capital of which country?"], "Ukraine")
        self.assertEqual(answers["Should you say a hour or an hour?"], "an hour")
        self.assertEqual(answers["Is a mango a kind of fruit or a kind of tool?"], "fruit")
        # A kiwi is also a bird, so "Is a kiwi an animal?" -> "no" taught a false statement.
        self.assertFalse([prompt for prompt in answers if "a kiwi" in prompt and "animal" in prompt])

    def test_bare_questions_get_short_answers_and_sentences_are_asked_for(self):
        answers = {row["prompt"]: row["answer"] for row in short_facts.short_fact_rows()}
        self.assertEqual(answers["What color is snow?"], "white")
        self.assertEqual(answers["Question: How many legs does a spider have?"], "8")
        self.assertEqual(answers["Which city is the capital of Poland?"], "Warsaw")
        self.assertEqual(answers["What color is snow? Answer in a full sentence."], "Snow is white.")
        sentences = {sentence for _, sentence, _ in short_facts.FACTS}
        sentences |= {f"{city} is the capital of {country}." for country, city in short_facts.CAPITALS}
        asked = [prompt for prompt, answer in answers.items() if answer in sentences]
        self.assertTrue(asked)
        self.assertTrue(all(prompt.endswith(short_facts.SENTENCE_CUE) for prompt in asked))
        # Only the phrasing each sentence answers gets it.
        self.assertNotIn("A triangle has how many corners? Answer in a full sentence.", answers)
        self.assertEqual(answers["How many sides does a triangle have? Answer in a full sentence."],
                         "A triangle has 3 sides.")

    def test_yes_no_rows_are_balanced_and_all_kept(self):
        rows = short_facts.yes_no_rows()
        answers = [row["answer"] for row in rows]
        self.assertEqual(answers.count("yes"), answers.count("no"))
        self.assertEqual(set(answers), {"yes", "no"})
        kept = {row["prompt"] for row in short_facts.short_fact_rows()}
        self.assertEqual([row["prompt"] for row in rows if row["prompt"] not in kept], [])
        answer = {row["prompt"]: row["answer"] for row in rows}
        self.assertEqual(answer["Yes or no: is a bat a mammal?"], "yes")
        self.assertEqual(answer["Is a spider an insect? Answer yes or no."], "no")
        self.assertEqual(answer["Is a scarf a piece of clothing?"], "yes")
        self.assertEqual(answer["Is a hammer a vehicle?"], "no")

    def test_reading_rows_answer_from_the_passage_or_say_it_is_missing(self):
        rows = short_facts.reading_rows()
        kept = {row["prompt"] for row in short_facts.short_fact_rows()}
        self.assertEqual([row["prompt"] for row in rows if row["prompt"] not in kept], [])
        everyday = json.loads((ROOT / "data/everyday_eval.json").read_text())["rows"]
        eval_names = set(re.findall(r"[A-Z][a-z]+", " ".join(row["prompt"] for row in everyday)
                                    + " ".join(HOLDOUT_PROMPTS)))
        self.assertFalse(set(short_facts.READING_NAMES) & eval_names)
        # The everyday reading rows use these prefixes; keeping them out keeps that category a transfer test.
        eval_prefixes = ("Read this:", "Passage:", "Story:")
        self.assertFalse([row["prompt"] for row in rows if row["prompt"].startswith(eval_prefixes)])
        categories = {}
        for row in rows:
            categories.setdefault(row["group"], set()).add(row["category"])
            if row["category"] == "unknown":
                self.assertEqual(row["answer"], short_facts.NOT_STATED)
                continue
            if row["answer"] in ("yes", "no"):
                continue
            answer = row["answer"].removeprefix("because ").removeprefix("in the ").removesuffix(" o'clock")
            spellings = (answer, short_facts.NUMBER_WORDS[int(answer)]) if answer.isdigit() else (answer,)
            self.assertTrue(any(spelling in row["prompt"] for spelling in spellings), row)
        # Every passage that leaves a detail out has a twin that states it.
        self.assertTrue(all("english" in found for found in categories.values()))
        self.assertGreater(sum("unknown" in found for found in categories.values()), 20)

    def test_nature_rows_are_kept_with_wrapped_and_one_word_variants(self):
        rows = short_facts.short_fact_rows()
        answer = {row["prompt"]: row["answer"] for row in rows}
        self.assertEqual([row["prompt"] for row in short_facts.nature_rows() if row["prompt"] not in answer], [])
        self.assertEqual(answer["Which planet is famous for its bright rings?"], "Saturn")
        self.assertEqual(answer["Question: What do we call a baby goat? Reply with one word."], "kid")
        self.assertEqual(answer["Please answer: What happens to butter when it gets hot?"], "it melts")
        self.assertNotIn("What happens to butter when it gets hot? Reply with one word.", answer)

    def test_plurals_are_unique_and_avoid_the_eval_nouns(self):
        singulars = [singular for singular, _ in short_facts.PLURALS]
        self.assertEqual(len(singulars), len(set(singulars)))
        everyday = json.loads((ROOT / "data/everyday_eval.json").read_text())["rows"]
        eval_nouns = {re.findall(r"[a-z]+", row["prompt"])[-1] for row in everyday if row["category"] == "plurals"}
        self.assertEqual(len(eval_nouns), 22)
        self.assertFalse(set(singulars) & eval_nouns)
        answer = {row["prompt"]: row["answer"] for row in short_facts.short_fact_rows()}
        self.assertEqual(answer["What do you call more than one elf?"], "elves")
        self.assertEqual(answer["Spell the plural of the word roof."], "roofs")
        self.assertEqual(answer["What is the singular of fungi?"], "fungus")

    def test_inverted_phrasings_ask_for_the_word_not_a_sentence(self):
        # The 2026-09-29 1.7B adapter answered "Add is the opposite of which word?" with "was" and
        # "How do you say more than one shelf?" with "more than one shelf".
        answer = {row["prompt"]: row["answer"] for row in short_facts.short_fact_rows()}
        self.assertEqual(answer["Loud is the opposite of what?"], "quiet")
        self.assertEqual(answer["Quiet is the opposite of what?"], "loud")
        self.assertEqual(answer["How would you say more than one wolf?"], "wolves")

    def test_both_directions_of_an_opposite_share_a_split_group(self):
        # Separate groups let split_rows hold out "opposite of quiet" while "opposite of loud" trains.
        group = {row["prompt"]: row["group"] for row in short_facts.short_fact_rows()}
        self.assertEqual(group["Loud is the opposite of what?"], group["Quiet is the opposite of what?"])

    def test_both_directions_of_a_neighbor_fact_share_a_split_group(self):
        # "after Monday" and "before Tuesday" state one fact; separate groups let the eval split hold one out.
        group = {row["prompt"]: row["group"] for row in short_facts.short_fact_rows()}
        for forward, backward in (
            ("What day comes after Thursday?", "What day comes before Friday?"),
            ("What day comes after Thursday?", "Today is Friday. What day was it yesterday?"),
            ("What month comes after June?", "Which month is right before July? Reply with one word."),
            ("What number comes right after 41?", "What number comes just before 42?"),
            ("Which letter comes after K in the alphabet?", "Which letter comes before L in the alphabet?"),
            ("It is 3 o'clock now. What time will it be in 4 hours?",
             "It is 7 o'clock now. What time was it 4 hours ago?"),
        ):
            with self.subTest(forward=forward):
                self.assertEqual(group[forward], group[backward])
        self.assertNotEqual(group["What day comes after Thursday?"], group["What day comes before Thursday?"])

    def test_both_orders_of_a_sum_product_or_comparison_share_a_split_group(self):
        group = {row["prompt"]: row["group"] for row in short_facts.short_fact_rows()}
        for first, second in (("What is 2 + 5?", "What is 5 + 2?"), ("What is 4 x 6?", "What is 6 x 4?"),
                              ("Which is larger, 1 or 4?", "Which is larger, 4 or 1?"),
                              ("Is 1 bigger than 4? Answer yes or no.", "Is 4 bigger than 1? Answer yes or no.")):
            with self.subTest(first=first):
                self.assertEqual(group[first], group[second])
        # Subtraction keeps its order: 7 - 2 and 7 - 5 are different facts.
        self.assertNotEqual(group["What is 7 - 2?"], group["What is 7 - 5?"])

    def test_counting_rows_skip_the_eval_neighbors(self):
        answer = {row["prompt"]: row["answer"] for row in short_facts.short_fact_rows()}
        self.assertEqual(answer["Today is Thursday. What day will it be the day after tomorrow?"], "Saturday")
        self.assertEqual(answer["Today is Monday. What day was it two days ago?"], "Saturday")
        self.assertEqual(answer["Question: How many days are in September?"], "30")
        self.assertEqual(answer["How many hours are in three days? Reply with the number."], "72")
        for prompt in ("Today is Friday. What day is tomorrow?", "Today is Wednesday. What day was it yesterday?",
                       "How many days are in April?", "How many days are in February?"):
            self.assertNotIn(prompt, answer)

    def test_sort_and_code_rows_are_correct_and_avoid_holdout_words(self):
        for row in short_facts.sort_rows():
            listed = row["prompt"].split(": ", 1)[-1].rstrip(".") if ":" in row["prompt"] else None
            if listed:
                self.assertEqual(", ".join(sorted(listed.split(", "))), row["answer"], row["prompt"])
                self.assertNotEqual(listed, row["answer"], row["prompt"])
        for row in short_facts.sort_rows() + short_facts.code_rows():
            words = set(re.findall(r"[a-z]+", row["prompt"].casefold()))
            self.assertFalse(words & short_facts.HOLDOUT_SORT_WORDS, row["prompt"])
        answers = {row["prompt"]: row["answer"] for row in short_facts.short_fact_rows()}
        self.assertEqual(answers['What is len("tiger") in Python?'], "5")
        self.assertEqual(answers["What is len([2, 3, 4]) in Python?"], "3")
        self.assertEqual(answers['In Python, what does "plum".upper() give?'], '"PLUM"')


class SftDataTests(unittest.TestCase):
    def test_record_budget_and_schema(self):
        record = sft.sft_record("r1", " Name a color. ", "Blue.", "s", "CC0-1.0")
        self.assertEqual(validate_record(record).prompt, "Name a color.")
        self.assertIsNone(sft.sft_record("r2", "Tell me.", "x" * 2000, "s", "CC0-1.0"))
        self.assertIsNone(sft.sft_record("r3", "   ", "Blue.", "s", "CC0-1.0"))

    def test_dolly_keeps_short_responses_with_context(self):
        rows = sft.dolly_rows([
            {"instruction": "Who wrote it?", "context": "Ann wrote the poem.", "response": "Ann."},
            {"instruction": "Explain.", "context": "", "response": "word " * 200},
            {"instruction": "", "context": "", "response": "Empty."},
        ])
        self.assertEqual([group for group, _ in rows], ["dolly:0"])
        self.assertIn("Context:\nAnn wrote the poem.", rows[0][1]["prompt"])
        self.assertEqual(rows[0][1]["license"], "CC-BY-SA-3.0")

    def test_dolly_one_sentence_answers_to_bare_questions_carry_the_sentence_cue(self):
        rows = dict(sft.dolly_rows([
            {"instruction": "Which planet is the hottest?", "context": "",
             "response": "Venus is the hottest planet."},
            {"instruction": "Why is the sky blue?", "context": "",
             "response": "Air scatters blue light the most. So the sky looks blue."},
            {"instruction": "Who won?", "context": "Mo won the race.", "response": "Mo won the race."},
            {"instruction": "What is the capital of Peru?", "context": "", "response": "Lima."},
            {"instruction": "Describe a cat.", "context": "", "response": "A cat is a small furry pet."},
        ]))
        cue = short_facts.SENTENCE_CUE
        self.assertEqual(rows["dolly:0"]["prompt"], "Which planet is the hottest?" + cue)
        for group in ("dolly:1", "dolly:2", "dolly:3", "dolly:4"):
            self.assertNotIn(cue, rows[group]["prompt"], group)

    def test_oasst_takes_best_ranked_english_reply_to_english_root(self):
        def message(message_id, parent, role, lang, text, **extra):
            return dict(message_id=message_id, parent_id=parent, role=role, lang=lang, text=text,
                        deleted=False, **extra)

        rows = sft.oasst_pairs([
            message("p1", None, "prompter", "en", "What is a noun?", review_result=True),
            message("a2", "p1", "assistant", "en", "Second best.", rank=1),
            message("a1", "p1", "assistant", "en", "A noun names a person, place or thing.", rank=0,
                    detoxify={"toxicity": 0.01}),
            message("a3", "p1", "assistant", "en", "Unranked.", rank=None),
            message("p2", None, "prompter", "de", "Was ist ein Nomen?", review_result=True),
            message("a4", "p2", "assistant", "de", "Ein Wort.", rank=0),
            message("p3", None, "prompter", "en", "Say something rude.", review_result=True),
            message("a5", "p3", "assistant", "en", "Rude reply.", rank=0, detoxify={"toxicity": 0.9}),
        ])
        self.assertEqual([record["id"] for _, record in rows], ["oasst-a1"])
        self.assertEqual(rows[0][1]["license"], "Apache-2.0")

    def test_holdout_conflicts_in_prompt_answer_or_containment(self):
        stems = corpus.holdout_stems(HOLDOUT_PROMPTS)
        normalized = [corpus.normalize_overlap(prompt) for prompt in HOLDOUT_PROMPTS]

        def conflict(prompt, answer):
            return sft.holdout_conflict({"prompt": prompt, "answer": answer}, stems, normalized)

        self.assertTrue(conflict("Quiz: how many days are in one week?", "Seven."))
        self.assertTrue(conflict("Tell me a fact.", "Which animal says meow? A cat."))
        self.assertTrue(conflict("Nora has a red hat", "Yes."))
        self.assertFalse(conflict("How many days are in two weeks?", "14"))

    def test_filter_drops_rows_holding_secrets(self):
        rows = [("g0", sft.sft_record("r0", "Name a color.", "Blue.", "s", "CC0-1.0")),
                ("g1", sft.sft_record("r1", "What is the key?", "AKIA" + "A" * 16, "s", "CC0-1.0"))]
        dropped = {}
        kept = sft.filter_rows(rows, corpus.holdout_stems(HOLDOUT_PROMPTS), HOLDOUT_PROMPTS, dropped)
        self.assertEqual([record["id"] for _, record in kept], ["r0"])
        self.assertEqual(dropped, {"secret_pattern": 1})

    def test_split_keeps_groups_together_and_caps_eval(self):
        rows = [(f"g{index // 3}", {"id": f"r{index}"}) for index in range(3000)]
        train, evaluation = sft.split_rows(rows, eval_modulus=10, max_eval=60)
        self.assertEqual(len(train) + len(evaluation), 3000)
        self.assertEqual(len(evaluation), 60)
        eval_groups = {int(record["id"][1:]) // 3 for record in evaluation}
        self.assertFalse(eval_groups & {int(record["id"][1:]) // 3 for record in train})

    def test_build_sft_is_disjoint_and_valid(self):
        dolly = [{"instruction": "How many days are in one week?", "context": "", "response": "7"},
                 {"instruction": "What is the capital of France?", "context": "", "response": "Paris, of course."},
                 {"instruction": "Name a warm color.", "context": "", "response": "Orange."}]
        train, evaluation, stats = sft.build_sft(dolly, [], HOLDOUT_PROMPTS)
        self.assertEqual(stats["dropped"], {"holdout_overlap": 1, "duplicate_prompt": 1})
        self.assertTrue(evaluation)
        self.assertFalse({_prompt_key(r["prompt"]) for r in train}
                         & {_prompt_key(r["prompt"]) for r in evaluation})
        ids = [record["id"] for record in train + evaluation]
        self.assertEqual(len(ids), len(set(ids)))
        for record in train + evaluation:
            self.assertEqual(validate_record(record).task_type, "language_generation")
        france = [r for r in train + evaluation if r["prompt"] == "What is the capital of France?"]
        self.assertEqual(france[0]["source"], short_facts.SOURCE)


class RunnerSafetyTests(unittest.TestCase):
    def test_import_loads_no_dataset_or_model_library(self):
        code = ("import sys; import compute.stage1_corpus, compute.stage3_sft_data; "
                "print(sorted(name for name in ('datasets', 'torch', 'transformers') if name in sys.modules))")
        environment = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT / "src"), str(ROOT)]))
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=environment,
                                capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout.strip(), "[]")

    def test_runners_import_only_what_the_kaggle_bundle_ships(self):
        # compute/package.py ships src/cognition_slm, compute/ and scripts/score_holdout.py only,
        # so an import of any other scripts/ module crashes on Kaggle after the data download.
        import ast

        allowed = set(sys.stdlib_module_names) | {"__future__", "datasets", "cognition_slm", "compute"}
        for module in (corpus, sft, short_facts):
            with self.subTest(module=module.__name__):
                tree = ast.parse(Path(module.__file__).read_text())
                names = {alias.name.split(".")[0] for node in ast.walk(tree)
                         if isinstance(node, ast.Import) for alias in node.names}
                names |= {node.module.split(".")[0] for node in ast.walk(tree)
                          if isinstance(node, ast.ImportFrom) and node.module}
                self.assertEqual(names - allowed, set())

    def test_manifest_helpers_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            sft.write_jsonl([{"prompt": "Café?", "answer": "Yes."}], path)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["prompt"], "Café?")
            self.assertEqual(corpus.digest(path), hashlib.sha256(path.read_bytes()).hexdigest())
            corpus.write_json(Path(directory) / "m.json", {"rows": 1})
            self.assertEqual(json.loads((Path(directory) / "m.json").read_text()), {"rows": 1})

    @unittest.skipIf(Path("/kaggle/working").is_dir(), "the off-Kaggle guard only applies locally")
    def test_runners_refuse_to_run_locally(self):
        for module in (corpus, sft):
            with self.subTest(module=module.__name__):
                with self.assertRaisesRegex(RuntimeError, "Kaggle"):
                    module.main([])


if __name__ == "__main__":
    unittest.main()
