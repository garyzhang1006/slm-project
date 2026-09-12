"""Project-authored elementary exercises, grouped before deterministic splitting.

This measures compositional transfer within templates, not general English ability.
No external corpus, benchmark answers, or model-generated labels are used.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections import Counter


SOURCE = "project-authored elementary curriculum"
LICENSE = "CC0-1.0"
NAMES = ("Ada", "Ben", "Cora", "Drew", "Elin", "Finn", "Gia", "Hugo", "Iris", "Jude", "Kira", "Milo")
OBJECTS = ("hat", "bag", "kite", "scarf", "ball", "cup", "box", "pen")
COLORS = ("red", "blue", "white", "black", "orange", "purple")
PLACES = ("shelf", "desk", "bench", "table", "chair", "mat")
WORDS = ("acorn", "beach", "cloud", "drum", "eagle", "field", "grape", "hill", "island", "jacket", "leaf", "moon")


def normalize_prompt(text: str) -> str:
    return " ".join(text.split()).casefold()


def _cases():
    for operation in ("add", "subtract", "multiply", "divide"):
        for a, b in itertools.product(range(1, 25), repeat=2):
            left, right = (a * b, b) if operation == "divide" else (a, b)
            symbol, result = {
                "add": ("+", a + b), "subtract": ("-", a - b),
                "multiply": ("*", a * b), "divide": ("/", a),
            }[operation]
            wording = {"add": "plus", "subtract": "minus", "multiply": "times", "divide": "divided by"}[operation]
            operands = sorted((left, right)) if operation in {"add", "multiply"} else (left, right)
            group = f"arithmetic:{operation}:{operands[0]}:{operands[1]}"
            # Commutative reversals share a split but keep distinct prompt identities.
            variants = (
                (f"Calculate {left} {symbol} {right}. Give only the number.", str(result)),
                (f"What is {left} {symbol} {right}? Answer in one sentence.", f"The answer is {result}."),
                (f"Find the result of {left} {symbol} {right}. Keep your answer short.", str(result)),
                (f"Work out {left} {wording} {right}. Give the result.", str(result)),
            )
            yield "arithmetic", group, variants
    for name, item, color in itertools.product(NAMES, OBJECTS, COLORS):
        fact = f"{name}'s {item} is {color}."
        yield "attribute", f"attribute:{name}:{item}:{color}", (
            (f"{fact} What color is {name}'s {item}?", f"It is {color}."),
            (f"{fact} Name the color of the {item}. Use one word.", color),
            (f"Read: {fact} Tell me the {item}'s color in a full sentence.", f"The {item} is {color}."),
        )
    for name, item, place in itertools.product(NAMES, OBJECTS, PLACES):
        fact = f"{name} put the {item} on the {place}."
        yield "location", f"location:{name}:{item}:{place}", (
            (f"{fact} Where is the {item}?", f"On the {place}."),
            (f"{fact} Name the surface holding the {item}.", place),
            (f"Read: {fact} Where did {name} put the {item}? Use a sentence.", f"{name} put it on the {place}."),
        )
    for giver, receiver in itertools.permutations(NAMES, 2):
        for item in OBJECTS:
            fact = f"{giver} gives a {item} to {receiver}."
            yield "transfer", f"transfer:{giver}:{receiver}:{item}", (
                (f"{fact} Who receives the {item}?", receiver),
                (f"{fact} Who gave the {item}? Answer with a name.", giver),
                (f"{fact} Say who gets the {item} in a sentence.", f"{receiver} gets the {item}."),
            )
    for words in itertools.combinations(WORDS, 3):
        # Every ordering of a word set is held together, across both instruction types.
        group = "words:" + ":".join(words)
        sequence = " ".join((words[2], words[0], words[1]))
        yield "copy", group, (
            (f"Copy these words exactly: {sequence}", sequence),
            (f"Repeat only this phrase: {sequence}", sequence),
            (f"Write the phrase '{sequence}' without quotation marks or extra words.", sequence),
        )
        ordered = ", ".join(words)
        yield "sort", group, (
            (f"Sort these words alphabetically, separated by commas: {sequence}", ordered),
            (f"Put {words[1]}, {words[2]}, {words[0]} in alphabetical order. Use commas.", ordered),
            (f"Alphabetize this list and use comma separators: {words[2]}, {words[1]}, {words[0]}", ordered),
        )
    subjects = [(f"The {noun}", False) for noun in ("boy", "girl", "dog", "cat", "bird", "teacher", "student", "singer", "baker", "driver", "farmer", "runner")]
    subjects += [(subject + "s", True) for subject, _ in subjects]
    predicates = (
        ("is", "are", "ready"), ("was", "were", "here"),
        ("has", "have", "a home"), ("likes", "like", "music"),
        ("needs", "need", "water"), ("wants", "want", "food"),
        ("walks", "walk", "slowly"), ("runs", "run", "quickly"),
        ("stays", "stay", "nearby"), ("sleeps", "sleep", "at night"),
        ("looks", "look", "happy"), ("seems", "seem", "tired"),
    )
    for (subject, plural), (singular_verb, plural_verb, ending) in itertools.product(subjects, predicates):
        verb = plural_verb if plural else singular_verb
        sentence = f"{subject} {verb} {ending}."
        group = f"grammar:{subject}:{singular_verb}:{ending}"
        yield "grammar", group, (
            (f"Choose {singular_verb} or {plural_verb}: {subject} ___ {ending}. Give only the verb.", verb),
            (f"Complete the sentence with {singular_verb} or {plural_verb}: {subject} ___ {ending}. Write the whole sentence.", sentence),
            (f"Which verb fits? {subject} ({singular_verb}/{plural_verb}) {ending}. Answer with the verb.", verb),
        )


def build_curriculum(reserved_prompts: set[str]) -> tuple[list[dict], list[dict], dict]:
    """Return canonical rows and provenance; keep each case entirely in one split."""
    reserved = {normalize_prompt(prompt) for prompt in reserved_prompts}
    rows = {"train": [], "dev": []}
    groups = {"train": set(), "dev": set()}
    counts = {"train": Counter(), "dev": Counter()}
    seen: dict[str, str] = {}
    excluded = 0
    for category, group, variants in _cases():
        digest = hashlib.sha256(group.encode("utf-8")).hexdigest()
        split = "dev" if int(digest[:8], 16) % 20 == 0 else "train"
        for prompt, answer in variants:
            normalized = normalize_prompt(prompt)
            if normalized in reserved:
                excluded += 1
                continue
            if normalized in seen:
                raise ValueError(f"Duplicate curriculum prompt: {prompt!r}")
            seen[normalized] = answer
            prompt_digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]
            row = {
                "id": f"elem-{category}-{digest[:16]}-{prompt_digest}",
                "prompt": prompt, "answer": answer,
                "task_type": "language_generation", "confidence": 1.0,
                "error_category": "none", "source": SOURCE, "license": LICENSE,
            }
            if any(ord(char) < 32 or ord(char) == 127 for char in prompt + answer):
                raise ValueError(f"Control character in {row['id']}")
            # Bound both storage and the byte-token prompt/answer after XML wrapping.
            serialized = json.dumps(row, ensure_ascii=False)
            training_text = f"<task_type>language_generation</task_type>\n<instruction>\n{prompt}\n</instruction>\n<answer>\n{answer}"
            if len(serialized.encode("utf-8")) > 512 or len(training_text.encode("utf-8")) + 2 > 512:
                raise ValueError(f"Curriculum example exceeds 512 bytes: {row['id']}")
            rows[split].append(row)
            groups[split].add(digest[:16])
            counts[split][category] += 1
    manifest = {
        "source": SOURCE, "license": LICENSE, "version": 1,
        "split_policy": "sha256 underlying case modulo 20; bucket 0 is dev; commutative reversals grouped",
        "evaluation_scope": "template compositional transfer; does not establish general English ability",
        "reserved_prompts_excluded": excluded,
        "counts": {split: dict(counts[split]) for split in rows},
        "groups": {split: sorted(groups[split]) for split in rows},
        "sha256": {split: hashlib.sha256(json.dumps(rows[split], sort_keys=True).encode()).hexdigest() for split in rows},
    }
    return rows["train"], rows["dev"], manifest
