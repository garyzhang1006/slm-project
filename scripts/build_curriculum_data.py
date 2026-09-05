"""Build a deterministic curriculum with evaluation prompts held out by wording.

The splits share underlying concepts and answers, so evaluation measures transfer
to different wording rather than acquisition of unseen programming skills.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SOURCE = "project-authored curriculum"
LICENSE = "CC0-1.0"


def _row(record_id: str, prompt: str, answer: str, task_type: str) -> dict[str, object]:
    return {
        "id": record_id,
        "prompt": prompt,
        "answer": answer,
        "task_type": task_type,
        "confidence": 0.9,
        "error_category": "none",
        "source": SOURCE,
        "license": LICENSE,
    }


GREETING_WORDINGS = (
    "Write a warm greeting for {name}.",
    "Give {name} a concise welcome.",
    "Say hello to {name} in one friendly sentence.",
    "Create a welcoming opening addressed to {name}.",
    "Welcome {name} with a positive sentence.",
    "Start a conversation with {name} politely.",
    "Write a brief, kind hello for {name}.",
    "Open a helpful conversation with {name}.",
)

CONCEPTS = (
    ("a loop", "A loop repeats instructions until it reaches a stopping condition."),
    ("a dictionary", "A dictionary stores values under keys so code can look them up by name."),
    ("a function", "A function packages reusable instructions and can receive inputs and return a result."),
    ("a variable", "A variable gives a value a name so a program can use or update it later."),
    ("a list", "A list keeps an ordered collection of values that code can inspect or change."),
    ("an exception", "An exception reports a problem during execution so code can handle or propagate it."),
    ("a test", "A test checks an expected behavior with a known input and result."),
    ("an algorithm", "An algorithm is a finite sequence of steps for solving a problem."),
    ("a database", "A database stores organized information so it can be queried and updated."),
    ("an API", "An API defines how one program requests data or actions from another program."),
    ("a model checkpoint", "A model checkpoint is a saved copy of learned weights and training state."),
    ("a context window", "A context window is the maximum input and output sequence a model can process at once."),
)

GRAMMAR_FIXES = (
    ("She walk to school.", "She walks to school."),
    ("The results was clear.", "The results were clear."),
    ("He have two ideas.", "He has two ideas."),
    ("We was ready to begin.", "We were ready to begin."),
    ("This tools are useful.", "These tools are useful."),
    ("I seen the message yesterday.", "I saw the message yesterday."),
    ("They is testing the model.", "They are testing the model."),
    ("The code need a test.", "The code needs a test."),
    ("Please sends the report.", "Please send the report."),
    ("An useful example helps.", "A useful example helps."),
    ("The answer were concise.", "The answer was concise."),
    ("She can explains the result.", "She can explain the result."),
)

WRITING_SCENARIOS = (
    ("a study group", "Welcome to the study group. Bring one question and one idea to share."),
    ("a project update", "The project is moving forward, and the next step is to review the latest results."),
    ("a teammate about a review", "Could you review the draft by Friday and share any edits?"),
    ("a new user", "Welcome. Start with a small question, then use the result to choose your next experiment."),
    ("a meeting reminder", "A quick reminder that our meeting starts tomorrow at ten."),
    ("a thank-you note", "Thank you for taking the time to review the work and share thoughtful feedback."),
    ("a clear status update", "The data is ready, the first check passed, and the remaining work is evaluation."),
    ("a polite request", "Could you send the file when you have a moment? Thank you."),
)

DIRECT_LANGUAGE = (
    ("hi", "Hello! What would you like to explore today?"),
    ("hello", "Hello. I can help with code, explanations, summaries, or translations."),
    ("What can you help me with?", "I can help write and explain code, summarize text, improve wording, and translate short passages."),
    ("Thanks for your help.", "You are welcome. I am glad that helped."),
)

UNARY_SPECS = (
    ("square", "number", "number * number", "returns the square of a number"),
    ("cube", "number", "number * number * number", "returns the cube of a number"),
    ("double", "value", "value * 2", "returns twice a value"),
    ("absolute_value", "number", "abs(number)", "returns the absolute value of a number"),
    ("is_even", "number", "number % 2 == 0", "returns True when a number is even"),
    ("is_positive", "number", "number > 0", "returns True when a number is positive"),
    ("to_lower", "text", "text.lower()", "returns text in lowercase"),
    ("reverse_text", "text", "text[::-1]", "returns text reversed"),
    ("count_vowels", "text", "sum(character.lower() in \"aeiou\" for character in text)", "counts vowels without case sensitivity"),
    ("word_count", "text", "len(text.split())", "counts whitespace-separated words"),
    ("is_palindrome", "text", "text.lower() == text.lower()[::-1]", "returns True when text is a case-insensitive palindrome"),
    ("first_character", "text", "text[0]", "returns the first character of text"),
)

BINARY_SPECS = (
    ("add", "left", "right", "left + right", "returns the sum of two numbers"),
    ("subtract", "left", "right", "left - right", "subtracts right from left"),
    ("multiply", "left", "right", "left * right", "returns the product of two numbers"),
    ("maximum", "left", "right", "max(left, right)", "returns the larger of two numbers"),
    ("minimum", "left", "right", "min(left, right)", "returns the smaller of two numbers"),
    ("join_words", "first", "second", "f\"{first} {second}\"", "joins two words with one space"),
)

LIST_SPECS = (
    ("sum_values", "values", "return sum(values)", "returns the sum of a list of numbers"),
    ("sorted_values", "values", "return sorted(values)", "returns a sorted copy of a list"),
    ("unique_sorted", "values", "return sorted(set(values))", "returns sorted unique values"),
    ("positive_values", "values", "return [value for value in values if value > 0]", "keeps only positive values"),
    ("double_values", "values", "return [value * 2 for value in values]", "returns a list with every value doubled"),
    ("count_items", "values", "return len(values)", "returns the number of items in a list"),
    ("last_value", "values", "return values[-1]", "returns the last item in a non-empty list"),
    ("dedupe", "values", "return list(dict.fromkeys(values))", "removes duplicates while preserving order"),
)

DEBUG_SPECS = (
    ("def square(number): number * number", "def square(number):\n    return number * number"),
    ("def add(left, right) return left + right", "def add(left, right):\n    return left + right"),
    ("def first(values): return values[1]", "def first(values):\n    return values[0]"),
    ("def total(values): return sum(values[:-1])", "def total(values):\n    return sum(values)"),
    ("def lower(text): return text.lower", "def lower(text):\n    return text.lower()"),
    ("def contains(values, target): return target not in values", "def contains(values, target):\n    return target in values"),
    ("def parse(text): return int(text)", "def parse(text):\n    try:\n        return int(text)\n    except ValueError:\n        return None"),
    ("def append(item, values=[]): values.append(item); return values", "def append(item, values=None):\n    if values is None:\n        values = []\n    values.append(item)\n    return values"),
)

EXPLANATIONS = (
    ("Why use a set for membership checks?", "A set uses hashing, so membership is usually constant time instead of scanning every list item."),
    ("What does a for loop do?", "A for loop visits each item in an iterable and runs its body once for that item."),
    ("What does return do in a function?", "return ends the function and sends a value back to the caller."),
    ("Why write a test before changing code?", "A focused test records expected behavior and can reveal a regression after the change."),
    ("What does a dictionary key do?", "A key identifies the value that a dictionary stores under that name."),
    ("Why catch a narrow exception?", "A narrow exception handler addresses the expected failure while leaving unrelated bugs visible."),
    ("What is a pure function?", "A pure function gives the same output for the same input and does not change outside state."),
    ("What is recursion?", "Recursion is a function solving a problem by calling itself on a smaller case until a base case."),
)

EVAL_EXPLANATION_QUESTIONS = (
    "How does set membership compare with searching a list?",
    "Explain how a for statement processes an iterable.",
    "What happens when a Python function reaches return?",
    "How can a test protect existing behavior during an edit?",
    "How do keys identify values in a dictionary?",
    "Why avoid catching every exception in one handler?",
    "Describe the output and side-effect properties of a pure function.",
    "Explain how recursive calls eventually stop.",
)

ALGORITHMS = (
    ("binary search", "On a sorted list, compare the target with the midpoint and discard the half that cannot contain it."),
    ("a stack", "A stack removes the most recently added item first, which makes it useful for nested delimiters."),
    ("a queue", "A queue removes the oldest item first, which models fair arrival order."),
    ("two pointers", "Two pointers move through a sequence while preserving an interval that may still contain the answer."),
    ("dynamic programming", "Dynamic programming stores solutions to overlapping subproblems and reuses them."),
    ("a hash table", "A hash table maps keys to buckets so lookup is usually constant time."),
)


def _language_rows(split: str) -> list[dict[str, object]]:
    names = ("Maya", "Noah", "Riley", "Sam", "Avery", "Jordan", "Taylor", "Lee")
    if split == "eval":
        names = ("Quinn", "Casey", "Drew", "Parker")
    rows: list[dict[str, object]] = []
    for name_index, name in enumerate(names):
        for wording_index, wording in enumerate(GREETING_WORDINGS):
            rows.append(_row(f"{split}-english-greeting-{name_index}-{wording_index}", wording.format(name=name), f"Hello, {name}! What would you like to explore today?", "language_generation"))
    concept_wordings = ("Explain {concept} in plain English.", "Give a short definition of {concept}.", "What is {concept}?", "Describe {concept} for a beginner.")
    grammar_wordings = ("Correct the grammar: {text}", "Rewrite this sentence correctly: {text}", "Improve the grammar in: {text}")
    writing_wordings = ("Write a concise message for {scenario}.", "Create a friendly note about {scenario}.", "Draft two clear sentences for {scenario}.")
    if split == "eval":
        concept_wordings = ("In one sentence, describe the purpose of {concept}.",)
        grammar_wordings = ("Return a grammatically correct version of this sentence: {text}",)
        writing_wordings = ("Compose a brief note suitable for {scenario}.",)
    for concept_index, (concept, answer) in enumerate(CONCEPTS):
        for wording_index, wording in enumerate(concept_wordings):
            rows.append(_row(f"{split}-english-concept-{concept_index}-{wording_index}", wording.format(concept=concept), answer, "language_generation"))
    for fix_index, (incorrect, corrected) in enumerate(GRAMMAR_FIXES):
        for wording_index, wording in enumerate(grammar_wordings):
            rows.append(_row(f"{split}-english-grammar-{fix_index}-{wording_index}", wording.format(text=incorrect), corrected, "language_generation"))
    for scenario_index, (scenario, answer) in enumerate(WRITING_SCENARIOS):
        for wording_index, wording in enumerate(writing_wordings):
            rows.append(_row(f"{split}-english-writing-{scenario_index}-{wording_index}", wording.format(scenario=scenario), answer, "language_generation"))
    if split == "train":
        for direct_index, (prompt, answer) in enumerate(DIRECT_LANGUAGE):
            rows.append(_row(f"{split}-english-direct-{direct_index}", prompt, answer, "language_generation"))
    return rows


def _code_rows(split: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    unary_wordings = ("Write a Python function named {name} that {description}.", "Implement {name} in Python; it should {description}.", "Give Python code for a function that {description}.", "Create a small Python helper that {description}.")
    if split == "eval":
        unary_wordings = ("Provide the implementation of {name}: a Python function that {description}.",)
    for spec_index, (name, argument, expression, description) in enumerate(UNARY_SPECS):
        answer = f"def {name}({argument}):\n    return {expression}"
        for wording_index, wording in enumerate(unary_wordings):
            rows.append(_row(f"{split}-python-unary-{spec_index}-{wording_index}", wording.format(name=name, description=description), answer, "code_generation"))
    binary_wordings = ("Write {name}({left}, {right}) in Python to {description}.", "Implement a Python function called {name} that {description}.", "Generate Python code for {name}; it should {description}.")
    if split == "eval":
        binary_wordings = ("Supply Python code defining {name}({left}, {right}), which {description}.",)
    for spec_index, (name, left, right, expression, description) in enumerate(BINARY_SPECS):
        answer = f"def {name}({left}, {right}):\n    return {expression}"
        for wording_index, wording in enumerate(binary_wordings):
            rows.append(_row(f"{split}-python-binary-{spec_index}-{wording_index}", wording.format(name=name, left=left, right=right, description=description), answer, "code_generation"))
    list_wordings = ("Write a Python function named {name} that {description}.", "Implement {name}(values) so it {description}.", "Create Python code that {description}.")
    if split == "eval":
        list_wordings = ("Define a Python helper called {name} which {description}.",)
    for spec_index, (name, argument, body, description) in enumerate(LIST_SPECS):
        answer = f"def {name}({argument}):\n    {body}"
        for wording_index, wording in enumerate(list_wordings):
            rows.append(_row(f"{split}-python-list-{spec_index}-{wording_index}", wording.format(name=name, description=description), answer, "code_generation"))
    debug_wordings = ("Fix this Python function: {broken}", "Correct the bug in this Python code: {broken}", "Repair this Python snippet: {broken}")
    if split == "eval":
        debug_wordings = ("Show a corrected implementation for this Python function: {broken}",)
    for debug_index, (broken, fixed) in enumerate(DEBUG_SPECS):
        for wording_index, wording in enumerate(debug_wordings):
            rows.append(_row(f"{split}-python-debug-{debug_index}-{wording_index}", wording.format(broken=broken), fixed, "code_debugging"))
    for explanation_index, (question, answer) in enumerate(EXPLANATIONS):
        if split == "eval":
            question = EVAL_EXPLANATION_QUESTIONS[explanation_index]
        rows.append(_row(f"{split}-python-explanation-{explanation_index}", question, answer, "code_explanation"))
    algorithm_wordings = (
        "Describe the core idea behind {name}.",
        "Explain {name} for a beginner.",
        "Give a short plain-English explanation of {name}.",
    )
    if split == "eval":
        algorithm_wordings = ("How does {name} work? Give its main principle.",)
    for algorithm_index, (name, answer) in enumerate(ALGORITHMS):
        for wording_index, wording in enumerate(algorithm_wordings):
            rows.append(_row(f"{split}-python-algorithm-{algorithm_index}-{wording_index}", wording.format(name=name), answer, "algorithm_reasoning"))
    return rows


def build_rows(split: str) -> list[dict[str, object]]:
    if split not in {"train", "eval"}:
        raise ValueError("split must be train or eval")
    return _language_rows(split) + _code_rows(split)


def write_jsonl(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--eval", dest="eval_path", type=Path, required=True)
    args = parser.parse_args()
    train_rows = build_rows("train")
    eval_rows = build_rows("eval")
    write_jsonl(train_rows, args.train)
    write_jsonl(eval_rows, args.eval_path)
    print(json.dumps({"train_records": len(train_rows), "eval_records": len(eval_rows), "source": SOURCE, "license": LICENSE}))


if __name__ == "__main__":
    main()
