"""Bounded lexical retrieval of verbatim references, without model execution."""

from __future__ import annotations

import re

MAX_SOURCE_BYTES = 12_000
MAX_PROMPT_BYTES = 2_000
_WORDS = re.compile(r"[a-z0-9]+(?:'[a-z]+)?", re.IGNORECASE)
_STOP = frozenset("a an the is are was were be been being do does did can could would should will shall may might what which who whom whose when where why how i you he she it we they me my your his her its our their of to in on at by for from with about and or but as that this these those please tell explain answer question according source passage text".split())
# Past forms map to the base verb before suffix stripping. Ambiguous forms (saw, found, left, felt, rose) are left out.
_IRREGULAR = {form: base for base, forms in (
    ("become", "became"), ("begin", "began begun"), ("break", "broke broken"), ("bring", "brought"),
    ("build", "built"), ("buy", "bought"), ("catch", "caught"), ("choose", "chose chosen"),
    ("come", "came"), ("draw", "drew drawn"), ("drink", "drank drunk"), ("drive", "drove driven"),
    ("eat", "ate eaten"), ("fight", "fought"), ("fly", "flew flown"), ("forget", "forgot forgotten"),
    ("get", "got gotten"), ("give", "gave given"), ("go", "went gone"), ("grow", "grew grown"),
    ("hear", "heard"), ("hide", "hid hidden"), ("hold", "held"), ("keep", "kept"), ("know", "knew known"),
    ("lead", "led"), ("lose", "lost"), ("make", "made"), ("mean", "meant"), ("meet", "met"),
    ("pay", "paid"), ("ride", "rode ridden"), ("run", "ran"), ("say", "said"), ("see", "seen"),
    ("sell", "sold"), ("send", "sent"), ("sing", "sang sung"), ("sit", "sat"), ("sleep", "slept"),
    ("speak", "spoken"), ("spend", "spent"), ("stand", "stood"), ("steal", "stole stolen"),
    ("swim", "swam swum"), ("take", "took taken"), ("teach", "taught"), ("tell", "told"),
    ("think", "thought"), ("throw", "threw thrown"), ("understand", "understood"), ("wake", "woke woken"),
    ("wear", "wore worn"), ("win", "won"), ("write", "wrote written"),
) for form in forms.split()}


def _stem(word: str) -> str:
    # Light suffix stripping so "panels"/"panel" and "used"/"use" match; _IRREGULAR covers common past forms.
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    # "used" leaves a 2-letter stem, so ing/ed allow it and "use" drops its e to meet it at "us".
    for suffix, minimum in (("ing", 2), ("ed", 2), ("s", 3)):
        if word.endswith(suffix) and not word.endswith("ss") and len(word) - len(suffix) >= minimum:
            word = word[: -len(suffix)]
            break
    return word[:-1] if word.endswith("e") and len(word) > 2 else word


def _terms(text: str) -> set[str]:
    words = _WORDS.findall(text.casefold().replace("\u2019", "'"))
    words = (word.removesuffix("'s") for word in words)
    return {_stem(_IRREGULAR.get(word, word)) for word in words if word not in _STOP}


def source_excerpts(request: dict) -> dict:
    """Return relevant quotes; lexical matches do not certify an answer as true."""
    if not isinstance(request, dict):
        raise ValueError("Expected a JSON object.")
    if set(request) - {"prompt", "source_text"}:
        raise ValueError("Only prompt and source_text are accepted for source excerpts.")
    for key, limit in (("prompt", MAX_PROMPT_BYTES), ("source_text", MAX_SOURCE_BYTES)):
        value = request.get(key)
        if not isinstance(value, str):
            raise ValueError(f"{key} must be a string.")
        if len(value.encode("utf-8")) > limit:
            raise ValueError(f"{key} exceeds {limit} UTF-8 bytes. Shorten the text.")
    if not request["prompt"].strip():
        raise ValueError("Enter a question.")
    query = _terms(request["prompt"])
    # Keep paragraph context: a later sentence may correct an earlier claim.
    passages = re.split(r"\r?\n\s*\r?\n", request["source_text"])
    ranked = []
    seen = set()
    for index, raw in enumerate(passages):
        passage = raw.strip()
        if not passage or passage in seen:
            continue
        seen.add(passage)
        terms = _terms(passage)
        overlap = query & terms
        if query and overlap and len(overlap) / len(query) >= 0.6:
            ranked.append((len(overlap), len(overlap) / max(1, len(terms)), index, passage))
    ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
    sources = [{"id": f"S{i + 1}", "text": item[3]} for i, item in enumerate(ranked[:3])]
    return {
        "mode": "source_excerpts",
        "abstained": not sources,
        "sources": sources,
        "text": "\n\n".join(f"[{source['id']}] {source['text']}" for source in sources)
        if sources else "I don't have enough information in the supplied references. Try a more specific question or add a relevant source.",
        "finish_reason": "sources" if sources else "insufficient_evidence",
    }
