"""Bounded lexical retrieval of verbatim references, without model execution."""

from __future__ import annotations

import re

MAX_SOURCE_BYTES = 12_000
MAX_PROMPT_BYTES = 2_000
_WORDS = re.compile(r"[a-z0-9]+(?:'[a-z]+)?", re.IGNORECASE)
_STOP = frozenset("a an the is are was were be been being do does did can could would should will shall may might what which who whom whose when where why how i you he she it we they me my your his her its our their of to in on at by for from with about and or but as that this these those please tell explain answer question according source passage text".split())


def _terms(text: str) -> set[str]:
    return {word for word in _WORDS.findall(text.casefold()) if word not in _STOP}


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
