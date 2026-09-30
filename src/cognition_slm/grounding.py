"""Bounded lexical retrieval of verbatim references, without model execution."""

from __future__ import annotations

import re
import unicodedata

MAX_SOURCE_BYTES = 12_000
MAX_PROMPT_BYTES = 2_000
# Python's \w leaves out combining marks, which would split words in scripts such as Hindi.
_MARKS = "".join(chr(code) for code in range(0x300, 0x20000) if unicodedata.category(chr(code))[0] == "M")
# Han and hiragana have no spaces between words, so each of their characters is a term of its own.
_SINGLE = "\u3041-\u309f\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0003134f"
_LETTER = f"(?:[^\\W_{_SINGLE}]|[{_MARKS}])"
# Words are runs of letters, digits and marks in any script, found the same way in questions,
# ranking and highlights, so a highlight always covers a whole ranked word.
_WORDS = re.compile(f"(?=[^\\W_])[{_SINGLE}]|{_LETTER}+(?:'{_LETTER}+)?")
_STOP = frozenset("a an the is are was were be been being do does did can could would should will shall may might what which who whom whose when where why how i you he she it we they me us him them my your his her its our their of to in on at by for from with about and or but as that this these those please tell explain answer question according source passage text many much any some there".split())
# Questions typed without apostrophes spell "what's" as "whats", which would otherwise stem into a topic.
_STOP |= frozenset("whats wheres whos hows whens whys theres thats".split())
# A negated auxiliary says no more about the topic than the auxiliary does, so "Why can't I print?" asks about printing.
_STOP |= frozenset("have has had must".split())
_STOP |= frozenset(f"{word}{end}" for word in "do does did is are was were has have had could would should might must".split()
                   for end in ("n't", "nt")) | frozenset("can't cant won't wont shan't shant ain't aint".split())
# Hiragana mostly spells grammar (particles and verb endings), and these Han characters spell function and
# question words, so as single-character terms they would let a question match any passage in its language.
_STOP |= frozenset(chr(code) for code in range(0x3041, 0x30a0)) | frozenset("的了是在和与也都就很吗呢吧啊么什谁哪怎样这那个为何誰")
# A hiragana letter with the voicing marks that decomposed text keeps apart from it.
_KANA = re.compile("[\u3041-\u3096\u309d\u309e][\u3099\u309a]*")
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
    # The plural goes first, so "meetings" goes on to lose its ing as "meeting" does.
    if word.endswith("ies") and len(word) > 4:
        word = word[:-3] + "y"
    elif word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        word = word[:-1]
    # "used" leaves a 2-letter stem, so ing/ed allow it and "use" drops its e to meet it at "us".
    for suffix in ("ing", "ied", "ed"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 2:
            word = word[: -len(suffix)] + ("y" if suffix == "ied" else "")
            # "stopped" and "running" double the last letter of "stop" and "run"; "called" and "missed" do not.
            if suffix != "ied" and len(word) > 2 and word[-1] == word[-2] and word[-1] not in "aeiouflsz":
                word = word[:-1]
            break
    # A final y reads as i, so "copy" meets "copied" and "movie" meets "movies" (read as "movy").
    if word.endswith("y") and len(word) > 2:
        return word[:-1] + "i"
    return word[:-1] if word.endswith("e") and len(word) > 2 else word


def _terms(text: str) -> set[str]:
    # NFKD before casefolding turns styled letters such as a math bold P into a plain P that then folds to p.
    # Dropping accents after the second NFKD lets cafe match café, and the dotted capital I folds to i plus a dot.
    folded = unicodedata.normalize("NFKD", unicodedata.normalize("NFKD", text).casefold().replace("\u2019", "'"))
    words = _WORDS.findall("".join(char for char in folded if unicodedata.category(char) != "Mn"))
    words = (word.removesuffix("'s") for word in words)
    return {_stem(_IRREGULAR.get(word, word)) for word in words if word not in _STOP}


def _kana_pairs(text: str) -> list[tuple[str, int, int]]:
    """Each pair of neighbouring hiragana letters, with its start and end offsets in code points."""
    letters = [(unicodedata.normalize("NFC", match.group()), match.start(), match.end()) for match in _KANA.finditer(text)]
    return [(first[0] + second[0], first[1], second[2]) for first, second in zip(letters, letters[1:]) if first[2] == second[1]]


def _matches(passage: str, query: set[str], pairs: bool = False) -> list[list[int]]:
    """Start and end offsets, in code points, of the passage words that share a term with the question."""
    if pairs:
        # Neighbouring pairs overlap, so they merge into one span per run of matched letters.
        spans: list[list[int]] = []
        for pair, start, end in _kana_pairs(passage):
            if pair not in query:
                continue
            if spans and start <= spans[-1][1]:
                spans[-1][1] = end
            else:
                spans.append([start, end])
        return spans
    # Symbols such as ㎏ or ﬁ become letters only under NFKD, as in _terms, so words are found in each character's
    # decomposition and mapped back to the characters they came from. Curly apostrophes become straight ones, so a
    # word like can’t stays whole.
    origins, pieces = [], []
    for index, char in enumerate(passage.replace("\u2019", "'")):
        piece = unicodedata.normalize("NFKD", char)
        pieces.append(piece)
        origins += [index] * len(piece)
    spans: list[list[int]] = []
    for match in _WORDS.finditer("".join(pieces)):
        start, end = origins[match.start()], origins[match.end() - 1] + 1
        # ½ decomposes to 1⁄2, two words from one character, which get one span.
        if (not spans or start >= spans[-1][1]) and _terms(passage[start:end]) & query:
            spans.append([start, end])
    return spans


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
    # A question spelled only in hiragana has no terms, so it matches on pairs of neighbouring hiragana, which
    # carry more of each word than the single letters that particles and verb endings share.
    pairs = not query
    if pairs:
        query = {pair for pair, _, _ in _kana_pairs(request["prompt"])}
    # Keep paragraph context: a later sentence may correct an earlier claim.
    passages = re.split(r"\r?\n\s*\r?\n", request["source_text"])
    ranked = []
    seen = set()
    for index, raw in enumerate(passages):
        passage = raw.strip()
        if not passage or passage in seen:
            continue
        seen.add(passage)
        terms = {pair for pair, _, _ in _kana_pairs(passage)} if pairs else _terms(passage)
        overlap = query & terms
        if query and overlap and len(overlap) / len(query) >= 0.6:
            ranked.append((len(overlap), len(overlap) / max(1, len(terms)), index, passage))
    ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
    sources = [{"id": f"S{i + 1}", "text": item[3], "matches": _matches(item[3], query, pairs)} for i, item in enumerate(ranked[:3])]
    return {
        "mode": "source_excerpts",
        "abstained": not sources,
        "sources": sources,
        "text": "\n\n".join(f"[{source['id']}] {source['text']}" for source in sources)
        if sources else "I don't have enough information in the supplied references. Try a more specific question or add a relevant source.",
        "finish_reason": "sources" if sources else "insufficient_evidence",
    }
