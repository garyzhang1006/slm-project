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
# A clock time with am or pm is one word, so "9 am" can rank apart from "9 pm" while "am" alone stays a stop word.
# Minutes and a range share the am or pm, as in "9:30 am" and "9-11 am".
_CLOCK = r"\d{1,2}(?::\d\d)?(?: ?[-\u2013] ?\d{1,2}(?::\d\d)?)? ?[aApP]\.?[mM]\b\.?"
_TIME = re.compile(_CLOCK)
# Hours are the numbers not written after a colon.
_HOUR = re.compile(r"(?<![:\d])\d{1,2}")
# A number written with leading zeros, as the 08 of "08:30", is the same number as a bare 8. A zero after a
# decimal point, comma or colon stays, so 1.05 keeps apart from 1.5 and the minutes of 9:05 stay 05.
_PADDED = re.compile(r"(?<![\d.,:])0+(?=\d)")
# A number with thousands separators is one word, so 1,200 is the number 1200 and not the numbers 1 and 200.
_THOUSANDS = r"\d{1,3}(?:,\d{3})+(?!\d)"
# Letters joined by hyphens are one word, as in Wi-Fi; digits are left out, so 10-20 stays two numbers.
_ALPHA = f"(?:[^\\W\\d_{_SINGLE}]|[{_MARKS}])"
# Words are runs of letters, digits and marks in any script, found the same way in questions,
# ranking and highlights, so a highlight always covers a whole ranked word.
_WORDS = re.compile(f"{_CLOCK}|{_THOUSANDS}|(?=[^\\W_])[{_SINGLE}]|{_ALPHA}+(?:-{_ALPHA}+)+(?:'{_LETTER}+)?|"
                    f"{_LETTER}+(?:'{_LETTER}+)?")
# A number glued to a unit of two or more letters, as in 10GB, is also written apart; gate 12B stays one word.
_GLUED = re.compile(r"(\d+)([^\W\d_]{2,})")
_STOP = frozenset("a an the am is are was were be been being do does did can could would should will shall may might what which who whom whose when where why how i you he she it we they me us him them my your his her its our their of to in on at by for from with about and or but as that this these those please tell explain answer question according source passage text many much any some there".split())
# "What do the sources say about printing?" asks only about printing, yet "What are the sources of protein?" asks
# about sources, so these words count when a passage has them and a question may leave them out.
_OPTIONAL = frozenset("say says said saying mention mentions mentioned sources passages texts".split())
# Questions typed without apostrophes spell "what's" as "whats", which would otherwise stem into a topic.
_STOP |= frozenset("whats wheres whos hows whens whys theres thats".split())
# A negated auxiliary says no more about the topic than the auxiliary does, so "Why can't I print?" asks about printing.
_STOP |= frozenset("have has had must".split())
_STOP |= frozenset(f"{word}{end}" for word in "do does did is are was were has have had could would should might must".split()
                   for end in ("n't", "nt")) | frozenset("can't cant cannot won't wont shan't shant ain't aint".split())
# Contracted pronouns and question words say no more than the words they shorten, so "I'm" and "where'd" are not
# topics. Forms such as "well", "shed" and "wed" stay, since they are also ordinary words.
_STOP |= frozenset(f"{word}'{end}" for word in "i you he she it we they what where who how when why there that".split()
                   for end in ("m", "ve", "re", "d", "ll")) | frozenset("im ive youre youve youll youd theyre theyve theyll theyd weve".split())
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
            break
    # "stopped" and "running" double the last letter of "stop" and "run", so a double consonant reads as one;
    # "add" undoubles too, to meet "added". "called" and "missed" keep theirs, and digits never undouble,
    # so 100 stays apart from 10.
    if len(word) > 2 and word[-1] == word[-2] and word[-1] in "bcdgkmnprt":
        word = word[:-1]
    # A final y reads as i, so "copy" meets "copied" and "movie" meets "movies" (read as "movy").
    if word.endswith("y") and len(word) > 2:
        return word[:-1] + "i"
    return word[:-1] if word.endswith("e") and len(word) > 2 else word


def _words(text: str, before: str = "") -> list[str]:
    # NFKD before casefolding turns styled letters such as a math bold P into a plain P that then folds to p.
    # Dropping accents after the second NFKD lets cafe match café, and the dotted capital I folds to i plus a dot.
    folded = unicodedata.normalize("NFKD", unicodedata.normalize("NFKD", text).casefold().replace("\u2019", "'"))
    plain = "".join(char for char in folded if unicodedata.category(char) != "Mn")
    # before is the character ahead of text in its passage, so a highlight reads the 05 of 1.05 as ranking did.
    words = _WORDS.findall(_PADDED.sub("", before + plain)[len(before):])
    return [word.removesuffix("'s").replace(",", "") for word in words]


def _pieces(word: str) -> list[str]:
    """The pieces a word is also written as, such as wi and fi for Wi-Fi or 10 and gb for 10GB."""
    if _TIME.fullmatch(word):
        return [word]
    pieces: list[str] = []
    for part in word.split("-"):
        glued = _GLUED.fullmatch(part)
        pieces += glued.groups() if glued else [part]
    return pieces


def _term(form: str) -> str | None:
    return None if form in _STOP else _stem(_IRREGULAR.get(form, form))


def _word_terms(word: str) -> set[str]:
    if _TIME.fullmatch(word):
        # "9:30 am" keeps 9 and 30 and adds 9am, so it still matches a bare 9 but ranks above "9:30 pm".
        meridiem = word.replace(".", "")[-2] + "m"
        other = "pm" if meridiem == "am" else "am"
        hours = _HOUR.findall(word)
        terms = set()
        # "12 am" and "12 pm" are each written for both noon and midnight, so the first hour of a range that ends
        # right on 12, as in "9-12 am", gets both halves of the day.
        if len(hours) == 2 and re.findall(r"\d+(?::\d\d)?", word)[-1] in ("12", "12:00"):
            terms.add(hours[0] + other)
        # Otherwise the am or pm follows the last hour, so a first hour past it, as in "11-1 pm" or
        # "11:30-12:30 pm", is in the other half of the day.
        elif len(hours) == 2 and int(hours[0]) % 12 > int(hours[1]) % 12:
            terms.add(hours.pop(0) + other)
        return terms | set(re.findall(r"\d+", word)) | {hour + meridiem for hour in hours}
    # A word in pieces is a term whole as well, so the wifi of a question meets the Wi-Fi of a passage.
    pieces = _pieces(word)
    forms = pieces + ["".join(pieces)] if len(pieces) > 1 else pieces
    return {term for term in map(_term, forms) if term is not None}


def _terms(text: str, before: str = "") -> set[str]:
    return set().union(*(_word_terms(word) for word in _words(text, before)))


def _units(text: str) -> dict[object, tuple[list[frozenset[str]], bool]]:
    """Each distinct word of text once, with the term sets that spell it and whether a question may leave it out.

    Wi-Fi is spelled {wifi} or {wi, fi}, so it counts once, and a passage with either spelling has it. To-do has only
    its whole form, since to and do are stop words, and a passage writing "to do" can never hold it, so a question
    may leave it out, as it did when to-do was two stop words.
    """
    units: dict[object, tuple[list[frozenset[str]], bool]] = {}
    meta: set[str] = set()
    for word in _words(text):
        pieces = _pieces(word)
        if len(pieces) == 1:
            # A clock time gives several terms, such as 9, 30 and 9am for "9:30 am", and each counts on its own.
            for term in _word_terms(word):
                # "texts" leaves the text of "Is texting mentioned in the texts?" required.
                optional = word in _OPTIONAL and units.get(term, ([], True))[1]
                units[term] = ([frozenset({term})], optional)
                if optional:
                    meta.add(term)
            continue
        whole = _term("".join(pieces))
        parts = frozenset(term for term in map(_term, pieces) if term is not None)
        spellings = [spelling for spelling in (frozenset({whole} if whole else ()), parts) if spelling]
        if spellings:
            units[(whole, parts)] = (spellings, not parts)
    # "What do the sources say?" has no other topic, so its words stay required.
    if all(optional for _, optional in units.values()):
        units.update((term, (units[term][0], False)) for term in meta)
    return units


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
    joined = "".join(pieces)
    for match in _WORDS.finditer(joined):
        start, end = origins[match.start()], origins[match.end() - 1] + 1
        before = joined[match.start() - 1] if match.start() else ""
        # ½ decomposes to 1⁄2, two words from one character, which get one span.
        if (not spans or start >= spans[-1][1]) and _terms(passage[start:end], before) & query:
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
    units = _units(request["prompt"])
    # A question spelled only in hiragana has no terms, so it matches on pairs of neighbouring hiragana, which
    # carry more of each word than the single letters that particles and verb endings share.
    pairs = not units
    if pairs:
        units = {pair: ([frozenset({pair})], False) for pair, _, _ in _kana_pairs(request["prompt"])}
    query = set().union(*(spelling for spellings, _ in units.values() for spelling in spellings))
    # Keep paragraph context: a later sentence may correct an earlier claim.
    passages = re.split(r"\r?\n\s*\r?\n", request["source_text"])
    ranked = []
    seen = set()
    for index, raw in enumerate(passages):
        passage = raw.strip()
        if not passage or passage in seen:
            continue
        seen.add(passage)
        if pairs:
            terms = {pair for pair, _, _ in _kana_pairs(passage)}
            size = len(terms)
        else:
            # Density counts the passage's words as the question's are counted, so Wi-Fi is one word, not three.
            passage_units = _units(passage)
            terms = set().union(*(spelling for spellings, _ in passage_units.values() for spelling in spellings))
            size = len(passage_units)
        found = [(any(spelling <= terms for spelling in spellings), optional) for spellings, optional in units.values()]
        matched = sum(hit for hit, _ in found)
        asked = sum(1 for hit, optional in found if hit or not optional)
        if matched and matched / asked >= 0.6:
            ranked.append((matched, matched / max(1, size), index, passage))
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
