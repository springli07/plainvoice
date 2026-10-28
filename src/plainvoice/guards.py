"""Conservative, local checks for facts a rewrite must keep.

These checks compare surface forms, not meaning. Passing them does not prove
semantic equivalence. In particular, entity relationships, attribution and
implicit claims still need review. Failing them blocks output rather than
silently discarding the source's qualifications.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class _Protected:
    start: int
    end: int
    kind: str


_FENCE_OPEN = re.compile(r"(?m)^[ \t]{0,3}(?P<fence>`{3,}|~{3,})[^\n]*")
_BACKTICKS = re.compile(r"`+")
_QUOTE_PAIRS = {"“": "”", "‘": "’", "「": "」", "『": "』", "《": "》", "〈": "〉"}
_URL = re.compile(r"https?://[^\s<>\"'`，。；！？]+", re.IGNORECASE)
_ASCII_WORD = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")

# Long units come first so changing 10 万元 to 10 元 is detected. Unit spelling
# is retained exactly; MB and Mb, for example, must not be silently exchanged.
_UNITS = (
    r"平方公里|平方千米|平方米|立方米|万亿元|万元|亿元|人民币|美元|欧元|英镑|日元|港元|"
    r"小时|分钟|公里|千米|厘米|毫米|毫升|千克|毫克|毫秒|微秒|纳秒|人次|"
    r"(?:kilometers?|kilometres?|centimeters?|centimetres?|millimeters?|millimetres?|"
    r"kilograms?|milligrams?|milliseconds?|seconds?|minutes?|hours?|days?|weeks?|months?|years?|"
    r"dollars?|euros?|pounds?|meters?|metres?|liters?|litres?|percent)(?![A-Za-z])|"
    r"(?:TiB|GiB|MiB|KiB|TB|GB|MB|KB|Tb|Gb|Mb|kb|GHz|MHz|kHz|Hz|kWh|kW|mAh|"
    r"km|cm|mm|kg|mg|ml|mL|ms|ns|us|fps|bps|V|W|A|g|m|s|h|B)(?![A-Za-z])|"
    r"%|％|‰|°C|°F|℃|℉|年|月|日|天|周|秒|米|克|吨|元|万|亿|兆|个|件|名|台|人|次|倍|岁|项|家|套|页"
)
_ARABIC_FACT = re.compile(
    r"(?<![A-Za-z\d_])(?:[¥￥$€£]\s*)?[+−-]?"
    r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+−-]?\d+)?"
    rf"(?:\s*(?:{_UNITS}))?"
)
_CHINESE_FACT = re.compile(
    r"(?<![零〇一二两三四五六七八九十百千万亿点])[零〇一二两三四五六七八九十百千万亿点]+"
    r"(?:万亿元|万元|亿元|小时|分钟|公里|千米|厘米|毫米|毫升|千克|毫克|毫秒|微秒|纳秒|"
    r"年|月|日|天|周|秒|米|克|吨|元|个|件|名|台|人|次|倍|岁|项|家|套|页)"
)
_NEGATION = re.compile(
    r"尚未|并未|没有|不能|不得|不可|并非|不是|未必|不会|不曾|无需|无须|未曾|未能|未|不|无|"
    r"\b(?:cannot|can't|won't|don't|doesn't|didn't|isn't|aren't|wasn't|weren't|"
    r"hasn't|haven't|hadn't|couldn't|wouldn't|shouldn't|mustn't|not|no|never|without)\b",
    re.IGNORECASE,
)
# These stock introductions do not negate the statement that follows. Their
# deletion is an expected style edit, so a bare 不/无 within them is not treated
# as a factual polarity marker. This is a short grammatical exception list,
# not an attempt to infer meaning from arbitrary Chinese prose.
_NONNEGATING_INTRO = re.compile(r"不难(?:看出|发现|理解)|不言而喻|毋庸置疑|无疑")
_QUALIFICATION = re.compile(
    r"大部分|不超过|不低于|至少|至多|最多|最少|可能|或许|也许|大约|预计|据称|暂时|部分|"
    r"某些|通常|一般|几乎|多数|少数|尚未|未必|仅有|仅限|只有|"
    r"(?:仅|约)(?=\s*[\d零〇一二两三四五六七八九十百千万亿])|"
    r"\b(?:at\s+least|at\s+most|may|might|could|must|should|likely|possibly|estimated|"
    r"approximately|about|only|some|most|many|typically|usually)\b",
    re.IGNORECASE,
)


def _mask(text: str, spans: list[_Protected] | list[tuple[int, int]]) -> str:
    chars = list(text)
    for span in spans:
        start, end = (span.start, span.end) if isinstance(span, _Protected) else span
        # Keep line boundaries so Markdown fence matching and quote boundaries
        # retain their original offsets.
        for index in range(start, end):
            if chars[index] != "\n":
                chars[index] = " "
    return "".join(chars)


def _quote_endings(text: str) -> dict[int, int]:
    """Index quote pairs with bounded scans, including unmatched openers.

    A lookahead scan for each opener is quadratic for long unbalanced input.
    Independent stacks preserve the small parser's same-delimiter nesting
    behavior; the caller still selects outermost completed spans greedily.
    """
    escaped: list[bool] = []
    slash_run = 0
    for char in text:
        escaped.append(bool(slash_run % 2))
        slash_run = slash_run + 1 if char == "\\" else 0

    stacks: dict[str, list[int]] = {char: [] for char in _QUOTE_PAIRS}
    closing_to_opening = {closing: opening for opening, closing in _QUOTE_PAIRS.items()}
    endings: dict[int, int] = {}
    for index, character in enumerate(text):
        if escaped[index]:
            continue
        if character in stacks:
            stacks[character].append(index)
        elif character in closing_to_opening:
            stack = stacks[closing_to_opening[character]]
            if stack:
                endings[stack.pop()] = index + 1

    # Straight quotes have no nesting rule. Find each opener's next eligible
    # closing mark once, rather than searching the remaining line repeatedly.
    next_close: dict[str, int] = {}
    for index in range(len(text) - 1, -1, -1):
        character = text[index]
        if character not in {"\"", "'"} or escaped[index]:
            continue
        previous = text[index - 1] if index else ""
        following = text[index + 1] if index + 1 < len(text) else ""
        can_open = not (
            (character == "'" and previous in _ASCII_WORD)
            or (character == '"' and previous.isdigit())
        )
        if can_open and character in next_close:
            endings[index] = next_close[character] + 1
        if character != "'" or following not in _ASCII_WORD:
            next_close[character] = index
    return endings


def _inline_code_spans(text: str) -> list[_Protected]:
    """Pair equal-width backtick runs within a line without repeated searches."""
    runs = list(_BACKTICKS.finditer(text))
    keys: list[tuple[int, int]] = []
    line = previous_end = 0
    for run in runs:
        line += text.count("\n", previous_end, run.start())
        keys.append((line, run.end() - run.start()))
        previous_end = run.end()
    next_same: list[int | None] = [None] * len(runs)
    next_runs: dict[tuple[int, int], int] = {}
    for index in range(len(runs) - 1, -1, -1):
        key = keys[index]
        next_same[index] = next_runs.get(key)
        next_runs[key] = index
    result: list[_Protected] = []
    position = 0
    for index, opening in enumerate(runs):
        if opening.start() < position:
            continue
        closing_index = next_same[index]
        if closing_index is not None:
            end = runs[closing_index].end()
            result.append(_Protected(opening.start(), end, "code"))
            position = end
    return result


def _protected(text: str) -> list[_Protected]:
    result: list[_Protected] = []
    position = 0
    while opening := _FENCE_OPEN.search(text, position):
        marker = opening.group("fence")
        closing_pattern = re.compile(
            rf"(?m)^[ \t]{{0,3}}{re.escape(marker[0])}{{{len(marker)},}}[ \t]*$"
        )
        next_line = text.find("\n", opening.end())
        closing = closing_pattern.search(text, next_line + 1) if next_line >= 0 else None
        end = closing.end() if closing else len(text)
        result.append(_Protected(opening.start(), end, "code"))
        position = end

    masked = _mask(text, result)
    result.extend(_inline_code_spans(masked))

    masked = _mask(text, result)
    endings = _quote_endings(masked)
    position = 0
    for start in range(len(masked)):
        if start < position or start not in endings:
            continue
        position = endings[start]
        result.append(_Protected(start, position, "quote"))
    return sorted(result, key=lambda span: (span.start, span.end))


def protected_spans(text: str) -> list[tuple[int, int]]:
    """Return merged, half-open source offsets for code and quoted material.

    The parser is deliberately small; it supports fenced/inline Markdown code,
    common Chinese paired quotes and balanced straight English quotes. An open
    fenced block is protected through end of input. It is not a Markdown parser.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    merged: list[tuple[int, int]] = []
    for span in _protected(text):
        if merged and span.start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], span.end))
        else:
            merged.append((span.start, span.end))
    return merged


def _url_spans(text: str) -> list[tuple[int, int]]:
    result = []
    for match in _URL.finditer(text):
        value = match.group().rstrip(".,;:!?、，。；！？）")
        # A surrounding prose parenthesis is not part of the URL. Balanced URL
        # parentheses are kept (for example a Wikipedia article URL).
        surplus = max(0, value.count(")") - value.count("("))
        trailing = len(value) - len(value.rstrip(")"))
        trim = min(surplus, trailing)
        if trim:
            value = value[:-trim]
        if value:
            result.append((match.start(), match.start() + len(value)))
    return result


def _facts(text: str) -> dict[str, Counter[str]]:
    protected = _protected(text)
    urls = _url_spans(text)
    outside = _mask(_mask(text, protected), urls)
    negation_text = _NONNEGATING_INTRO.sub(lambda match: " " * len(match.group()), outside)
    return {
        "code": Counter(text[span.start : span.end] for span in protected if span.kind == "code"),
        "quotes": Counter(text[span.start : span.end] for span in protected if span.kind == "quote"),
        "urls": Counter(text[start:end] for start, end in urls),
        "numbers": Counter(
            # Allow formatting whitespace around units to change, while keeping
            # the numeric literal, sign and unit spelling intact.
            re.sub(r"\s+", "", match.group())
            for pattern in (_ARABIC_FACT, _CHINESE_FACT)
            for match in pattern.finditer(outside)
        ),
        "negations": Counter(match.group().casefold() for match in _NEGATION.finditer(negation_text)),
        "qualifications": Counter(
            re.sub(r"\s+", " ", match.group().casefold()) for match in _QUALIFICATION.finditer(outside)
        ),
    }


def preservation_issues(source: str, candidate: str) -> list[str]:
    """List conservative preservation failures without echoing user content.

    Surface forms are compared as multisets, so moving an unchanged quote is
    allowed and deleting one of two identical facts is detected. Exact negation
    and qualification markers are intentionally conservative: a legitimate
    paraphrase may need review. Neither passing nor failing is an AI score.
    """
    if not isinstance(source, str) or not isinstance(candidate, str):
        raise TypeError("source and candidate must be strings")
    before, after = _facts(source), _facts(candidate)
    messages = {
        "code": "Code blocks or inline code changed.",
        "quotes": "Quoted material changed.",
        "urls": "URLs changed.",
        "numbers": "Numeric facts or units changed.",
        "negations": "Negation markers changed; review the meaning.",
        "qualifications": "Qualification markers changed; review the meaning.",
    }
    return [message for key, message in messages.items() if before[key] != after[key]]
