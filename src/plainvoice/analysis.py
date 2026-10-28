"""Offline observations for reviewing prose, with exact source locations."""

from __future__ import annotations

import math
import re
from bisect import bisect_right
from statistics import fmean, pstdev
from typing import Any

from .guards import protected_spans
from .profiles import StyleProfile, default_profile

MAX_TEXT_CHARACTERS = 50_000
DISCLAIMER = (
    "这些结果是写作偏好和表面统计，不是 AI 创作概率，也不证明文字是否由人类创作。"
    "短语、句长和段落统计需结合语境复核；引文与代码不计入短语命中。"
)


def analyze(text: str, profile: StyleProfile | None = None) -> dict[str, Any]:
    """Report literal wording preferences without labeling the author's identity.

    Offsets are Python Unicode code-point offsets into the unchanged input. A
    phrase touching a quotation or code span is excluded. Statistical summaries
    omit protected content and are descriptive rather than optimization targets.
    """
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    if len(text) > MAX_TEXT_CHARACTERS:
        raise ValueError(f"text exceeds {MAX_TEXT_CHARACTERS} characters")
    if any(0xD800 <= ord(char) <= 0xDFFF for char in text):
        raise ValueError("text must contain valid Unicode characters")
    selected = default_profile() if profile is None else profile
    if not isinstance(selected, StyleProfile):
        raise ValueError("profile must be a StyleProfile")

    spans = protected_spans(text)
    span_ends = [end for _, end in spans]
    findings: list[dict[str, Any]] = []
    for rule in selected.rules:
        if rule.bias >= 0:
            continue
        for match in re.finditer(re.escape(rule.text), text):
            # Spans are merged and sorted; skip those ending at/before this hit.
            # Avoid rescanning every quotation for every literal match.
            index = bisect_right(span_ends, match.start())
            if index < len(spans) and spans[index][0] < match.end():
                continue
            findings.append(
                {
                    "start": match.start(),
                    "end": match.end(),
                    "text": match.group(),
                    "message": "命中当前文风偏好中的模板短语。请结合上下文决定保留、具体说明或精简。",
                    "rule": "phrase-preference",
                    "bias": rule.bias,
                }
            )
    findings.sort(key=lambda item: (item["start"], item["end"], item["text"]))

    # Replace with spaces instead of deleting, so protected blocks cannot join
    # two unrelated words or sentences in the derived statistics.
    prose_characters = list(text)
    for start, end in spans:
        prose_characters[start:end] = [" " if char != "\n" else "\n" for char in text[start:end]]
    prose = "".join(prose_characters)
    sentences = [part.strip() for part in re.split(r"[。！？!?]+|(?<!\d)\.|\.(?!\d)", prose)]
    lengths = [len(re.sub(r"\s", "", sentence)) for sentence in sentences if sentence.strip()]
    mean = fmean(lengths) if lengths else 0.0
    cv = pstdev(lengths) / mean if len(lengths) > 1 and mean else 0.0
    if not math.isfinite(cv):  # defensive; bounded text makes this unreachable
        cv = 0.0
    paragraphs = [paragraph for paragraph in re.split(r"\n\s*\n", prose) if paragraph.strip()]
    prose_count = len(re.sub(r"\s", "", prose))
    return {
        "schema_version": 1,
        "profile": selected.name,
        "character_count": len(text),
        "findings": findings,
        "metrics": {
            "sentence_count": len(lengths),
            "paragraph_count": len(paragraphs),
            "sentence_length_mean": round(mean, 2),
            "sentence_length_cv": round(cv, 3),
            "phrase_matches": len(findings),
            "phrases_per_1000_characters": round(len(findings) * 1000 / prose_count, 2) if prose_count else 0.0,
        },
        "disclaimer": DISCLAIMER,
    }
