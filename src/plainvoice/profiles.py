"""Validated, portable preferences for wording rather than authorship labels."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MAX_RULES = 256
MAX_PHRASE_LENGTH = 160
MAX_PROFILE_FILE_BYTES = 256 * 1024


def _has_surrogate(value: str) -> bool:
    return any(0xD800 <= ord(char) <= 0xDFFF for char in value)


def _real_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


@dataclass(frozen=True)
class PhraseRule:
    """A literal phrase and its bounded additive logit preference."""

    text: str
    bias: float

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("rule text must be a nonempty string")
        if len(self.text) > MAX_PHRASE_LENGTH:
            raise ValueError(f"rule text must have at most {MAX_PHRASE_LENGTH} characters")
        if any(ord(char) < 32 or 0xD800 <= ord(char) <= 0xDFFF for char in self.text):
            raise ValueError("rule text must not contain control characters or surrogates")
        bias = _real_number(self.bias, "rule bias")
        if abs(bias) > 3:
            raise ValueError("absolute rule bias must not exceed 3")
        object.__setattr__(self, "bias", bias)


@dataclass(frozen=True)
class StyleProfile:
    """A small explicit preference set, safe to serialize and inspect."""

    name: str
    description: str
    rules: tuple[PhraseRule, ...]
    max_bias: float = 2.0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or not self.name.strip()
            or len(self.name) > 80
            or _has_surrogate(self.name)
        ):
            raise ValueError("profile name must contain 1 to 80 characters")
        if (
            not isinstance(self.description, str)
            or len(self.description) > 2000
            or _has_surrogate(self.description)
        ):
            raise ValueError("profile description must be a string of at most 2000 characters")
        if not isinstance(self.rules, tuple) or len(self.rules) > MAX_RULES:
            raise ValueError(f"profile rules must be a tuple of at most {MAX_RULES} rules")
        max_bias = _real_number(self.max_bias, "max_bias")
        if not 0 < max_bias <= 3:
            raise ValueError("max_bias must be greater than 0 and at most 3")
        seen: set[str] = set()
        for rule in self.rules:
            if not isinstance(rule, PhraseRule):
                raise ValueError("every rule must be a PhraseRule")
            if rule.text in seen:
                raise ValueError("duplicate phrase rules are not allowed")
            if abs(rule.bias) > max_bias:
                raise ValueError("rule bias exceeds profile max_bias")
            seen.add(rule.text)
        object.__setattr__(self, "max_bias", max_bias)


def default_profile() -> StyleProfile:
    """Return illustrative preferences; they are not a learned human-word list."""
    return StyleProfile(
        name="plain-zh",
        description="中文简明写作的可编辑偏好；命中仅供复核，不证明文本由 AI 创作。",
        rules=(
            PhraseRule("在当今快速发展的时代", -0.9),
            PhraseRule("不难看出", -0.6),
            PhraseRule("综上所述", -0.5),
            PhraseRule("让我们拭目以待", -0.7),
            PhraseRule("毋庸置疑", -0.7),
            PhraseRule("具有十分重要的意义", -0.6),
            PhraseRule("开启全新的篇章", -0.6),
        ),
    )


def profile_from_dict(data: dict[str, Any]) -> StyleProfile:
    """Parse versioned JSON, rejecting misspellings and ambiguous values."""
    if not isinstance(data, dict):
        raise ValueError("profile must be a JSON object")
    required = {"schema_version", "name", "description", "rules", "max_bias"}
    if set(data) != required:
        raise ValueError("profile requires exactly schema_version, name, description, rules, max_bias")
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise ValueError("unsupported profile schema_version; expected 1")
    if not isinstance(data["rules"], list) or len(data["rules"]) > MAX_RULES:
        raise ValueError(f"rules must be a JSON array of at most {MAX_RULES} entries")
    rules = []
    for item in data["rules"]:
        if not isinstance(item, dict) or set(item) != {"text", "bias"}:
            raise ValueError("each rule requires exactly text and bias")
        rules.append(PhraseRule(text=item["text"], bias=item["bias"]))
    return StyleProfile(
        name=data["name"],
        description=data["description"],
        rules=tuple(rules),
        max_bias=data["max_bias"],
    )


def profile_to_dict(profile: StyleProfile) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "name": profile.name,
        "description": profile.description,
        "rules": [{"text": rule.text, "bias": rule.bias} for rule in profile.rules],
        "max_bias": profile.max_bias,
    }


def load_profile(path: str | Path) -> StyleProfile:
    profile_path = Path(path)
    with profile_path.open("rb") as handle:
        raw = handle.read(MAX_PROFILE_FILE_BYTES + 1)
    if len(raw) > MAX_PROFILE_FILE_BYTES:
        raise ValueError("profile file exceeds 256 KiB")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        parsed: dict[str, Any] = {}
        for key, value in pairs:
            if key in parsed:
                raise ValueError("duplicate JSON keys are not allowed in a profile")
            parsed[key] = value
        return parsed

    try:
        data = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("profile must be valid UTF-8 JSON") from exc
    return profile_from_dict(data)
