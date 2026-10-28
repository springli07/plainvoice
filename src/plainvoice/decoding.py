"""Tokenizer-aware, bounded phrase completion steering.

Weights apply to a phrase's final token only after its earlier tokens have
actually been generated. They are style preferences, not human/AI labels.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .profiles import StyleProfile


class Tokenizer(Protocol):
    def encode(self, text: str, **kwargs: Any) -> list[int]: ...


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _weight(value: Any, label: str, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or abs(result) > maximum:
        raise ValueError(f"{label} must be finite and within +/-{maximum:g}")
    return result


def tokenizer_vocab_size(tokenizer: Any) -> int | None:
    """Use the full available token-ID range, including added HF tokens."""
    sizes: list[int] = []
    for attr in ("n_vocab", "vocab_size"):
        value = getattr(tokenizer, attr, None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            sizes.append(value)
    try:
        size = len(tokenizer)
        if isinstance(size, int) and size > 0:
            sizes.append(size)
    except TypeError:
        pass
    return max(sizes) if sizes else None


def special_token_ids(tokenizer: Any) -> frozenset[int]:
    ids = set(getattr(tokenizer, "all_special_ids", ()) or ())
    # tiktoken uses a mapping of special spellings to IDs.
    mapping = getattr(tokenizer, "_special_tokens", {})
    if isinstance(mapping, dict):
        ids.update(mapping.values())
    return frozenset(_integer(value, "special token ID") for value in ids)


def encode_text(tokenizer: Tokenizer, text: str) -> tuple[int, ...]:
    """Support both HuggingFace's API and tiktoken's smaller encode API."""
    try:
        values = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        values = tokenizer.encode(text)
    if hasattr(values, "tolist"):
        values = values.tolist()
    if not isinstance(values, (list, tuple)):
        raise ValueError("Tokenizer must return a flat token-ID sequence")
    return tuple(_integer(value, "token ID") for value in values)


@dataclass(frozen=True)
class TokenPhrase:
    token_ids: tuple[int, ...]
    bias: float
    text: str = ""


@dataclass(frozen=True)
class PhraseBiasPlan:
    tokenizer_id: str
    max_bias: float
    phrases: tuple[TokenPhrase, ...]
    vocab_size: int | None = None
    allow_positive: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.tokenizer_id, str) or not self.tokenizer_id.strip():
            raise ValueError("tokenizer_id is required")
        if len(self.tokenizer_id) > 512:
            raise ValueError("tokenizer_id is too long")
        maximum = _weight(self.max_bias, "max_bias", 3.0)
        if maximum <= 0:
            raise ValueError("max_bias must be positive")
        if not isinstance(self.allow_positive, bool):
            raise ValueError("allow_positive must be a boolean")
        if self.vocab_size is not None:
            _integer(self.vocab_size, "vocab_size", minimum=1)
        if not isinstance(self.phrases, tuple) or len(self.phrases) > 4096:
            raise ValueError("phrases must be a tuple with at most 4096 entries")
        seen: set[tuple[int, ...]] = set()
        for phrase in self.phrases:
            if not isinstance(phrase, TokenPhrase):
                raise ValueError("Invalid phrase entry")
            ids = phrase.token_ids
            if not isinstance(ids, tuple) or not ids or len(ids) > 256:
                raise ValueError("Each phrase needs 1..256 token IDs")
            for token in ids:
                _integer(token, "token ID")
                if self.vocab_size is not None and token >= self.vocab_size:
                    raise ValueError("Phrase token ID exceeds vocabulary")
            if ids in seen:
                raise ValueError("Duplicate token sequence")
            seen.add(ids)
            bias = _weight(phrase.bias, "phrase bias", maximum)
            if bias > 0 and not self.allow_positive:
                raise ValueError("Positive boosts require allow_positive=True")
            if not isinstance(phrase.text, str) or len(phrase.text) > 1024:
                raise ValueError("Invalid phrase text")

    def validate(self, tokenizer_id: str, vocab_size: int) -> None:
        """Fail closed rather than apply token IDs to a different vocabulary."""
        _integer(vocab_size, "vocab_size", minimum=1)
        if tokenizer_id != self.tokenizer_id:
            raise ValueError("Tokenizer identity does not match the phrase plan")
        # Some models pad their output embedding table beyond len(tokenizer).
        # That is safe; a model with fewer output slots is not.
        if self.vocab_size is not None and self.vocab_size > vocab_size:
            raise ValueError("Vocabulary size is smaller than the phrase plan")
        if any(token >= vocab_size for p in self.phrases for token in p.token_ids):
            raise ValueError("Phrase token ID exceeds model vocabulary")

    def biases(self, generated_ids: Sequence[int]) -> dict[int, float]:
        """Return next-token adjustments from one sequence's generated history."""
        history = tuple(generated_ids)
        result: dict[int, float] = {}
        for phrase in self.phrases:
            prefix, target = phrase.token_ids[:-1], phrase.token_ids[-1]
            if prefix and (len(history) < len(prefix) or history[-len(prefix) :] != prefix):
                continue
            if phrase.bias > 0:
                # A boost must not encourage immediately repeating its phrase.
                repeated = phrase.token_ids + prefix
                if len(history) >= len(repeated) and history[-len(repeated) :] == repeated:
                    continue
            result[target] = result.get(target, 0.0) + phrase.bias
        return {
            token: max(-self.max_bias, min(self.max_bias, bias))
            for token, bias in sorted(result.items())
            if bias
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "phrase_completion",
            "tokenizer_id": self.tokenizer_id,
            "vocab_size": self.vocab_size,
            "max_bias": self.max_bias,
            "allow_positive": self.allow_positive,
            "phrases": [
                {"text": p.text, "token_ids": list(p.token_ids), "bias": p.bias}
                for p in self.phrases
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PhraseBiasPlan:
        if not isinstance(data, dict) or type(data.get("schema_version")) is not int or data.get("schema_version") != 1:
            raise ValueError("Unsupported phrase plan schema")
        if data.get("kind") != "phrase_completion":
            raise ValueError("Expected a phrase_completion plan")
        entries = data.get("phrases")
        if not isinstance(entries, list) or len(entries) > 4096:
            raise ValueError("phrases must be a list with at most 4096 entries")
        phrases: list[TokenPhrase] = []
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("token_ids"), list):
                raise ValueError("Invalid phrase entry")
            if "bias" not in entry:
                raise ValueError("Phrase bias is required")
            phrases.append(TokenPhrase(tuple(entry["token_ids"]), entry["bias"], entry.get("text", "")))
        return cls(
            tokenizer_id=data.get("tokenizer_id"),
            max_bias=data.get("max_bias"),
            phrases=tuple(phrases),
            vocab_size=data.get("vocab_size"),
            allow_positive=data.get("allow_positive", False),
        )


def compile_plan(
    profile: StyleProfile,
    tokenizer: Tokenizer,
    tokenizer_id: str,
    *,
    allow_positive: bool = False,
) -> PhraseBiasPlan:
    """Compile exact tokenizer spellings; no language-independent token blacklist."""
    vocab_size = tokenizer_vocab_size(tokenizer)
    special = special_token_ids(tokenizer)
    weights: dict[tuple[int, ...], float] = {}
    labels: dict[tuple[int, ...], str] = {}
    for rule in profile.rules:
        if rule.bias > 0 and not allow_positive:
            raise ValueError("Positive boosts require allow_positive=True")
        # Encoding a leading space matters for tokenizers with word-boundary merges.
        variants = {encode_text(tokenizer, rule.text), encode_text(tokenizer, " " + rule.text)}
        # A simple leading space token creates a redundant longer suffix. Keep
        # only the shortest equivalent match so one rule never counts twice.
        variants = {
            ids for ids in variants
            if not any(
                shorter and len(shorter) < len(ids) and ids[-len(shorter) :] == shorter
                for shorter in variants
            )
        }
        for ids in variants:
            if not ids:
                raise ValueError("A phrase encoded to an empty sequence")
            if any(token in special for token in ids):
                raise ValueError("Style phrases must not contain special tokens")
            weights[ids] = weights.get(ids, 0.0) + rule.bias
            labels.setdefault(ids, rule.text)
    phrases = tuple(
        TokenPhrase(ids, max(-profile.max_bias, min(profile.max_bias, bias)), labels[ids])
        for ids, bias in sorted(weights.items())
        if bias
    )
    return PhraseBiasPlan(tokenizer_id, profile.max_bias, phrases, vocab_size, allow_positive)
