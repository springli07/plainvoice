"""Estimate bounded token preferences from user-supplied comparison corpora."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from typing import Any

from .decoding import _integer, _weight, encode_text, special_token_ids, tokenizer_vocab_size


def fit_token_bias(
    human_texts: Sequence[str],
    model_texts: Sequence[str],
    tokenizer: Any,
    tokenizer_id: str,
    *,
    min_count: int = 5,
    smoothing: float = 1.0,
    max_bias: float = 1.0,
    allow_positive: bool = False,
) -> dict[str, Any]:
    """Fit log smoothed frequency ratios, not authorship probabilities.

    Corpora should be paired by topic, genre, and length; this function cannot
    remove those confounders. Keep separate held-out text for evaluation.
    """
    _integer(min_count, "min_count", minimum=1)
    maximum = _weight(max_bias, "max_bias", 3.0)
    if maximum <= 0:
        raise ValueError("max_bias must be positive")
    if isinstance(smoothing, bool) or not isinstance(smoothing, (int, float)):
        raise ValueError("smoothing must be a positive finite number")
    if not math.isfinite(smoothing) or smoothing <= 0:
        raise ValueError("smoothing must be a positive finite number")
    if not isinstance(allow_positive, bool):
        raise ValueError("allow_positive must be a boolean")
    if not isinstance(tokenizer_id, str) or not tokenizer_id.strip() or len(tokenizer_id) > 512:
        raise ValueError("tokenizer_id is required and must be <= 512 characters")
    if not callable(getattr(tokenizer, "decode", None)):
        raise ValueError("Corpus tokenizer must support decode for whitespace filtering")
    specials = special_token_ids(tokenizer)
    vocab_size = tokenizer_vocab_size(tokenizer)
    excluded: set[int] = set(specials)
    usable: set[int] = set()

    def count(texts: Sequence[str], label: str) -> tuple[Counter[int], int]:
        if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence) or not texts:
            raise ValueError(f"{label} corpus must be a nonempty text sequence")
        counts: Counter[int] = Counter()
        for text in texts:
            if not isinstance(text, str):
                raise ValueError(f"{label} corpus contains a non-text document")
            for token in encode_text(tokenizer, text):
                if vocab_size is not None and token >= vocab_size:
                    raise ValueError("Tokenizer returned an out-of-vocabulary token")
                if token in excluded:
                    continue
                if token not in usable:
                    decoded = tokenizer.decode([token])
                    if not isinstance(decoded, str):
                        raise ValueError("Tokenizer.decode must return text")
                    if not decoded or decoded.isspace():
                        excluded.add(token)
                        continue
                    usable.add(token)
                counts[token] += 1
        if not counts:
            raise ValueError(f"{label} corpus has no usable tokens")
        return counts, len(texts)

    human, human_documents = count(human_texts, "human")
    model, model_documents = count(model_texts, "model")
    vocabulary = sorted(human.keys() | model.keys())
    human_total, model_total = sum(human.values()), sum(model.values())
    human_denominator = human_total + smoothing * len(vocabulary)
    model_denominator = model_total + smoothing * len(vocabulary)
    weights: dict[str, float] = {}
    for token in vocabulary:
        if human[token] + model[token] < min_count:
            continue
        human_rate = (human[token] + smoothing) / human_denominator
        model_rate = (model[token] + smoothing) / model_denominator
        bias = max(-maximum, min(maximum, math.log(human_rate / model_rate)))
        if bias > 0 and not allow_positive:
            continue
        if abs(bias) > 1e-12:
            weights[str(token)] = round(bias, 8)
    return {
        "schema_version": 1,
        "kind": "token_log_odds",
        "tokenizer_id": tokenizer_id,
        "vocab_size": vocab_size,
        "max_bias": maximum,
        "allow_positive": allow_positive,
        "biases": weights,
        "metadata": {
            "human_documents": human_documents,
            "model_documents": model_documents,
            "human_tokens": human_total,
            "model_tokens": model_total,
            "observed_vocabulary": len(vocabulary),
            "min_count": min_count,
            "smoothing": smoothing,
            "method": "log smoothed normalized token-frequency ratio",
            "limitations": "Corpus/topic dependent; not an authorship classifier or proof of human preference.",
        },
    }


def validate_token_bias(
    table: dict[str, Any], tokenizer_id: str, vocab_size: int | None = None
) -> dict[int, float]:
    """Validate a persisted table against the exact declared tokenizer."""
    if not isinstance(table, dict) or type(table.get("schema_version")) is not int or table.get("schema_version") != 1:
        raise ValueError("Unsupported token-bias schema")
    if table.get("kind") != "token_log_odds":
        raise ValueError("Expected a token_log_odds table")
    if not isinstance(tokenizer_id, str) or not tokenizer_id.strip():
        raise ValueError("tokenizer_id is required")
    if table.get("tokenizer_id") != tokenizer_id:
        raise ValueError("Tokenizer identity does not match the token-bias table")
    saved_vocab = table.get("vocab_size")
    if saved_vocab is not None:
        _integer(saved_vocab, "table vocab_size", minimum=1)
    if vocab_size is not None:
        _integer(vocab_size, "vocab_size", minimum=1)
        if saved_vocab is not None and saved_vocab > vocab_size:
            raise ValueError("Vocabulary size is smaller than the token-bias table")
    maximum = _weight(table.get("max_bias"), "max_bias", 3.0)
    if maximum <= 0:
        raise ValueError("max_bias must be positive")
    allow_positive = table.get("allow_positive", False)
    if not isinstance(allow_positive, bool):
        raise ValueError("allow_positive must be a boolean")
    weights = table.get("biases")
    if not isinstance(weights, dict) or len(weights) > 100_000:
        raise ValueError("biases must be an object with <= 100000 entries")
    result: dict[int, float] = {}
    limits = [size for size in (vocab_size, saved_vocab) if size is not None]
    limit = min(limits) if limits else None
    for key, value in weights.items():
        if not isinstance(key, str) or not key.isascii() or not key.isdigit() or str(int(key)) != key:
            raise ValueError("Token IDs must be canonical nonnegative decimal strings")
        token = int(key)
        if limit is not None and token >= limit:
            raise ValueError("Token ID exceeds vocabulary")
        bias = _weight(value, "token bias", maximum)
        if bias > 0 and not allow_positive:
            raise ValueError("Positive token bias requires explicit opt-in metadata")
        if bias:
            result[token] = bias
    return result
