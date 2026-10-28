"""Dependency-light HuggingFace-compatible logits processor.

Importing PlainVoice never imports torch, downloads models, or starts inference.
The callable accepts torch tensors when used with transformers.generate().
"""

from __future__ import annotations

from typing import Any

from .corpus import validate_token_bias
from .decoding import PhraseBiasPlan, _integer


class PhraseLogitsProcessor:
    """Apply each batch row's phrase completion and corpus preferences.

    ``prompt_length`` is the padded input width passed to generate(), so prompt
    text cannot trigger a phrase penalty in the generated answer. Beam rows are
    handled independently, with no mutable history shared across requests.
    """

    def __init__(
        self,
        plan: PhraseBiasPlan,
        prompt_length: int,
        tokenizer_id: str,
        vocab_size: int,
        token_bias: dict[str, Any] | None = None,
    ) -> None:
        _integer(prompt_length, "prompt_length")
        plan.validate(tokenizer_id, vocab_size)
        self.plan = plan
        self.prompt_length = prompt_length
        self.vocab_size = vocab_size
        self.token_bias = (
            validate_token_bias(token_bias, tokenizer_id, vocab_size) if token_bias is not None else {}
        )

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        if len(input_ids.shape) != 2 or len(scores.shape) != 2:
            raise ValueError("input_ids and scores must both be rank-2 tensors")
        if input_ids.shape[0] != scores.shape[0] or scores.shape[1] != self.vocab_size:
            raise ValueError("Batch or vocabulary dimensions do not match")
        if input_ids.shape[1] < self.prompt_length:
            raise ValueError("Input sequence is shorter than the configured prompt")
        adjusted = scores.clone()
        for row in range(input_ids.shape[0]):
            history = input_ids[row, self.prompt_length :].tolist()
            bias = self.plan.biases(history)
            for token, value in self.token_bias.items():
                if value > 0 and history and history[-1] == token:
                    continue
                bias[token] = bias.get(token, 0.0) + value
            for token, value in bias.items():
                bounded = max(-self.plan.max_bias, min(self.plan.max_bias, value))
                adjusted[row, token] += bounded
        return adjusted
