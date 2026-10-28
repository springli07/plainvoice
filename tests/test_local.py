"""Processor tests use a tiny tensor stand-in; real torch integration is optional."""

import copy
import math

import pytest

from plainvoice.decoding import PhraseBiasPlan, TokenPhrase
from plainvoice.local import PhraseLogitsProcessor


class Row:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return list(self.values)


class Matrix:
    def __init__(self, values):
        self.values = values
        self.shape = (len(values), len(values[0]))

    def clone(self):
        return Matrix(copy.deepcopy(self.values))

    def __getitem__(self, index):
        row, col = index
        result = self.values[row][col]
        return Row(result) if isinstance(col, slice) else result

    def __setitem__(self, index, value):
        row, col = index
        self.values[row][col] = value


def processor(*, table=None, prompt_length=2):
    plan = PhraseBiasPlan("test-v1", 2.0, (TokenPhrase((1, 2), -1.5),), 10)
    return PhraseLogitsProcessor(plan, prompt_length, "test-v1", 10, table)


def test_batch_histories_are_independent_and_original_scores_are_not_mutated():
    ids = Matrix([[8, 8, 9, 1], [8, 8, 9, 4]])
    scores = Matrix([[0.0] * 10, [0.0] * 10])
    adjusted = processor()(ids, scores)
    assert adjusted[0, 2] == -1.5
    assert adjusted[1, 2] == 0.0
    assert scores[0, 2] == 0.0


def test_prompt_suffix_cannot_trigger_generated_phrase_penalty():
    adjusted = processor()(Matrix([[8, 1]]), Matrix([[0.0] * 10]))
    assert adjusted[0, 2] == 0.0


def test_static_and_context_weights_are_combined_then_clamped():
    table = {"schema_version": 1, "kind": "token_log_odds", "tokenizer_id": "test-v1", "vocab_size": 10, "max_bias": 1.0, "biases": {"2": -1.0}}
    adjusted = processor(table=table)(Matrix([[8, 8, 1]]), Matrix([[0.0] * 10]))
    assert adjusted[0, 2] == -2.0


def test_positive_static_boost_does_not_encourage_immediate_token_repeat():
    table = {"schema_version": 1, "kind": "token_log_odds", "tokenizer_id": "test-v1", "vocab_size": 10, "max_bias": 1.0, "allow_positive": True, "biases": {"3": 0.8}}
    adjusted = processor(table=table)(Matrix([[8, 8, 3]]), Matrix([[0.0] * 10]))
    assert adjusted[0, 3] == 0.0
    adjusted = processor(table=table)(Matrix([[8, 8, 4]]), Matrix([[0.0] * 10]))
    assert adjusted[0, 3] == 0.8


def test_existing_model_masks_remain_masked():
    values = [[0.0] * 10]
    values[0][2] = float("-inf")
    adjusted = processor()(Matrix([[8, 8, 1]]), Matrix(values))
    assert adjusted[0, 2] == float("-inf")


def test_contextual_logit_bias_changes_target_probability_and_preserves_other_odds():
    scores = Matrix([[0.0] * 10])
    adjusted = processor()(Matrix([[8, 8, 1]]), scores)
    probabilities = [math.exp(value) for value in adjusted.values[0]]
    total = sum(probabilities)
    probabilities = [value / total for value in probabilities]
    assert probabilities[2] < 0.1
    assert probabilities[3] / probabilities[4] == 1.0
    unchanged = processor()(Matrix([[8, 8, 4]]), scores)
    assert unchanged.values == scores.values


def test_mismatched_dimensions_and_vocabulary_are_rejected():
    with pytest.raises(ValueError, match="dimensions"):
        processor()(Matrix([[8, 8]]), Matrix([[0.0] * 9]))
    with pytest.raises(ValueError, match="shorter"):
        processor()(Matrix([[8]]), Matrix([[0.0] * 10]))
    plan = PhraseBiasPlan("test-v1", 2.0, (), 10)
    with pytest.raises(ValueError, match="identity"):
        PhraseLogitsProcessor(plan, 0, "other", 10)


def test_real_torch_logits_processor_if_installed():
    torch = pytest.importorskip("torch", reason="Install the local extra for real tensor integration")
    ids = torch.tensor([[8, 8, 1], [8, 8, 4]])
    scores = torch.zeros((2, 10))
    adjusted = processor()(ids, scores)
    assert adjusted[0, 2].item() == -1.5
    assert adjusted[1, 2].item() == 0.0
    assert scores[0, 2].item() == 0.0
