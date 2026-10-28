import math

import pytest

from plainvoice.corpus import fit_token_bias, validate_token_bias


class CharacterTokenizer:
    vocab_size = 128
    all_special_ids = [127]

    def encode(self, text, **kwargs):
        return [ord(char) for char in text]

    def decode(self, tokens):
        return "".join(chr(token) for token in tokens)


def fit(human, model, **kwargs):
    return fit_token_bias(human, model, CharacterTokenizer(), "chars-v1", **kwargs)


def test_smoothed_normalized_ratios_and_conservative_default():
    result = fit(["aaaaab"], ["abbbbb"], min_count=2)
    assert str(ord("a")) not in result["biases"]
    assert result["biases"][str(ord("b"))] == -1.0
    assert result["metadata"]["human_tokens"] == 6
    assert result["metadata"]["model_tokens"] == 6


def test_positive_corpus_preferences_are_explicit_and_bounded():
    result = fit(["aaaaab"], ["abbbbb"], min_count=2, allow_positive=True, max_bias=0.4)
    assert result["biases"] == {"97": 0.4, "98": -0.4}
    assert validate_token_bias(result, "chars-v1", 128) == {97: 0.4, 98: -0.4}


def test_minimum_support_and_special_whitespace_exclusion():
    result = fit(["aa c\x7f"], ["bb d\x7f"], min_count=2, allow_positive=True)
    assert "32" not in result["biases"]
    assert "127" not in result["biases"]
    assert str(ord("c")) not in result["biases"]
    assert str(ord("d")) not in result["biases"]
    assert result["metadata"]["human_tokens"] == 3


def test_frequency_normalization_avoids_document_length_bias():
    result = fit(["ab"], ["abababab"], min_count=1, allow_positive=True)
    assert result["biases"] == {}


def test_identical_inputs_are_reproducible_and_corpus_order_does_not_change_weights():
    first = fit(["aaab", "baab"], ["bbba", "abba"], min_count=1, allow_positive=True)
    second = fit(["baab", "aaab"], ["abba", "bbba"], min_count=1, allow_positive=True)
    assert first == second


@pytest.mark.parametrize("options", [{"min_count": 0}, {"min_count": True}, {"smoothing": 0}, {"smoothing": math.nan}, {"max_bias": math.inf}, {"allow_positive": "yes"}])
def test_invalid_fitting_options_are_rejected(options):
    with pytest.raises(ValueError):
        fit(["abc"], ["abc"], **options)


@pytest.mark.parametrize("human,model", [([], ["abc"]), ([" "], ["abc"]), (["abc"], []), (["abc"], [None]), ("abc", ["abc"])])
def test_empty_or_invalid_corpora_are_rejected(human, model):
    with pytest.raises(ValueError):
        fit(human, model)


def test_bias_table_rejects_tokenizer_mismatch_and_invalid_ids():
    table = fit(["aaaa"], ["bbbb"], min_count=1)
    with pytest.raises(ValueError, match="identity"):
        validate_token_bias(table, "other")
    with pytest.raises(ValueError, match="Vocabulary"):
        validate_token_bias(table, "chars-v1", 10)
    for invalid in ({"-1": -0.5}, {"01": -0.5}, {"1": math.nan}, {"128": -0.5}, {"1": 0.5}):
        bad = {**table, "biases": invalid}
        with pytest.raises(ValueError):
            validate_token_bias(bad, "chars-v1", 128)


def test_model_output_vocabulary_can_be_padded_but_pad_ids_cannot_receive_weights():
    table = fit(["aaaa"], ["bbbb"], min_count=1)
    assert validate_token_bias(table, "chars-v1", 256)
    with pytest.raises(ValueError, match="exceeds"):
        validate_token_bias({**table, "biases": {"200": -0.5}}, "chars-v1", 256)


def test_corpus_token_decoding_is_cached():
    class CountingTokenizer(CharacterTokenizer):
        calls = 0

        def decode(self, tokens):
            self.calls += 1
            return super().decode(tokens)

    tokenizer = CountingTokenizer()
    fit_token_bias(["a " * 100], ["b " * 100], tokenizer, "cached", min_count=1)
    assert tokenizer.calls == 3


def test_real_tiktoken_corpus_fit_with_constructed_byte_vocabulary():
    tiktoken = pytest.importorskip("tiktoken", reason="Install the tokens extra")
    tokenizer = tiktoken.Encoding(
        name="plainvoice-byte-test-v1",
        pat_str=r"(?s).",
        mergeable_ranks={bytes([value]): value for value in range(256)},
        special_tokens={},
    )
    table = fit_token_bias(
        ["aa自然表达"], ["bb模板表达"], tokenizer, tokenizer.name,
        min_count=2, allow_positive=True,
    )
    weights = validate_token_bias(table, tokenizer.name, tokenizer.n_vocab)
    assert weights[ord("a")] > 0
    assert weights[ord("b")] < 0
    assert ord(" ") not in weights
