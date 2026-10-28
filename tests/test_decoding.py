import math

import pytest

from plainvoice.decoding import PhraseBiasPlan, TokenPhrase, compile_plan
from plainvoice.profiles import PhraseRule, StyleProfile


class CharacterTokenizer:
    vocab_size = 128
    all_special_ids = [127]

    def encode(self, text, **kwargs):
        return [ord(char) for char in text]

    def decode(self, tokens):
        return "".join(chr(token) for token in tokens)


def plan(*phrases, allow_positive=False, max_bias=2.0):
    return PhraseBiasPlan("test-v1", max_bias, tuple(phrases), 128, allow_positive)


def test_multitoken_penalty_only_applies_to_phrase_completion():
    steering = plan(TokenPhrase((1, 2, 3), -1.5))
    assert steering.biases([]) == {}
    assert steering.biases([1]) == {}
    assert steering.biases([9, 1, 2]) == {3: -1.5}
    assert steering.biases([1, 9, 2]) == {}
    assert steering.biases([1, 2, 3]) == {}


def test_shared_characters_are_not_globally_blacklisted():
    steering = plan(TokenPhrase((10, 11, 12), -1.0), TokenPhrase((10, 11, 13), -0.5))
    assert steering.biases([10]) == {}
    assert steering.biases([10, 11]) == {12: -1.0, 13: -0.5}
    assert steering.biases([88, 11]) == {}


def test_overlapping_suffixes_accumulate_with_a_hard_bound():
    steering = plan(TokenPhrase((1, 2, 3), -1.5), TokenPhrase((2, 3), -1.5))
    assert steering.biases([1, 2]) == {3: -2.0}


def test_positive_boost_is_opt_in_and_does_not_encourage_immediate_repetition():
    with pytest.raises(ValueError, match="Positive"):
        plan(TokenPhrase((1, 2), 0.5))
    steering = plan(TokenPhrase((1, 2), 0.5), TokenPhrase((4,), 0.5), allow_positive=True)
    assert steering.biases([1])[2] == 0.5
    assert 2 not in steering.biases([1, 2, 1])
    assert 4 not in steering.biases([4])
    assert steering.biases([1, 2, 9, 1])[2] == 0.5


def test_plan_roundtrip_preserves_behavior_and_metadata():
    original = plan(TokenPhrase((1, 2), -0.5, "phrase"))
    restored = PhraseBiasPlan.from_dict(original.to_dict())
    assert restored == original
    assert restored.biases([1]) == {2: -0.5}
    restored.validate("test-v1", 128)
    restored.validate("test-v1", 256)  # Model output embeddings may be padded.
    with pytest.raises(ValueError, match="identity"):
        restored.validate("other-tokenizer", 128)
    with pytest.raises(ValueError, match="Vocabulary"):
        restored.validate("test-v1", 64)


@pytest.mark.parametrize("ids,bias", [((), -1), ((-1,), -1), ((True,), -1), ((128,), -1), ((1,), math.nan), ((1,), math.inf), ((1,), -4)])
def test_invalid_token_sequences_and_weights_fail_closed(ids, bias):
    with pytest.raises(ValueError):
        plan(TokenPhrase(ids, bias))


def test_duplicate_encoded_sequences_are_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        plan(TokenPhrase((1, 2), -1), TokenPhrase((1, 2), -0.5))


def test_compile_uses_actual_word_boundary_encodings():
    tokenizer = CharacterTokenizer()
    profile = StyleProfile("test", "test", (PhraseRule("ab", -1.0),))
    compiled = compile_plan(profile, tokenizer, "chars")
    assert {p.token_ids for p in compiled.phrases} == {(97, 98)}
    assert compiled.biases([97]) == {98: -1.0}
    assert compiled.biases([32, 97]) == {98: -1.0}
    assert compiled.biases([]) == {}


def test_compile_keeps_nonredundant_word_boundary_merges():
    class BoundaryTokenizer:
        n_vocab = 10

        def encode(self, text, **kwargs):
            return [5, 2] if text.startswith(" ") else [1, 2]

    profile = StyleProfile("test", "test", (PhraseRule("ab", -1.0),))
    compiled = compile_plan(profile, BoundaryTokenizer(), "boundary")
    assert {p.token_ids for p in compiled.phrases} == {(1, 2), (5, 2)}
    assert compiled.biases([1]) == {2: -1.0}
    assert compiled.biases([5]) == {2: -1.0}


def test_compile_tiktoken_style_signature_and_deduplicated_variants():
    class NoKeywordTokenizer:
        n_vocab = 10

        def encode(self, text):
            return [1, 2]  # Both spellings normalize identically.

    profile = StyleProfile("test", "test", (PhraseRule("ab", -1.0),))
    compiled = compile_plan(profile, NoKeywordTokenizer(), "normalized")
    assert len(compiled.phrases) == 1
    assert compiled.biases([1]) == {2: -1.0}


@pytest.mark.parametrize("encoded", [[127], []])
def test_special_tokens_and_empty_encodings_are_not_accepted(encoded):
    profile = StyleProfile("test", "test", (PhraseRule("ab", -1.0),))
    class UnsafeTokenizer(CharacterTokenizer):
        def encode(self, text, **kwargs):
            return encoded

    with pytest.raises(ValueError):
        compile_plan(profile, UnsafeTokenizer(), "unsafe")


def test_compiled_profile_positive_boost_requires_explicit_opt_in():
    profile = StyleProfile("test", "test", (PhraseRule("ab", 0.5),))
    with pytest.raises(ValueError, match="Positive"):
        compile_plan(profile, CharacterTokenizer(), "chars")
    assert compile_plan(profile, CharacterTokenizer(), "chars", allow_positive=True).allow_positive


def test_real_tiktoken_api_with_constructed_byte_vocabulary_needs_no_downloads():
    """Exercise the actual library, without claiming a pretrained model vocabulary."""
    tiktoken = pytest.importorskip("tiktoken", reason="Install the tokens extra")
    tokenizer = tiktoken.Encoding(
        name="plainvoice-byte-test-v1",
        pat_str=r"(?s).",
        mergeable_ranks={bytes([value]): value for value in range(256)},
        special_tokens={},
    )
    profile = StyleProfile("test-zh", "test", (PhraseRule("值得注意的是", -1.0),))
    compiled = compile_plan(profile, tokenizer, tokenizer.name)
    ids = tokenizer.encode("值得注意的是")
    assert len(ids) > 1
    assert compiled.biases(ids[:-1]) == {ids[-1]: -1.0}
    assert compiled.biases(tokenizer.encode("注意安全")) == {}
    compiled.validate(tokenizer.name, tokenizer.n_vocab)
