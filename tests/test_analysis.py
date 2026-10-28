import math

import pytest

from plainvoice.analysis import analyze
from plainvoice.profiles import PhraseRule, StyleProfile


def profile(*rules):
    return StyleProfile("test", "", tuple(PhraseRule(*rule) for rule in rules))


def test_exact_unicode_offsets_and_repeated_occurrences():
    text = "🙂 前言。不难看出，方案可行。\n\n不难看出，仍须测试。"
    result = analyze(text, profile(("不难看出", -0.5)))
    assert len(result["findings"]) == 2
    for finding in result["findings"]:
        assert text[finding["start"]:finding["end"]] == finding["text"] == "不难看出"
    assert result["findings"][0]["start"] == text.index("不难看出")
    assert result["metrics"]["paragraph_count"] == 2


def test_protected_material_does_not_trigger_preferences():
    text = '“不难看出”是原话。`不难看出` 是代码。\n```text\n不难看出\n```\n不难看出，试验结束。'
    result = analyze(text, profile(("不难看出", -0.5)))
    assert len(result["findings"]) == 1
    assert result["findings"][0]["start"] == text.rindex("不难看出")


def test_positive_preferences_are_not_reported_as_problems():
    result = analyze("直接说具体变化。", profile(("具体变化", 0.4)))
    assert result["findings"] == []


def test_overlapping_literal_phrases_keep_exact_spans():
    result = analyze("a+b+c", profile(("a+b", -0.5), ("b+c", -0.4)))
    assert [item["text"] for item in result["findings"]] == ["a+b", "b+c"]


@pytest.mark.parametrize("text", ["", " \n\n ", "`code`", "“原话”"])
def test_empty_or_protected_only_text_produces_finite_descriptive_metrics(text):
    result = analyze(text)
    assert result["findings"] == []
    assert all(math.isfinite(value) for value in result["metrics"].values())
    assert result["metrics"]["sentence_count"] == 0
    assert "不是 AI 创作概率" in result["disclaimer"]
    assert not any("score" in key or "probability" in key for key in result)


def test_sentence_metrics_avoid_splitting_decimal_values():
    result = analyze("售价 1.25 元。仍待验证！最后一段")
    assert result["metrics"]["sentence_count"] == 3
    assert analyze("Run 1. Run 2.")["metrics"]["sentence_count"] == 2


def test_statistical_observations_do_not_invent_bad_style_findings():
    result = analyze("短句。短句。短句。")
    assert result["metrics"]["sentence_length_cv"] == 0
    assert result["findings"] == []


@pytest.mark.parametrize("text", [None, 123, ["text"], "x" * 50_001, "\ud800"])
def test_analysis_rejects_invalid_or_oversized_text(text):
    with pytest.raises(ValueError):
        analyze(text)
def test_many_interleaved_quotes_and_plain_hits_keep_exact_offsets():
    text = '“不难看出”不难看出。' * 3000
    result = analyze(text)
    assert len(result["findings"]) == 3000
    for finding in result["findings"]:
        assert text[finding["start"]:finding["end"]] == "不难看出"
        assert text[finding["start"] - 1] == "”"
