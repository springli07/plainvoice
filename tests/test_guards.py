"""Meaningful negative cases for local preservation checks."""

import pytest

from plainvoice.guards import preservation_issues, protected_spans


def test_protected_offsets_cover_code_and_quotes_only():
    text = '外面。`n = 12`，他说“保持原话”。\n```python\nprint("x")\n```\n尾声。'
    chunks = [text[start:end] for start, end in protected_spans(text)]
    assert chunks == ['`n = 12`', '“保持原话”', '```python\nprint("x")\n```']


def test_fence_marker_must_match_and_unclosed_fence_is_protected():
    text = "开头\n~~~~text\n``` unrelated\nno closing fence"
    assert [text[a:b] for a, b in protected_spans(text)] == [
        "~~~~text\n``` unrelated\nno closing fence"
    ]


def test_apostrophes_are_not_quoted_passages():
    text = "I don't think it's wrong. 'Keep this.'"
    assert [text[a:b] for a, b in protected_spans(text)] == ["'Keep this.'"]
    assert preservation_issues("I don't accept it.", "I accept it.") == [
        "Negation markers changed; review the meaning."
    ]


def test_quotes_with_embedded_apostrophe_and_escapes():
    text = '''He said "don't change it"; 'don\'t change it' and “嵌套‘引语’”。'''
    chunks = [text[a:b] for a, b in protected_spans(text)]
    assert chunks == ['"don\'t change it"', "'don't change it'", '“嵌套‘引语’”']


@pytest.mark.parametrize(
    ("source", "candidate"),
    [
        ("费用是 10 万元。", "费用是 10 元。"),
        ("收入约 2.5 亿元。", "收入约 2.8 亿元。"),
        ("增长 15%。", "增长 15。"),
        ("容量 16 MB。", "容量 16 Mb。"),
        ("He waited 2 days.", "He waited 2 months."),
        ("已测试。", "已测试 20 次。"),
        ("三人到场。", "二人到场。"),
        ("每次 5 元，共付 5 元。", "每次 5 元。"),
    ],
)
def test_changed_or_added_numbers_and_units_are_rejected(source, candidate):
    assert "Numeric facts or units changed." in preservation_issues(source, candidate)


def test_number_unit_whitespace_can_change_without_changing_fact():
    assert preservation_issues("增长 15 %，收入 10 万元。", "增长15%，收入10万元。") == []


def test_quote_and_code_changes_are_separate_from_numeric_changes():
    issues = preservation_issues('他说“12 人”。代码 `x = 2`。', '他说“15 人”。代码 `x = 3`。')
    assert issues == ["Code blocks or inline code changed.", "Quoted material changed."]


def test_unchanged_quoted_facts_can_move_without_false_failure():
    assert preservation_issues('他说“12 人”。实际有 15 人。', '实际有15人。他说“12 人”。') == []


def test_duplicate_quoted_fact_cannot_be_silently_dropped():
    assert preservation_issues('“同意”然后又说“同意”。', '他说“同意”。') == ["Quoted material changed."]


def test_urls_keep_query_and_numbers_and_ignore_sentence_punctuation():
    source = "见 https://example.com/x?n=12。"
    assert preservation_issues(source, "详情见 https://example.com/x?n=12。") == []
    assert preservation_issues(source, "详情见 https://example.com/x?n=13。") == ["URLs changed."]


def test_balanced_url_parentheses_are_kept():
    source = "见 (https://example.com/wiki/Foo_(bar))。"
    assert preservation_issues(source, "链接 https://example.com/wiki/Foo_(bar)。") == []


@pytest.mark.parametrize(
    ("source", "candidate", "expected"),
    [
        ("这项方案尚未完成。", "这项方案已经完成。", "Negation markers changed; review the meaning."),
        ("这项方案可能有效。", "这项方案有效。", "Qualification markers changed; review the meaning."),
        ("至少 5 人参加。", "5 人参加。", "Qualification markers changed; review the meaning."),
        ("It may help.", "It will help.", "Qualification markers changed; review the meaning."),
        ("It is NOT safe.", "It is safe.", "Negation markers changed; review the meaning."),
    ],
)
def test_qualifications_and_negations_require_review(source, candidate, expected):
    assert expected in preservation_issues(source, candidate)


def test_guard_messages_do_not_echo_sensitive_source_content():
    source = 'The private password is "example-secret-42".'
    issues = preservation_issues(source, 'The private password is "different".')
    assert issues == ["Quoted material changed."]
    assert "example-secret" not in " ".join(issues)


def test_surface_guard_does_not_pretend_to_prove_equivalence():
    # Attribution changes are beyond these lexical checks. This limitation is
    # explicit in the public API docstring rather than hidden by a confidence score.
    assert preservation_issues("甲欠乙 5 元。", "乙欠甲 5 元。") == []


def test_removing_nonnegating_stock_introduction_is_allowed():
    assert preservation_issues("不难看出，这个版本支持导出。", "这个版本支持导出。") == []
    assert preservation_issues("这个版本不支持导出。", "这个版本支持导出。") == [
        "Negation markers changed; review the meaning."
    ]


def test_non_string_inputs_fail_with_clear_type_error():
    with pytest.raises(TypeError):
        protected_spans(None)
    with pytest.raises(TypeError):
        preservation_issues("text", 42)


def test_unmatched_outer_quotes_still_allow_completed_inner_quote():
    text = "“外“内”"
    assert protected_spans(text) == [(2, 5)]
    # Different paired delimiters retain the parser's earliest-opening policy.
    assert protected_spans("“外「内”后」") == [(0, 5)]


def test_quote_escaping_uses_odd_backslash_parity():
    assert protected_spans('\\“内”') == []
    assert protected_spans('\\\\“内”') == [(2, 5)]
    assert protected_spans('\\\\\\“内”') == []


def test_maximum_length_unbalanced_quotes_do_not_rescan_each_opener():
    # These are API-sized adversarial inputs, not a machine-specific speed
    # assertion. No closing quote means no protected range may be invented.
    assert protected_spans("“" * 50_000) == []
    assert protected_spans("“" * 49_999 + "”") == [(49_998, 50_000)]
    assert protected_spans("'a" * 25_000) == []


def test_long_backslash_run_preserves_the_following_quote_offsets():
    text = "\\" * 49_996 + "“尾”"
    assert protected_spans(text) == [(49_996, 49_999)]


def test_many_distinct_inline_code_widths_can_find_a_late_matching_pair():
    prefix = "prefix " + "".join("`" * width + "x" for width in range(1, 300))
    block = "`" * 300 + "payload" + "`" * 300
    assert protected_spans(prefix + block) == [(len(prefix), len(prefix + block))]


def test_large_unclosed_fence_protects_through_the_end():
    text = "```text\n" + "``x\n" * 10_000
    assert protected_spans(text) == [(0, len(text))]


def test_long_unitless_numeral_runs_do_not_invent_numeric_facts():
    text = "一" * 50_000
    assert preservation_issues(text, text + "。") == []


def test_long_url_parenthesis_tail_is_trimmed_without_changing_url():
    text = "https://example.com/path" + ")" * 49_970
    assert preservation_issues(text, "https://example.com/path") == []
