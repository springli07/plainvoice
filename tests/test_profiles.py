import json
import math

import pytest

from plainvoice.profiles import (
    PhraseRule,
    StyleProfile,
    default_profile,
    load_profile,
    profile_from_dict,
    profile_to_dict,
)


def valid_profile():
    return profile_to_dict(default_profile())


def test_profile_round_trip_preserves_preferences(tmp_path):
    data = valid_profile()
    data["rules"].append({"text": "请举一个具体例子", "bias": 0.25})
    path = tmp_path / "偏好.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    result = load_profile(path)
    assert profile_to_dict(result) == data
    assert result.rules[-1].bias == 0.25


@pytest.mark.parametrize("value", [True, False, "1", None, math.nan, math.inf, -math.inf])
def test_bias_rejects_ambiguous_or_nonfinite_values(value):
    with pytest.raises(ValueError):
        PhraseRule("词组", value)


@pytest.mark.parametrize("text", ["", "  ", "x" * 161, "段\n落", "\ud800"])
def test_phrase_rejects_unusable_text(text):
    with pytest.raises(ValueError):
        PhraseRule(text, -0.5)


def test_profile_rejects_duplicate_and_out_of_range_rules():
    with pytest.raises(ValueError, match="duplicate"):
        StyleProfile("test", "", (PhraseRule("重复", -1), PhraseRule("重复", -0.5)))
    with pytest.raises(ValueError, match="exceeds"):
        StyleProfile("test", "", (PhraseRule("太强", -2),), max_bias=1)


@pytest.mark.parametrize("value", [0, -1, 3.1, True, "2", math.nan])
def test_profile_requires_bounded_nonzero_max_bias(value):
    with pytest.raises(ValueError):
        StyleProfile("test", "", (), max_bias=value)


def test_schema_rejects_unknown_fields_bad_version_and_bad_rules():
    data = valid_profile()
    data["max_bais"] = data.pop("max_bias")
    with pytest.raises(ValueError):
        profile_from_dict(data)
    data = valid_profile()
    data["schema_version"] = True
    with pytest.raises(ValueError):
        profile_from_dict(data)
    data = valid_profile()
    data["rules"] = [{"text": "test", "bias": -0.2, "unknown": 1}]
    with pytest.raises(ValueError):
        profile_from_dict(data)


def test_load_profile_rejects_non_json_and_oversized_input(tmp_path):
    path = tmp_path / "bad.json"
    path.write_bytes(b"\xff")
    with pytest.raises(ValueError, match="UTF-8"):
        load_profile(path)
    path.write_text(" " * (256 * 1024 + 1), encoding="utf-8")
    with pytest.raises(ValueError, match="256 KiB"):
        load_profile(path)


def test_no_default_rule_boosts_a_universal_human_vocabulary():
    assert default_profile().rules
    assert all(rule.bias < 0 for rule in default_profile().rules)


def test_load_profile_rejects_duplicate_json_keys(tmp_path):
    path = tmp_path / "ambiguous.json"
    path.write_text('{"schema_version":1,"schema_version":2}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON keys"):
        load_profile(path)


@pytest.mark.parametrize("field", ["name", "description"])
def test_profile_metadata_rejects_non_unicode_surrogates(field):
    data = valid_profile()
    data[field] = "\ud800"
    with pytest.raises(ValueError):
        profile_from_dict(data)
