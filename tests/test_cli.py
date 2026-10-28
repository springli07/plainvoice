import json
import os
import subprocess
import sys

import pytest

from plainvoice import cli


@pytest.fixture
def cli_run(tmp_path):
    process_environment = {name: value for name, value in os.environ.items()
                           if not name.startswith("PLAINVOICE_")}

    def run(*args, text=None):
        return subprocess.run(
            [sys.executable, "-m", "plainvoice", *args], input=text,
            text=True, capture_output=True, cwd=tmp_path, env=process_environment,
            timeout=15, check=False,
        )

    return run


def test_cli_reads_utf8_stdin_and_file_without_modifying_source(cli_run, tmp_path):
    source = "不难看出，成本可能为 12 万元。"
    stdin = cli_run("analyze", "-", "--json", text=source)
    assert stdin.returncode == 0, stdin.stderr
    assert json.loads(stdin.stdout)["character_count"] == len(source)
    path = tmp_path / "输入.txt"
    path.write_text(source, encoding="utf-8")
    file = cli_run("analyze", str(path), "--json")
    assert file.returncode == 0, file.stderr
    assert json.loads(file.stdout) == json.loads(stdin.stdout)
    assert path.read_text(encoding="utf-8") == source


@pytest.mark.parametrize("source", ["", " \n", "文" * 50_001])
def test_cli_rejects_empty_or_oversized_stdin(cli_run, source):
    result = cli_run("analyze", "--json", text=source)
    assert result.returncode == 2
    assert result.stdout == ""
    assert "Traceback" not in result.stderr


def test_cli_missing_file_and_model_are_actionable_errors(cli_run, tmp_path):
    missing = cli_run("analyze", str(tmp_path / "missing.txt"))
    assert missing.returncode == 2 and "Traceback" not in missing.stderr
    unconfigured = cli_run("rewrite", "--model", "test-model", text="原文。")
    assert unconfigured.returncode == 2
    assert "PLAINVOICE_API_KEY" in unconfigured.stderr
    assert unconfigured.stdout == ""


def test_demo_reports_mechanism_without_fabricated_generation(cli_run):
    result = cli_run("demo", "--json")
    assert result.returncode == 0, result.stderr
    demonstration = json.loads(result.stdout)
    assert demonstration["demo"] == "mechanism-only"
    assert demonstration["without_prefix"] == {}
    assert demonstration["with_prefix"] == {"3": -1.0}
    assert "text" not in demonstration


class ToyTokenizer:
    vocab_size = 256

    def encode(self, text, **kwargs):
        return [ord(char) for char in text]


def test_compile_creates_reusable_artifact_and_refuses_overwrite(monkeypatch, tmp_path, capsys):
    from plainvoice.decoding import PhraseBiasPlan
    from plainvoice.profiles import PhraseRule, StyleProfile, profile_to_dict

    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps(profile_to_dict(
        StyleProfile("test", "", (PhraseRule("ab", -0.5),)),
    )), encoding="utf-8")
    output = tmp_path / "plan.json"
    monkeypatch.setattr(cli, "_tokenizer", lambda name: ToyTokenizer())
    arguments = ["compile", "--profile", str(profile), "--tokenizer", "toy", "--out", str(output)]
    assert cli.main(arguments) == 0
    original = output.read_bytes()
    restored = PhraseBiasPlan.from_dict(json.loads(original))
    assert restored.biases([ord("a")]) == {ord("b"): -0.5}
    assert cli.main(arguments) == 2
    assert output.read_bytes() == original
    assert "Traceback" not in capsys.readouterr().err


def test_compile_cannot_overwrite_its_input_profile(monkeypatch, tmp_path):
    from plainvoice.profiles import PhraseRule, StyleProfile, profile_to_dict

    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps(profile_to_dict(
        StyleProfile("test", "", (PhraseRule("ab", -0.5),)),
    )), encoding="utf-8")
    original = profile.read_bytes()
    monkeypatch.setattr(cli, "_tokenizer", lambda name: ToyTokenizer())
    assert cli.main(["compile", "--profile", str(profile), "--out", str(profile)]) == 2
    assert profile.read_bytes() == original
