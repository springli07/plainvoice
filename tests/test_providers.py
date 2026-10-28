import json

import httpx
import pytest

from plainvoice import providers


@pytest.fixture(autouse=True)
def clean_model_environment(monkeypatch):
    for name in ("PLAINVOICE_API_KEY", "PLAINVOICE_MODEL", "PLAINVOICE_BASE_URL", "PLAINVOICE_BACKEND"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def mock_upstream(monkeypatch):
    original_client = httpx.Client
    requests = []

    def install(response=None, *, handler=None):
        def respond(request):
            requests.append(request)
            if handler:
                return handler(request)
            return response

        transport = httpx.MockTransport(respond)
        monkeypatch.setattr(
            providers.httpx, "Client",
            lambda **kwargs: original_client(transport=transport, trust_env=False, **kwargs),
        )
        return requests

    return install


def completion(text, finish_reason="stop"):
    return httpx.Response(200, json={
        "choices": [{"finish_reason": finish_reason, "message": {"content": text}}],
    })


def bias_table(weight=-0.5):
    return {
        "schema_version": 1, "kind": "token_log_odds", "tokenizer_id": "test-tokenizer",
        "vocab_size": 100, "max_bias": 1.0, "allow_positive": weight > 0,
        "biases": {"12": weight},
    }


def test_rewrite_preserves_user_data_and_keeps_key_out_of_results(mock_upstream):
    source = "不难看出，成本可能为 12 万元。保留 `limit=12` 和“原始报价”。"
    candidate = "成本可能为 12 万元。保留 `limit=12` 和“原始报价”。"
    requests = mock_upstream(completion(candidate))
    result = providers.rewrite(
        source, model="test-model", base_url="https://model.example/v1", api_key="private-key",
    )
    assert result["text"] == candidate
    assert result["preservation"]["passed"] is True
    assert result["steering_mode"] == "prompt-only"
    assert len(requests) == 1
    request = requests[0]
    assert str(request.url) == "https://model.example/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer private-key"
    body = json.loads(request.content)
    assert body["model"] == "test-model"
    assert body["messages"][-1] == {"role": "user", "content": source}
    assert "logit_bias" not in body
    assert "private-key" not in json.dumps(result)
    assert len(result["analysis_after"]["findings"]) < len(result["analysis_before"]["findings"])


def test_missing_remote_key_fails_before_any_network(monkeypatch):
    def forbidden_client(**kwargs):
        pytest.fail("missing credentials must not send an upstream request")

    monkeypatch.setattr(providers.httpx, "Client", forbidden_client)
    with pytest.raises(providers.BackendUnavailable, match="PLAINVOICE_API_KEY"):
        providers.rewrite("保留原文。", model="test", base_url="https://model.example/v1")


def test_explicit_loopback_backend_can_run_without_auth(mock_upstream):
    requests = mock_upstream(completion("原文。"))
    result = providers.rewrite("原文。", model="local-api", base_url="http://127.0.0.1:8080/v1")
    assert result["text"] == "原文。"
    assert "authorization" not in requests[0].headers


@pytest.mark.parametrize("endpoint", [
    "http://model.example/v1", "https://user:private-key@model.example/v1",
    "https://model.example/v1?api_key=private-key", "https://model.example/v1#private-key",
])
def test_bad_or_credential_bearing_endpoints_are_rejected_without_echoing_secrets(endpoint):
    with pytest.raises(ValueError) as caught:
        providers.rewrite("原文。", model="test", base_url=endpoint, api_key="private-key")
    assert "private-key" not in str(caught.value)


@pytest.mark.parametrize("finish_reason", ["length", "content_filter", "tool_calls"])
def test_incomplete_or_nontext_generation_is_not_returned(mock_upstream, finish_reason):
    mock_upstream(completion("部分原文。", finish_reason))
    with pytest.raises(providers.RewriteError, match="完整结束"):
        providers.rewrite("原文。", model="test", api_key="private-key")


@pytest.mark.parametrize("response", [
    httpx.Response(200, text="not-json private-key"),
    httpx.Response(200, json={"choices": []}),
    httpx.Response(200, json={"choices": [{"message": {"content": None}}]}),
    httpx.Response(401, text="private-key rejected by upstream"),
    httpx.Response(302, headers={"location": "https://other.example/"}),
])
def test_bad_upstream_responses_are_safe_failures(mock_upstream, response):
    requests = mock_upstream(response)
    with pytest.raises(providers.RewriteError) as caught:
        providers.rewrite("原文。", model="test", api_key="private-key")
    assert "private-key" not in str(caught.value)
    assert len(requests) == 1  # No retry or redirect with credentials.


@pytest.mark.parametrize("exception", [httpx.ConnectError, httpx.ReadTimeout])
def test_transport_errors_are_redacted_and_not_retried(mock_upstream, exception):
    def fail(request):
        raise exception("secret upstream URL private-key", request=request)

    requests = mock_upstream(handler=fail)
    with pytest.raises(providers.RewriteError) as caught:
        providers.rewrite("原文。", model="test", api_key="private-key")
    assert "private-key" not in str(caught.value)
    assert len(requests) == 1


@pytest.mark.parametrize("response", [
    httpx.Response(200, content=b"x" * 2_000_001),
    completion("文" * (providers.MAX_OUTPUT_CHARS + 1)),
])
def test_oversized_responses_are_rejected(mock_upstream, response):
    mock_upstream(response)
    with pytest.raises(providers.RewriteError, match="过大|长度限制"):
        providers.rewrite("原文。", model="test", api_key="private-key")


@pytest.mark.parametrize("candidate", ["预算为 13 万元。", "预算为 12 元。", "预算充足。"])
def test_factual_preservation_failure_blocks_output(mock_upstream, candidate):
    mock_upstream(completion(candidate))
    with pytest.raises(providers.RewriteError, match="保留检查未通过"):
        providers.rewrite("预算可能为 12 万元。", model="test", api_key="private-key")


def test_static_bias_requires_explicit_capability_and_tokenizer_confirmation(mock_upstream):
    requests = mock_upstream(completion("原文。"))
    for options in ({}, {"allow_static_bias": True}, {"tokenizer_id": "test-tokenizer"}):
        with pytest.raises(ValueError, match="静态偏置"):
            providers.rewrite("原文。", model="test", token_bias=bias_table(), **options)
    assert requests == []
    result = providers.rewrite(
        "原文。", model="test", api_key="private-key", token_bias=bias_table(),
        allow_static_bias=True, tokenizer_id="test-tokenizer",
    )
    assert json.loads(requests[0].content)["logit_bias"] == {"12": -0.5}
    assert result["steering_mode"] == "static-token-bias"


def test_positive_static_bias_needs_additional_explicit_opt_in(mock_upstream):
    requests = mock_upstream(completion("原文。"))
    kwargs = {
        "model": "test", "api_key": "private-key", "token_bias": bias_table(0.5),
        "allow_static_bias": True, "tokenizer_id": "test-tokenizer",
    }
    with pytest.raises(ValueError, match="正向"):
        providers.rewrite("原文。", **kwargs)
    assert requests == []
    providers.rewrite("原文。", allow_positive=True, **kwargs)
    assert json.loads(requests[0].content)["logit_bias"] == {"12": 0.5}
