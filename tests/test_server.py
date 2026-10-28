import http.client
import json
import threading

import pytest

from plainvoice import server
from plainvoice.providers import BackendUnavailable, RewriteError


@pytest.fixture
def workbench(monkeypatch):
    for name in ("PLAINVOICE_MODEL", "PLAINVOICE_API_KEY", "PLAINVOICE_BASE_URL", "PLAINVOICE_BACKEND"):
        monkeypatch.delenv(name, raising=False)
    instance = server.create_server(port=0)
    thread = threading.Thread(target=instance.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    yield instance
    instance.shutdown()
    instance.server_close()
    thread.join(timeout=3)
    assert not thread.is_alive()


def request(instance, method, path, value=None, *, raw=None, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", instance.server_port, timeout=3)
    headers = dict(headers or {})
    body = raw
    if value is not None:
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        headers.setdefault("Content-Type", "application/json")
    try:
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


def test_workbench_serves_packaged_assets_and_offline_analysis(workbench):
    for path, content_type in (("/", "text/html"), ("/static/app.css", "text/css"),
                               ("/static/app.js", "text/javascript")):
        status, headers, body = request(workbench, "GET", path)
        assert status == 200
        assert headers["Content-Type"].startswith(content_type)
        assert body
        assert "Access-Control-Allow-Origin" not in headers
        assert headers["X-Content-Type-Options"] == "nosniff"
    status, _, body = request(workbench, "GET", "/health")
    assert status == 200 and json.loads(body)["status"] == "ok"
    text = "不难看出，我们明天发布。"
    status, _, body = request(workbench, "POST", "/api/analyze", {"text": text})
    analysis = json.loads(body)
    assert status == 200
    assert analysis["character_count"] == len(text)
    assert any(hit["text"] == "不难看出" for hit in analysis["findings"])


@pytest.mark.parametrize("method,path,headers", [
    ("GET", "/api/config", {"Host": "attacker.example:8765"}),
    ("POST", "/api/analyze", {"Origin": "https://attacker.example"}),
    ("POST", "/api/rewrite", {"Origin": "null"}),
])
def test_host_and_cross_origin_requests_are_rejected(workbench, method, path, headers):
    status, _, body = request(workbench, method, path, {"text": "原文。"}, headers=headers)
    assert status == 403
    assert "error" in json.loads(body)


def test_same_origin_post_is_accepted(workbench):
    headers = {"Origin": f"http://127.0.0.1:{workbench.server_port}"}
    assert request(workbench, "POST", "/api/analyze", {"text": "原文。"}, headers=headers)[0] == 200


@pytest.mark.parametrize("value", [None, [], {"text": ""}, {"text": 42},
                                   {"text": "原文。", "base_url": "http://attacker.example"}])
def test_invalid_json_shapes_and_request_options_are_rejected(workbench, value):
    status, _, body = request(
        workbench, "POST", "/api/analyze", raw=json.dumps(value).encode(),
        headers={"Content-Type": "application/json"},
    )
    assert status == 400
    assert "error" in json.loads(body)


@pytest.mark.parametrize("raw,headers,status", [
    (b"{broken", {"Content-Type": "application/json"}, 400),
    (b"\xff", {"Content-Type": "application/json"}, 400),
    (b"{}", {"Content-Type": "text/plain"}, 415),
    (b"", {"Content-Type": "application/json", "Content-Length": "262145"}, 413),
    (b"", {"Content-Type": "application/json", "Content-Length": "invalid"}, 400),
    (b"", {"Content-Type": "application/json", "Transfer-Encoding": "chunked"}, 400),
])
def test_malformed_or_oversized_body_is_a_bounded_failure(workbench, raw, headers, status):
    actual_status, _, body = request(workbench, "POST", "/api/analyze", raw=raw, headers=headers)
    assert actual_status == status
    assert "error" in json.loads(body)


def test_unconfigured_rewrite_is_unavailable_but_analysis_works(workbench, monkeypatch):
    def no_generation(*args, **kwargs):
        pytest.fail("unconfigured server must not call a model")

    monkeypatch.setattr(server, "rewrite", no_generation)
    status, _, body = request(workbench, "POST", "/api/rewrite", {"text": "原文。"})
    assert status == 503
    assert "error" in json.loads(body)
    assert request(workbench, "POST", "/api/analyze", {"text": "原文。"})[0] == 200


def test_configuration_response_never_contains_credentials(workbench, monkeypatch):
    monkeypatch.setenv("PLAINVOICE_MODEL", "test-model")
    monkeypatch.setenv("PLAINVOICE_API_KEY", "secret-config-value")
    status, _, body = request(workbench, "GET", "/api/config")
    assert status == 200
    assert json.loads(body)["configured"] is True
    assert b"secret-config-value" not in body


@pytest.mark.parametrize("exception,status", [(BackendUnavailable("backend unavailable"), 503),
                                             (RewriteError("preservation failed"), 502),
                                             (RuntimeError("secret details"), 500)])
def test_generation_errors_map_to_http_failures_without_internal_details(
    workbench, monkeypatch, exception, status,
):
    monkeypatch.setenv("PLAINVOICE_MODEL", "test-model")
    monkeypatch.setenv("PLAINVOICE_API_KEY", "secret-config-value")

    def fail(*args, **kwargs):
        raise exception

    monkeypatch.setattr(server, "rewrite", fail)
    actual_status, _, body = request(workbench, "POST", "/api/rewrite", {"text": "原文。"})
    assert actual_status == status
    assert "error" in json.loads(body)
    assert b"secret details" not in body and b"secret-config-value" not in body


def test_workbench_refuses_public_listening():
    with pytest.raises(ValueError, match="回环"):
        server.create_server("0.0.0.0", 0)
