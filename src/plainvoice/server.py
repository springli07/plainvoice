"""A local workbench with same-origin JSON endpoints and no remote assets."""

from __future__ import annotations

import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from typing import Any
from urllib.parse import urlsplit

from plainvoice import __version__
from plainvoice.analysis import analyze
from plainvoice.providers import (
    BackendUnavailable,
    RewriteError,
    backend_config,
    rewrite,
    validate_text,
)

MAX_BODY_BYTES = 256 * 1024
_STATIC = {"/": ("index.html", "text/html; charset=utf-8"),
           "/static/app.css": ("app.css", "text/css; charset=utf-8"),
           "/static/app.js": ("app.js", "text/javascript; charset=utf-8")}


class WorkbenchServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 16


class Handler(BaseHTTPRequestHandler):
    server_version = "PlainVoice"
    sys_version = ""

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(70)

    def log_message(self, fmt: str, *args: Any) -> None:
        # No bodies, query strings, input text, credentials, or upstream exceptions in logs.
        pass

    def _respond(self, status: int, data: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; "
            "img-src 'self' data:; font-src 'self'; base-uri 'none'; frame-ancestors 'none'; "
            "form-action 'self'",
        )
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        self.wfile.write(data)

    def _json(self, status: int, value: Any) -> None:
        body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self._respond(status, body, "application/json; charset=utf-8")

    def _trusted_host(self) -> bool:
        """Limit local-server DNS rebinding and accidental cross-origin use."""
        host = self.headers.get("Host", "")
        try:
            parsed = urlsplit("http://" + host)
            address, port = self.server.server_address[:2]
            accepted = {"localhost", "127.0.0.1", "::1", address}
            return (
                parsed.hostname in accepted
                and parsed.port == port
                and not parsed.username
                and not parsed.password
                and parsed.path == ""
            )
        except ValueError:
            return False

    def do_GET(self) -> None:
        if not self._trusted_host():
            self._json(403, {"error": "请求 Host 不匹配。"})
            return
        if self.path == "/health":
            self._json(200, {"status": "ok", "version": __version__})
        elif self.path == "/api/config":
            self._json(200, backend_config())
        elif self.path in _STATIC:
            name, content_type = _STATIC[self.path]
            content = files("plainvoice").joinpath("static", name).read_bytes()
            self._respond(200, content, content_type)
        else:
            self._json(404, {"error": "路径不存在。"})

    def do_POST(self) -> None:
        if not self._trusted_host():
            self._json(403, {"error": "请求 Host 不匹配。"})
            return
        origin = self.headers.get("Origin")
        if origin is not None and origin != "http://" + self.headers.get("Host", ""):
            self._json(403, {"error": "仅允许同源请求。"})
            return
        if self.headers.get("Transfer-Encoding"):
            self._json(400, {"error": "不支持分块请求体。"})
            return
        if self.headers.get_content_type() != "application/json":
            self._json(415, {"error": "请使用 application/json。"})
            return
        if self.path not in {"/api/analyze", "/api/rewrite"}:
            self._json(404, {"error": "路径不存在。"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > MAX_BODY_BYTES:
                self._json(413, {"error": "请求体为空或超过 256 KiB。"})
                return
            raw = self.rfile.read(length)
            data = json.loads(raw)
            if not isinstance(data, dict) or set(data) != {"text"}:
                raise ValueError("请求必须只包含 text 字段。")
            text = validate_text(data["text"])
            if self.path == "/api/analyze":
                self._json(200, analyze(text))
            else:
                config = backend_config()
                if not config["configured"]:
                    raise BackendUnavailable(config["reason"])
                self._json(200, rewrite(text, backend=config["backend"]))
        except (ValueError, UnicodeError) as exc:
            self._json(400, {"error": str(exc) if not isinstance(exc, json.JSONDecodeError)
                              else "请求不是有效的 JSON。"})
        except BackendUnavailable as exc:
            self._json(503, {"error": str(exc)})
        except RewriteError as exc:
            self._json(502, {"error": str(exc)})
        except (TimeoutError, ConnectionError):
            self.close_connection = True
        except Exception:
            logging.getLogger(__name__).error("Workbench request failed (details suppressed)")
            self._json(500, {"error": "服务内部错误，请检查本机配置。"})


def create_server(host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    if host not in {"127.0.0.1", "localhost"}:
        raise ValueError("工作台仅支持本机回环监听，请使用 127.0.0.1 或 localhost。")
    return WorkbenchServer((host, port), Handler)
