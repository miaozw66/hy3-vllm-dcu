#!/usr/bin/env python3
"""Serve a local web UI at / and proxy /v1/* to vLLM.

Usage:
    HY3_WEB_UI_HTML=/path/to/web_ui.html python3 web_server.py [port]

Environment:
    HY3_WEB_UI_HTML: HTML page to serve. Defaults to web_ui.html beside this file.
    VLLM_BASE_URL: vLLM API base URL. Defaults to http://127.0.0.1:8000.
    HY3_WEB_HOST: bind address. Defaults to 127.0.0.1.
"""
import http.server
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

VLLM = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
DEFAULT_HTML_FILE = Path(__file__).with_name("web_ui.html")
HTML_FILE = Path(os.environ.get("HY3_WEB_UI_HTML", str(DEFAULT_HTML_FILE)))
HOST = os.environ.get("HY3_WEB_HOST", "127.0.0.1")


class ProxyHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/web_ui.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            with HTML_FILE.open("rb") as f:
                self.wfile.write(f.read())
        elif self.path == "/health":
            self._proxy("GET")
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path.startswith("/v1/"):
            self._proxy("POST")
        else:
            self.send_error(404)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _proxy(self, method):
        url = VLLM + self.path
        body = None
        if method == "POST":
            try:
                length = int(self.headers.get("Content-Length", 0))
            except (TypeError, ValueError):
                self.send_error(400, "Invalid Content-Length")
                return
            if length < 0:
                self.send_error(400, "Invalid Content-Length")
                return
            body = self.rfile.read(length) if length > 0 else None

        req = urllib.request.Request(url, data=body, method=method)
        req.add_header("Content-Type", self.headers.get("Content-Type", "application/json"))

        try:
            with urllib.request.urlopen(req, timeout=3600) as resp:
                self.send_response(resp.status)
                self.send_header("Access-Control-Allow-Origin", "*")
                content_type = resp.headers.get("Content-Type", "")
                if content_type.lower().startswith("text/event-stream"):
                    self.send_header("Content-Type", content_type)
                    self.send_header("Cache-Control", "no-cache, no-transform")
                    self.send_header("X-Accel-Buffering", "no")
                    self.end_headers()
                    while True:
                        line = resp.readline()
                        if not line:
                            break
                        self.wfile.write(line)
                        self.wfile.flush()
                else:
                    content = resp.read()
                    self.send_header("Content-Type", content_type or "application/json")
                    self.send_header("Content-Length", str(len(content)))
                    self.end_headers()
                    self.wfile.write(content)
        except urllib.error.HTTPError as e:
            content = e.read()
            content_type = e.headers.get("Content-Type") if e.headers else None
            self.send_response(e.code)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Type", content_type or "application/json")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            content = f"Proxy error: {e}".encode()
            self.send_response(502)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

    def log_message(self, format, *args):
        pass  # suppress logs


if __name__ == "__main__":
    if not HTML_FILE.is_file():
        raise SystemExit(
            f"Web UI file not found: {HTML_FILE}. Set HY3_WEB_UI_HTML to an HTML file."
        )

    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    server = http.server.ThreadingHTTPServer((HOST, port), ProxyHandler)
    print(f"HY3 Web UI ready: http://{HOST}:{port}")
    server.serve_forever()
