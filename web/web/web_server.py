#!/usr/bin/env python3
"""Simple HTTP server: serves web_ui.html at / and proxies /v1/* to vLLM.
Usage: python3 web_server.py [port]
Default port: 8080
"""
import sys
import http.server
import urllib.request
import urllib.error

VLLM = "http://localhost:8000"
HTML_FILE = "/data/mzw/vllm-hy3/web_ui.html"


class ProxyHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/web_ui.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            with open(HTML_FILE, "rb") as f:
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
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    server = http.server.ThreadingHTTPServer(("0.0.0.0", port), ProxyHandler)
    print(f"HY3 Web UI ready: http://<ip>:{port}")
    server.serve_forever()
