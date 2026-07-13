#!/usr/bin/env python3
"""Zero-dependency web server for the LLMServingSim live dashboard.

Serves ``index.html`` at ``/`` and the live-metrics snapshot at
``/api/metrics`` (re-read fresh on every request). Uses ONLY the Python
standard library, so it runs on the host with no pip installs and no Docker
changes.

Usage (from the repo root, on the host)::

    python3 serving/dashboard/serve.py                 # port 8000
    python3 serving/dashboard/serve.py --port 9000 --file outputs/dashboard/live.json

Then open http://localhost:8000 and run a sim with ``--dashboard``.
"""

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))


def _make_handler(index_path, metrics_path):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except BrokenPipeError:
                pass

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                try:
                    with open(index_path, "rb") as f:
                        self._send(200, f.read(), "text/html; charset=utf-8")
                except Exception as e:
                    self._send(500, str(e).encode(), "text/plain")
            elif path == "/api/metrics":
                try:
                    with open(metrics_path, "rb") as f:
                        self._send(200, f.read(), "application/json")
                except FileNotFoundError:
                    self._send(200, b'{"status":"waiting"}', "application/json")
                except Exception as e:
                    self._send(200, json.dumps({"status": "error", "error": str(e)}).encode(),
                               "application/json")
            else:
                self._send(404, b"not found", "text/plain")

        def log_message(self, *args):
            pass  # keep the console quiet

    return Handler


def main():
    ap = argparse.ArgumentParser(description="LLMServingSim live dashboard (stdlib server)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--file", default=os.path.join(_REPO, "outputs", "dashboard", "live.json"),
                    help="path to the live-metrics JSON written by `--dashboard`")
    a = ap.parse_args()
    handler = _make_handler(os.path.join(_HERE, "index.html"), os.path.abspath(a.file))
    srv = ThreadingHTTPServer((a.host, a.port), handler)
    print(f"LLMServingSim dashboard  ->  http://localhost:{a.port}")
    print(f"  metrics file: {os.path.abspath(a.file)}")
    print("  run a sim with --dashboard to populate it. Ctrl-C to stop.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
