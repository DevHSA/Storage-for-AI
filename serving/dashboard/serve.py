#!/usr/bin/env python3
"""Zero-dependency web server for the LLMServingSim live dashboard.

Serves ``index.html`` at ``/`` and the live-metrics snapshot at
``/api/metrics`` (re-read fresh on every request). Also serves the drag-and-drop
cluster-config builder at ``/config`` and accepts ``POST /api/config`` to write a
generated config straight into ``configs/cluster/`` (filename sanitised with
``os.path.basename`` so it cannot escape that directory). Uses ONLY the Python
standard library, so it runs on the host with no pip installs and no Docker
changes.

Usage (from the repo root, on the host)::

    python3 serving/dashboard/serve.py                 # port 8000
    python3 serving/dashboard/serve.py --port 9000 --file outputs/dashboard/live.json

Then open http://localhost:8000 (dashboard) or http://localhost:8000/config
(config builder), and run a sim with ``--dashboard``.
"""

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))


def _make_handler(index_path, metrics_path, metrics_path2=None):
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
            elif path in ("/config", "/config.html", "/config_builder.html"):
                try:
                    with open(os.path.join(_HERE, "config_builder.html"), "rb") as f:
                        self._send(200, f.read(), "text/html; charset=utf-8")
                except Exception as e:
                    self._send(500, str(e).encode(), "text/plain")
            elif path in ("/api/metrics", "/api/metrics2"):
                mp = metrics_path2 if path == "/api/metrics2" else metrics_path
                if not mp:
                    self._send(200, b'{"status":"waiting"}', "application/json")
                    return
                try:
                    with open(mp, "rb") as f:
                        self._send(200, f.read(), "application/json")
                except FileNotFoundError:
                    self._send(200, b'{"status":"waiting"}', "application/json")
                except Exception as e:
                    self._send(200, json.dumps({"status": "error", "error": str(e)}).encode(),
                               "application/json")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            path = self.path.split("?", 1)[0]
            if path != "/api/config":
                self._send(404, b"not found", "text/plain")
                return
            try:
                n = int(self.headers.get("Content-Length") or 0)
                obj = json.loads(self.rfile.read(n))
                # sanitise filename: basename only, .json suffix, into configs/cluster/
                name = os.path.basename(str(obj.pop("_filename", "cluster.json"))) or "cluster.json"
                if not name.endswith(".json"):
                    name += ".json"
                dest_dir = os.path.join(_REPO, "configs", "cluster")
                os.makedirs(dest_dir, exist_ok=True)
                dest = os.path.join(dest_dir, name)
                with open(dest, "w", encoding="utf-8") as f:
                    json.dump(obj, f, indent=2)
                rel = os.path.join("configs", "cluster", name)
                self._send(200, json.dumps({"ok": True, "path": rel}).encode(), "application/json")
            except Exception as e:
                self._send(400, json.dumps({"ok": False, "error": str(e)}).encode(), "application/json")

        def log_message(self, *args):
            pass  # keep the console quiet

    return Handler


def main():
    ap = argparse.ArgumentParser(description="LLMServingSim live dashboard (stdlib server)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--file", default=os.path.join(_REPO, "outputs", "dashboard", "live.json"),
                    help="path to the run-1 live-metrics JSON written by `--dashboard`")
    ap.add_argument("--file2", default=os.path.join(_REPO, "outputs", "dashboard", "live2.json"),
                    help="path to the run-2 live-metrics JSON (V3 comparison; served at /api/metrics2)")
    ap.add_argument("--dashboardv2", action="store_true",
                    help="serve the single-run V2 dashboard (index.html). Default: V3 run-comparison "
                    "dashboard (index_v3.html), which compares --file (run 1) vs --file2 (run 2).")
    a = ap.parse_args()
    index_file = "index.html" if a.dashboardv2 else "index_v3.html"
    handler = _make_handler(os.path.join(_HERE, index_file), os.path.abspath(a.file),
                            os.path.abspath(a.file2))
    srv = ThreadingHTTPServer((a.host, a.port), handler)
    print(f"LLMServingSim dashboard ({'V2 single-run' if a.dashboardv2 else 'V3 comparison'})  ->  http://localhost:{a.port}")
    print(f"  run 1 metrics: {os.path.abspath(a.file)}")
    if not a.dashboardv2:
        print(f"  run 2 metrics: {os.path.abspath(a.file2)}")
    print("  run a sim with --dashboard to populate these. Ctrl-C to stop.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
