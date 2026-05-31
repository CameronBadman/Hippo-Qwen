from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from python.librarian.inference import score_with_heuristic

MODEL = None
THRESHOLD = 0.5


def score_neighborhood(payload: dict[str, Any]) -> dict[str, Any]:
    if MODEL is not None:
        from python.librarian.inference import score_with_model

        return score_with_model(MODEL, payload, threshold=THRESHOLD)
    return score_with_heuristic(payload)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/healthz":
            self.write_json({"status": "ok"})
            return
        self.write_json({"error": "not found"}, status=404)

    def do_POST(self) -> None:
        if self.path != "/score-neighborhood":
            self.write_json({"error": "not found"}, status=404)
            return
        length = int(self.headers.get("content-length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        self.write_json(score_neighborhood(payload))

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def write_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    global MODEL, THRESHOLD
    parser = argparse.ArgumentParser()
    parser.add_argument("--addr", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    THRESHOLD = args.threshold
    if args.checkpoint:
        import torch

        from python.librarian.model import load_checkpoint

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        MODEL = load_checkpoint(args.checkpoint, device=device)
        print(f"loaded librarian checkpoint {args.checkpoint} on {device}", flush=True)
    server = HTTPServer((args.addr, args.port), Handler)
    mode = "model" if MODEL is not None else "heuristic"
    print(f"librarian {mode} service listening on http://{args.addr}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
