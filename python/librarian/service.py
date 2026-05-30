from __future__ import annotations

import argparse
import json
import math
import re
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> set[str]:
    return set(TOKEN_RE.findall(text.lower()))


def jaccard(left: str, right: str) -> float:
    a = tokens(left)
    b = tokens(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def cosine(a: list[float], b: list[float]) -> float:
    if not a or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def score_neighborhood(payload: dict[str, Any]) -> dict[str, Any]:
    anchor = payload["anchor"]
    candidates = payload.get("candidates", [])
    actions = []
    for candidate in candidates:
        semantic = cosine(anchor.get("embedding", []), candidate.get("embedding", []))
        lexical = jaccard(anchor.get("text", ""), candidate.get("text", ""))
        score = 0.65 * semantic + 0.35 * lexical
        if score < 0.25:
            continue
        edge_type = "same_topic" if lexical > 0.15 else "used_with"
        actions.append(
            {
                "candidate_id": candidate["id"],
                "connect_score": score,
                "edge_type": edge_type,
                "weight": min(1.2, 0.2 + score),
                "decay_rate": 0.02,
                "importance_delta": max(-0.04, min(0.08, (score - 0.5) * 0.08)),
            }
        )
    actions.sort(key=lambda item: (-item["connect_score"], item["candidate_id"]))
    actions = actions[:8]
    return {
        "actions": actions,
        "create_new_cluster": len(actions) == 0,
        "new_cluster_score": 1.0 - (actions[0]["connect_score"] if actions else 0.0),
    }


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--addr", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()
    server = HTTPServer((args.addr, args.port), Handler)
    print(f"librarian placeholder listening on http://{args.addr}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()

