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
FRAME_CACHE = None
FRAME_CONFIG = None


def score_neighborhood(payload: dict[str, Any]) -> dict[str, Any]:
    if MODEL is not None:
        from python.librarian.inference import score_with_model

        return score_with_model(MODEL, payload, threshold=THRESHOLD)
    return score_with_heuristic(payload)


def populate_cache(payload: dict[str, Any]) -> dict[str, Any]:
    if FRAME_CACHE is None or FRAME_CONFIG is None:
        return {"error": "frame cache is not configured"}
    from python.librarian.frame_builder import populate_frame_cache

    stats = populate_frame_cache(payload, FRAME_CACHE, FRAME_CONFIG)
    return {"status": "ok", "stats": stats, "cache": FRAME_CACHE.stats()}


def build_cached_frame(payload: dict[str, Any]) -> dict[str, Any]:
    if FRAME_CACHE is None or FRAME_CONFIG is None:
        return {"error": "frame cache is not configured"}
    from python.librarian.frame_builder import build_cached_graph_frame, frame_recall, select_seed_ids

    seed_ids = payload.get("seed_ids")
    base_scores = None
    seed_selection_ms = 0.0
    row = payload
    if "row" in payload:
        row = payload["row"]
    if not seed_ids:
        seed_ids, base_scores, seed_selection_ms = select_seed_ids(row, FRAME_CONFIG)
    frame, stats = build_cached_graph_frame(row, FRAME_CACHE, FRAME_CONFIG, seed_ids=list(seed_ids), base_scores=base_scores)
    stats["seed_selection_ms"] = max(float(stats.get("seed_selection_ms") or 0.0), seed_selection_ms)
    return {
        "status": "ok",
        "ids": [memory_id for memory_id in frame.get("ids", []) if memory_id],
        "frame_recall": frame_recall(frame),
        "stats": stats,
    }


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/healthz":
            self.write_json({"status": "ok"})
            return
        self.write_json({"error": "not found"}, status=404)

    def do_POST(self) -> None:
        if self.path not in {"/score-neighborhood", "/populate-frame-cache", "/build-cached-frame"}:
            self.write_json({"error": "not found"}, status=404)
            return
        length = int(self.headers.get("content-length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        if self.path == "/score-neighborhood":
            self.write_json(score_neighborhood(payload))
        elif self.path == "/populate-frame-cache":
            result = populate_cache(payload)
            self.write_json(result, status=400 if "error" in result else 200)
        else:
            result = build_cached_frame(payload)
            self.write_json(result, status=400 if "error" in result else 200)

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
    global MODEL, THRESHOLD, FRAME_CACHE, FRAME_CONFIG
    parser = argparse.ArgumentParser()
    parser.add_argument("--addr", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--frame-cache", default="")
    parser.add_argument("--frame-size", type=int, default=64)
    parser.add_argument("--graph-seed-count", type=int, default=16)
    parser.add_argument("--graph-depth", type=int, default=3)
    parser.add_argument("--graph-boost", type=float, default=0.85)
    parser.add_argument("--hide-role-features", action="store_true")
    args = parser.parse_args()
    THRESHOLD = args.threshold
    if args.checkpoint:
        import torch

        from python.librarian.model import load_checkpoint

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        MODEL = load_checkpoint(args.checkpoint, device=device)
        print(f"loaded librarian checkpoint {args.checkpoint} on {device}", flush=True)
    if args.frame_cache:
        from python.librarian.frame_builder import FrameBuilderConfig
        from python.librarian.frame_cache import FrameCache

        FRAME_CACHE = FrameCache(args.frame_cache)
        FRAME_CONFIG = FrameBuilderConfig(
            frame_size=args.frame_size,
            graph_seed_count=args.graph_seed_count,
            graph_depth=args.graph_depth,
            graph_boost=args.graph_boost,
            use_role_features=not args.hide_role_features,
        )
        print(f"frame cache enabled at {args.frame_cache}", flush=True)
    server = HTTPServer((args.addr, args.port), Handler)
    mode = "model" if MODEL is not None else "heuristic"
    print(f"librarian {mode} service listening on http://{args.addr}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
