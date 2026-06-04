from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


CACHE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CachedNeighbor:
    memory_id: str
    boost: float
    hop: int


@dataclass(frozen=True)
class CachedFrame:
    memory_id: str
    neighbors: tuple[CachedNeighbor, ...]
    version: int
    built_at: float


class FrameCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(self.path))
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self._ensure_schema()

    def close(self) -> None:
        self.connection.close()

    def _ensure_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS frames (
                memory_id TEXT PRIMARY KEY,
                version INTEGER NOT NULL,
                built_at REAL NOT NULL,
                payload TEXT NOT NULL
            );
            """
        )
        self.connection.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
            ("schema_version", str(CACHE_SCHEMA_VERSION)),
        )
        self.connection.commit()

    def put(self, memory_id: str, neighbors: Iterable[CachedNeighbor], version: int = CACHE_SCHEMA_VERSION) -> None:
        ordered = sorted(neighbors, key=lambda item: (-item.boost, item.hop, item.memory_id))
        payload = json.dumps(
            [{"id": item.memory_id, "boost": round(float(item.boost), 8), "hop": int(item.hop)} for item in ordered],
            separators=(",", ":"),
            sort_keys=True,
        )
        self.connection.execute(
            "INSERT OR REPLACE INTO frames(memory_id, version, built_at, payload) VALUES (?, ?, ?, ?)",
            (memory_id, version, time.time(), payload),
        )

    def put_many(self, rows: Iterable[tuple[str, Iterable[CachedNeighbor]]], version: int = CACHE_SCHEMA_VERSION) -> None:
        for memory_id, neighbors in rows:
            self.put(memory_id, neighbors, version)
        self.connection.commit()

    def get(self, memory_id: str) -> CachedFrame | None:
        row = self.connection.execute(
            "SELECT version, built_at, payload FROM frames WHERE memory_id = ?",
            (memory_id,),
        ).fetchone()
        if row is None:
            return None
        version, built_at, payload = row
        raw_neighbors = json.loads(payload)
        neighbors = tuple(
            CachedNeighbor(str(item["id"]), float(item["boost"]), int(item["hop"]))
            for item in raw_neighbors
        )
        return CachedFrame(memory_id=memory_id, neighbors=neighbors, version=int(version), built_at=float(built_at))

    def stats(self) -> dict[str, Any]:
        frame_count = int(self.connection.execute("SELECT COUNT(*) FROM frames").fetchone()[0])
        size_bytes = 0
        for path in [self.path, self.path.with_name(f"{self.path.name}-wal"), self.path.with_name(f"{self.path.name}-shm")]:
            if path.exists():
                size_bytes += path.stat().st_size
        return {"frame_count": frame_count, "size_bytes": size_bytes, "path": str(self.path)}
