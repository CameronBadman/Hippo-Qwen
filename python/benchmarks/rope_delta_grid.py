from __future__ import annotations

import argparse
import json
import math
import mmap
import struct
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from python.benchmarks.benchmark_librarian import evaluate_ranked
from python.benchmarks.hard_memory_regression import (
    GROWTH_SCENARIOS,
    aggregate,
    average,
    changed_existing_positions,
    grow_row,
    quantile,
    relevant_retention,
    top_retention,
)
from python.benchmarks.hierarchical_file_ann import Ranked, project_key, ranked_signature
from python.benchmarks.hippocampus_retrieval import (
    activation_overlap,
    build_embedding_backend,
    edge_activation,
    edge_type_boost,
    ensure_backend_embeddings,
    multihop_metrics,
    state_score,
)
from python.benchmarks.large_pool_retrieval import build_large_pool_case, context_decoy_metrics
from python.benchmarks.skeleton_memory_index import compact_output, mask_overlap, parse_scenarios, skeleton_flags
from python.librarian.features import activation_mask_for_text, fnv1a64


GRID_MAGIC = b"HRDG1\0\0\0"
GRID_VERSION = 1
HEADER = struct.Struct("<8sIIIIIf")
CELL = struct.Struct("<QHhhhIiiii")
NODE = struct.Struct("<IQQQHhhhIiiHff")
PAYLOAD_RECORD = struct.Struct("<QQIQHII")
EDGE = struct.Struct("<IQfff")
NO_PTR = -1


def dim_triples(dim_count: int, layer_count: int, schedule: str = "auto") -> list[tuple[int, int, int]]:
    if dim_count <= 0:
        return [(0, 0, 0)] * layer_count
    mode = str(schedule or "auto").lower()
    if mode == "auto":
        mode = "consecutive" if layer_count * 3 >= dim_count else "spread"
    triples = []
    if mode == "consecutive":
        for layer in range(layer_count):
            base = (layer * 3) % dim_count
            triples.append((base, (base + 1) % dim_count, (base + 2) % dim_count))
        return triples
    if mode != "spread":
        raise ValueError(f"unknown layer schedule: {schedule}")
    # Deterministic low-overlap triples spread over the whole embedding space.
    # The odd strides keep dimensions mixed without data-dependent rebalancing.
    stride_a = max(1, (dim_count // max(1, layer_count)) | 1)
    stride_b = max(1, ((dim_count // 3) + 1) | 1)
    stride_c = max(1, ((dim_count // 2) + 1) | 1)
    for layer in range(layer_count):
        base = (layer * stride_a) % dim_count
        triples.append((base, (base + stride_b) % dim_count, (base + stride_c) % dim_count))
    return triples


def clamp_i16(value: int) -> int:
    return max(-32768, min(32767, value))


def quantized_delta(value: float, origin: float, cell_width: float) -> int:
    if cell_width <= 0.0:
        raise ValueError("cell_width must be positive")
    return clamp_i16(int(math.floor((float(value) - float(origin)) / cell_width)))


def quantized_point(embedding: list[float], origin: list[float], triple: tuple[int, int, int], cell_width: float) -> tuple[int, int, int]:
    out = []
    for dim in triple:
        value = embedding[dim] if dim < len(embedding) else 0.0
        base = origin[dim] if dim < len(origin) else 0.0
        out.append(quantized_delta(value, base, cell_width))
    return (out[0], out[1], out[2])


def delta_energy(embedding: list[float], origin: list[float], triple: tuple[int, int, int]) -> float:
    total = 0.0
    for dim in triple:
        value = embedding[dim] if dim < len(embedding) else 0.0
        base = origin[dim] if dim < len(origin) else 0.0
        total += abs(float(value) - float(base))
    return total


def selected_layers(
    embedding: list[float],
    origin: list[float],
    triples: list[tuple[int, int, int]],
    min_delta: float,
    min_layers: int,
    max_layers: int,
) -> tuple[list[int], list[float]]:
    energies = [delta_energy(embedding, origin, triple) for triple in triples]
    selected = [layer for layer, energy in enumerate(energies) if energy >= min_delta]
    floor = max(0, min(int(min_layers), len(triples)))
    if len(selected) < floor:
        ranked = sorted(range(len(triples)), key=lambda layer: (-energies[layer], layer))
        selected = sorted(set(selected).union(ranked[:floor]))
    if max_layers > 0 and len(selected) > max_layers:
        selected = sorted(selected, key=lambda layer: (-energies[layer], layer))[:max_layers]
        selected.sort()
    return selected, energies


def static_node_priority(node: dict[str, Any]) -> tuple[float, int, int]:
    flags = int(node["flags"])
    conflict_penalty = 0.08 if flags & 2 else 0.0
    ignored_penalty = 0.05 if flags & 4 else 0.0
    stale_penalty = 0.05 if flags & 8 else 0.0
    score = float(node["importance"]) + float(node["state"]) - conflict_penalty - ignored_penalty - stale_penalty
    return (-score, int(node["node_hash"]), int(node["memory_index"]))


def split_by_3(value: int) -> int:
    value &= 0x1FFFFF
    value = (value | (value << 32)) & 0x1F00000000FFFF
    value = (value | (value << 16)) & 0x1F0000FF0000FF
    value = (value | (value << 8)) & 0x100F00F00F00F00F
    value = (value | (value << 4)) & 0x10C30C30C30C30C3
    value = (value | (value << 2)) & 0x1249249249249249
    return value


def morton3(x: int, y: int, z: int) -> int:
    bias = 1 << 20
    return split_by_3(x + bias) | (split_by_3(y + bias) << 1) | (split_by_3(z + bias) << 2)


def read_vector(card: dict[str, Any], dim_count: int) -> list[float]:
    raw = [float(value) for value in card.get("embedding") or []]
    if len(raw) >= dim_count:
        return raw[:dim_count]
    return raw + [0.0] * (dim_count - len(raw))


def payload_offsets(cards: list[dict[str, Any]], payload_path: Path) -> list[tuple[int, int, int, int]]:
    offsets = []
    with payload_path.open("wb") as handle:
        for card in cards:
            node_id = str(card.get("id") or "")
            text = str(card.get("text") or "")
            id_frame = node_id.encode("utf-8")
            text_frame = text.encode("utf-8")
            id_offset = handle.tell()
            handle.write(id_frame)
            text_offset = handle.tell()
            handle.write(text_frame)
            offsets.append((id_offset, len(id_frame), text_offset, len(text_frame)))
    return offsets


def build_rope_delta_grid(row: dict[str, Any], backend: Any, output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    row = ensure_backend_embeddings(row, backend)
    output_dir.mkdir(parents=True, exist_ok=True)
    grid_path = output_dir / "rope_grid.hrg"
    payload_path = output_dir / "payload.bin"
    records_path = output_dir / "payload.idx"
    edges_path = output_dir / "edges.bin"
    meta_path = output_dir / "rope_grid.json"
    layer_count = max(1, int(args.layers))
    dim_count = max(3, int(args.dim_count))
    cell_width = float(args.cell_width)
    min_node_layer_delta = float(getattr(args, "min_node_layer_delta", 0.0) or 0.0)
    min_node_layers = int(getattr(args, "min_node_layers", 1) or 0)
    max_node_layers = int(getattr(args, "max_node_layers", 0) or 0)

    cards = [dict(card) for card in row.get("candidates", [])]
    cards.sort(key=lambda card: str(card.get("id") or ""))
    if not cards:
        raise ValueError("cannot build an empty rope delta grid")
    id_to_index = {str(card.get("id") or ""): index for index, card in enumerate(cards)}
    origin = read_vector(cards[0], dim_count)
    triples = dim_triples(dim_count, layer_count, str(getattr(args, "layer_schedule", "auto")))
    offsets = payload_offsets(cards, payload_path)

    outgoing: dict[int, list[dict[str, Any]]] = {}
    for edge in (row.get("memory_graph") or {}).get("edges") or []:
        source = str(edge.get("source") or "")
        target = str(edge.get("target") or "")
        if source in id_to_index and target in id_to_index:
            outgoing.setdefault(id_to_index[source], []).append(edge)

    edge_frames: list[bytes] = []
    edge_offsets: list[int] = []
    edge_counts: list[int] = []
    for memory_index, card in enumerate(cards):
        edge_offsets.append(len(edge_frames))
        for edge in sorted(outgoing.get(memory_index, []), key=lambda item: (str(item.get("target") or ""), str(item.get("type") or ""))):
            edge_frames.append(
                EDGE.pack(
                    id_to_index[str(edge.get("target") or "")],
                    int(edge_activation(edge)),
                    float(edge.get("weight") or 0.0),
                    float(edge.get("confidence") or 0.5),
                    edge_type_boost(str(edge.get("type") or "used_with")),
                )
            )
        edge_counts.append(len(edge_frames) - edge_offsets[-1])

    cell_members: dict[tuple[int, int, int, int], list[int]] = {}
    node_frames: list[dict[str, Any]] = []
    for memory_index, card in enumerate(cards):
        embedding = read_vector(card, dim_count)
        active_layers, _ = selected_layers(
            embedding,
            origin,
            triples,
            min_node_layer_delta,
            min_node_layers,
            max_node_layers,
        )
        project_hash = fnv1a64(project_key(card))
        mask = activation_mask_for_text(f"{card.get('text', '')} {card.get('summary', '')}")
        flags = skeleton_flags(card)
        for layer in active_layers:
            triple = triples[layer]
            x, y, z = quantized_point(embedding, origin, triple, cell_width)
            cell_key = (layer, x, y, z)
            node_index = len(node_frames)
            cell_members.setdefault(cell_key, []).append(node_index)
            node_frames.append(
                {
                    "memory_index": memory_index,
                    "node_hash": fnv1a64(str(card.get("id") or "")),
                    "project_hash": project_hash,
                    "mask": int(mask),
                    "layer": layer,
                    "x": x,
                    "y": y,
                    "z": z,
                    "cell_key": cell_key,
                    "prev": NO_PTR,
                    "next": NO_PTR,
                    "flags": flags,
                    "importance": float(card.get("importance") or 0.5),
                    "state": float(state_score(card)),
                }
            )

    cell_keys = sorted(cell_members, key=lambda key: (key[0], morton3(key[1], key[2], key[3]), key[1], key[2], key[3]))
    cell_index_by_key = {key: index for index, key in enumerate(cell_keys)}
    cells = []
    last_by_layer: dict[int, int] = {}
    for cell_index, key in enumerate(cell_keys):
        layer, x, y, z = key
        members = sorted(cell_members[key], key=lambda index: static_node_priority(node_frames[index]))
        for member_position, node_index in enumerate(members):
            node_frames[node_index]["cell_index"] = cell_index
            node_frames[node_index]["prev"] = members[member_position - 1] if member_position > 0 else NO_PTR
            node_frames[node_index]["next"] = members[member_position + 1] if member_position + 1 < len(members) else NO_PTR
        prev_cell = last_by_layer.get(layer, NO_PTR)
        cells.append(
            {
                "key_hash": fnv1a64(f"{layer}:{x}:{y}:{z}"),
                "layer": layer,
                "x": x,
                "y": y,
                "z": z,
                "count": len(members),
                "head": members[0],
                "tail": members[-1],
                "prev": prev_cell,
                "next": NO_PTR,
            }
        )
        if prev_cell != NO_PTR:
            cells[prev_cell]["next"] = cell_index
        last_by_layer[layer] = cell_index

    with grid_path.open("wb") as handle:
        handle.write(HEADER.pack(GRID_MAGIC, GRID_VERSION, layer_count, dim_count, len(cards), len(cells), cell_width))
        for cell in cells:
            handle.write(
                CELL.pack(
                    int(cell["key_hash"]),
                    int(cell["layer"]),
                    int(cell["x"]),
                    int(cell["y"]),
                    int(cell["z"]),
                    int(cell["count"]),
                    int(cell["head"]),
                    int(cell["tail"]),
                    int(cell["prev"]),
                    int(cell["next"]),
                )
            )
        for node in node_frames:
            handle.write(
                NODE.pack(
                    int(node["memory_index"]),
                    int(node["node_hash"]),
                    int(node["project_hash"]),
                    int(node["mask"]),
                    int(node["layer"]),
                    int(node["x"]),
                    int(node["y"]),
                    int(node["z"]),
                    int(node["cell_index"]),
                    int(node["prev"]),
                    int(node["next"]),
                    int(node["flags"]),
                    float(node["importance"]),
                    float(node["state"]),
                )
            )

    with records_path.open("wb") as handle:
        for memory_index, card in enumerate(cards):
            id_offset, id_size, text_offset, text_size = offsets[memory_index]
            handle.write(
                PAYLOAD_RECORD.pack(
                    fnv1a64(str(card.get("id") or "")),
                    text_offset,
                    text_size,
                    id_offset,
                    id_size,
                    edge_offsets[memory_index],
                    edge_counts[memory_index],
                )
            )
    with edges_path.open("wb") as handle:
        for frame in edge_frames:
            handle.write(frame)

    meta = {
        "version": GRID_VERSION,
        "grid_path": str(grid_path),
        "payload_path": str(payload_path),
        "records_path": str(records_path),
        "edges_path": str(edges_path),
        "layers": layer_count,
        "layer_schedule": str(getattr(args, "layer_schedule", "auto")),
        "dim_count": dim_count,
        "cell_width": cell_width,
        "min_node_layer_delta": min_node_layer_delta,
        "min_node_layers": min_node_layers,
        "max_node_layers": max_node_layers,
        "memory_count": len(cards),
        "cell_count": len(cells),
        "edge_count": len(edge_frames),
        "node_record_count": len(node_frames),
        "origin": origin,
        "triples": triples,
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    meta["build_latency_ms"] = (time.perf_counter() - started) * 1000.0
    meta["grid_bytes"] = grid_path.stat().st_size
    meta["payload_bytes"] = payload_path.stat().st_size
    meta["records_bytes"] = records_path.stat().st_size
    meta["edges_bytes"] = edges_path.stat().st_size
    return meta


class RopeDeltaGrid:
    def __init__(self, meta: dict[str, Any]):
        self.meta = meta
        self.grid_path = Path(meta["grid_path"])
        self.payload_path = Path(meta["payload_path"])
        self.records_path = Path(meta["records_path"])
        self.edges_path = Path(meta["edges_path"])
        self.grid_handle = self.grid_path.open("rb")
        self.grid_data = mmap.mmap(self.grid_handle.fileno(), 0, access=mmap.ACCESS_READ)
        self.payload_handle = self.payload_path.open("rb")
        self.payload_data = mmap.mmap(self.payload_handle.fileno(), 0, access=mmap.ACCESS_READ)
        self.records_handle = self.records_path.open("rb")
        self.records_data = mmap.mmap(self.records_handle.fileno(), 0, access=mmap.ACCESS_READ)
        self.edges_handle = self.edges_path.open("rb")
        self.edges_data = (
            mmap.mmap(self.edges_handle.fileno(), 0, access=mmap.ACCESS_READ)
            if self.edges_path.stat().st_size > 0
            else None
        )
        magic, version, layers, dim_count, memory_count, cell_count, cell_width = HEADER.unpack_from(self.grid_data, 0)
        if magic != GRID_MAGIC or version != GRID_VERSION:
            raise ValueError(f"unsupported rope delta grid: {self.grid_path}")
        self.layers = int(layers)
        self.dim_count = int(dim_count)
        self.memory_count = int(memory_count)
        self.cell_count = int(cell_count)
        self.cell_width = float(cell_width)
        self.cells_offset = HEADER.size
        self.nodes_offset = self.cells_offset + self.cell_count * CELL.size
        self.cell_lookup: dict[tuple[int, int, int, int], int] = {}
        for cell_index in range(self.cell_count):
            _, layer, x, y, z, _, _, _, _, _ = self.cell(cell_index)
            self.cell_lookup[(layer, x, y, z)] = cell_index
        self.payload_reads = 0
        self.node_reads = 0
        self.edge_reads = 0

    def close(self) -> None:
        if self.edges_data is not None:
            self.edges_data.close()
        self.edges_handle.close()
        self.records_data.close()
        self.records_handle.close()
        self.payload_data.close()
        self.payload_handle.close()
        self.grid_data.close()
        self.grid_handle.close()

    def cell(self, index: int) -> tuple[int, int, int, int, int, int, int, int, int, int]:
        return CELL.unpack_from(self.grid_data, self.cells_offset + index * CELL.size)

    def node(self, index: int) -> tuple[int, int, int, int, int, int, int, int, int, int, int, int, float, float]:
        self.node_reads += 1
        return NODE.unpack_from(self.grid_data, self.nodes_offset + index * NODE.size)

    def payload_record(self, memory_index: int) -> tuple[int, int, int, int, int, int, int]:
        record_offset = memory_index * PAYLOAD_RECORD.size
        return PAYLOAD_RECORD.unpack_from(self.records_data, record_offset)

    def payload(self, memory_index: int) -> tuple[str, str]:
        self.payload_reads += 1
        _, text_offset, text_size, id_offset, id_size, _, _ = self.payload_record(memory_index)
        node_id = self.payload_data[id_offset : id_offset + id_size].decode("utf-8")
        text = self.payload_data[text_offset : text_offset + text_size].decode("utf-8")
        return node_id, text

    def edges(self, memory_index: int) -> list[tuple[int, int, float, float, float]]:
        if self.edges_data is None:
            return []
        _, _, _, _, _, edge_offset, edge_count = self.payload_record(memory_index)
        out = []
        for local in range(edge_count):
            self.edge_reads += 1
            out.append(EDGE.unpack_from(self.edges_data, (edge_offset + local) * EDGE.size))
        return out


def neighborhood_offsets(radius: int) -> list[tuple[int, int, int]]:
    offsets = []
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                distance = abs(dx) + abs(dy) + abs(dz)
                if distance <= radius:
                    offsets.append((dx, dy, dz))
    return sorted(offsets, key=lambda item: (abs(item[0]) + abs(item[1]) + abs(item[2]), item))


def query_embedding_for(row: dict[str, Any], backend: Any) -> list[float]:
    task = row.get("retrieval_task") or {}
    query = str(task.get("query") or row["anchor"]["text"])
    embedding = task.get("query_embedding")
    if embedding:
        return [float(value) for value in embedding]
    return [float(value) for value in backend.embed_one(query)]


def score_node(
    node: tuple[int, int, int, int, int, int, int, int, int, int, int, int, float, float],
    query_mask: int,
    query_project_hash: int,
    distance: int,
) -> tuple[int, float]:
    memory_index, _, project_hash, mask, _, _, _, _, _, _, _, flags, importance, state = node
    grid_score = 1.0 / (1.0 + float(distance))
    activation = mask_overlap(query_mask, int(mask))
    same_project = 1.0 if int(project_hash) == query_project_hash else 0.0
    conflict_penalty = 0.08 if flags & 2 else 0.0
    ignored_penalty = 0.05 if flags & 4 else 0.0
    stale_penalty = 0.05 if flags & 8 else 0.0
    score = (
        0.45 * grid_score
        + 0.25 * activation
        + 0.13 * same_project
        + 0.08 * float(importance)
        + float(state)
        - conflict_penalty
        - ignored_penalty
        - stale_penalty
    )
    return int(memory_index), score


def search_rope_delta_grid(row: dict[str, Any], backend: Any, meta: dict[str, Any], args: argparse.Namespace) -> tuple[Ranked, dict[str, float], set[str]]:
    task = row.get("retrieval_task") or {}
    query = str(task.get("query") or row["anchor"]["text"])
    query_embedding = query_embedding_for(row, backend)
    query_mask = activation_mask_for_text(query)
    query_project_hash = fnv1a64(project_key(row["anchor"]))
    radius = max(0, int(args.radius))
    offsets = neighborhood_offsets(radius)
    started = time.perf_counter()
    index = RopeDeltaGrid(meta)
    try:
        triples = [tuple(item) for item in meta["triples"]]
        origin = [float(value) for value in meta["origin"]]
        active_layers, _ = selected_layers(
            query_embedding,
            origin,
            triples,
            float(args.min_layer_delta),
            int(getattr(args, "min_query_layers", 1) or 0),
            int(getattr(args, "max_query_layers", 0) or 0),
        )
        best: dict[int, float] = {}
        votes: dict[int, int] = {}
        visited_nodes: set[int] = set()
        cells_touched = 0
        max_cell_scan = int(getattr(args, "max_cell_scan", 0) or 0)
        for layer in active_layers:
            triple = triples[layer]
            qx, qy, qz = quantized_point(query_embedding, origin, triple, index.cell_width)
            for dx, dy, dz in offsets:
                cell_index = index.cell_lookup.get((layer, qx + dx, qy + dy, qz + dz))
                if cell_index is None:
                    continue
                cells_touched += 1
                _, _, _, _, _, _, head, _, _, _ = index.cell(cell_index)
                current = head
                distance = abs(dx) + abs(dy) + abs(dz)
                cell_scans = 0
                while current != NO_PTR:
                    if max_cell_scan > 0 and cell_scans >= max_cell_scan:
                        break
                    cell_scans += 1
                    if current in visited_nodes:
                        node = index.node(current)
                        current = int(node[10])
                        continue
                    visited_nodes.add(current)
                    node = index.node(current)
                    memory_index, score = score_node(node, query_mask, query_project_hash, distance)
                    best[memory_index] = max(best.get(memory_index, -99.0), score)
                    votes[memory_index] = votes.get(memory_index, 0) + 1
                    current = int(node[10])

        scored = [(memory_index, score + 0.035 * max(0, votes.get(memory_index, 1) - 1)) for memory_index, score in best.items()]
        scored.sort(key=lambda item: (-item[1], item[0]))
        edge_expansions = 0
        frontier = [(memory_index, score, [memory_index]) for memory_index, score in scored[: max(0, int(args.edge_seed_count))]]
        for depth in range(max(0, int(args.graph_depth))):
            next_frontier = []
            for memory_index, current_score, path in frontier:
                for target_index, edge_mask, weight, confidence, type_boost in index.edges(memory_index):
                    if target_index in path:
                        continue
                    hop_gain = (
                        float(weight)
                        * float(type_boost)
                        * (0.70 + 0.45 * activation_overlap(query_mask, int(edge_mask)))
                        * (0.70 + 0.30 * float(confidence))
                        / (1.25 + depth)
                    )
                    score = 0.68 * current_score + hop_gain
                    if score > best.get(target_index, -99.0):
                        best[target_index] = score
                        votes[target_index] = votes.get(target_index, 0) + 1
                        edge_expansions += 1
                        next_frontier.append((target_index, score, path + [target_index]))
            frontier = next_frontier
        scored = [(memory_index, score + 0.035 * max(0, votes.get(memory_index, 1) - 1)) for memory_index, score in best.items()]
        scored.sort(key=lambda item: (-item[1], item[0]))
        fetch = scored[: max(1, int(args.final_fetch))]
        ranked = []
        for memory_index, score in fetch:
            node_id, text = index.payload(memory_index)
            ranked.append((node_id, score, text))
        stats = {
            "latency_ms": (time.perf_counter() - started) * 1000.0,
            "cells_touched": float(cells_touched),
            "active_query_layers": float(len(active_layers)),
            "skipped_layers": float(len(triples) - len(active_layers)),
            "node_records_read": float(index.node_reads),
            "edge_reads": float(index.edge_reads),
            "edge_expansions": float(edge_expansions),
            "raw_final_candidate_count": float(len(best)),
            "final_candidate_count": float(len(ranked)),
            "payload_reads": float(index.payload_reads),
            "unique_nodes_read": float(len(visited_nodes)),
        }
        return ranked, stats, set()
    finally:
        index.close()


def determinism_check(row: dict[str, Any], backend: Any, case_dir: Path, args: argparse.Namespace) -> tuple[Ranked, dict[str, float]]:
    meta = build_rope_delta_grid(row, backend, case_dir / "determinism_a", args)
    raw_ranked, stats, protected_ids = search_rope_delta_grid(row, backend, meta, args)
    ranked, stats = compact_output(row, raw_ranked, protected_ids, stats, args)
    expected = ranked_signature(raw_ranked)
    mismatches = 0
    repeat_latency = []
    for _ in range(max(0, int(args.determinism_repeats) - 1)):
        repeated, repeated_stats, _ = search_rope_delta_grid(row, backend, meta, args)
        repeat_latency.append(float(repeated_stats.get("latency_ms") or 0.0))
        if ranked_signature(repeated) != expected:
            mismatches += 1
    rebuilt_meta = build_rope_delta_grid(row, backend, case_dir / "determinism_b", args)
    rebuilt, rebuild_stats, _ = search_rope_delta_grid(row, backend, rebuilt_meta, args)
    if ranked_signature(rebuilt) != expected:
        mismatches += 1
    out = dict(stats)
    out["determinism_mismatches"] = float(mismatches)
    out["determinism_repeat_latency_ms"] = average(repeat_latency)
    out["determinism_rebuild_latency_ms"] = float(rebuild_stats.get("latency_ms") or 0.0)
    return ranked, out


def evaluate_one(row: dict[str, Any], ranked: Ranked, stats: dict[str, float], args: argparse.Namespace) -> dict[str, float]:
    out = {
        "latency_ms": float(stats.get("latency_ms") or 0.0),
        "unique_nodes_read": float(stats.get("unique_nodes_read") or 0.0),
        "payload_reads": float(stats.get("payload_reads") or 0.0),
        "node_records_read": float(stats.get("node_records_read") or 0.0),
        "edge_reads": float(stats.get("edge_reads") or 0.0),
        "edge_expansions": float(stats.get("edge_expansions") or 0.0),
        "cells_touched": float(stats.get("cells_touched") or 0.0),
        "active_query_layers": float(stats.get("active_query_layers") or 0.0),
        "skipped_layers": float(stats.get("skipped_layers") or 0.0),
        "raw_final_candidate_count": float(stats.get("raw_final_candidate_count") or 0.0),
        "final_candidate_count": float(stats.get("final_candidate_count") or 0.0),
        "vector_index_scan_count": float(stats.get("vector_index_scan_count") or 0.0),
        "query_embedding_ms": float(stats.get("query_embedding_ms") or 0.0),
        "query_token_ms": float(stats.get("query_token_ms") or 0.0),
        "routing_ms": float(stats.get("routing_ms") or 0.0),
        "layer_zero_collision_ms": float(stats.get("layer_zero_collision_ms") or 0.0),
        "candidate_filter_ms": float(stats.get("candidate_filter_ms") or 0.0),
        "fallback_ms": float(stats.get("fallback_ms") or 0.0),
        "candidate_cap_ms": float(stats.get("candidate_cap_ms") or 0.0),
        "candidate_score_ms": float(stats.get("candidate_score_ms") or 0.0),
        "vector_candidate_latency_ms": float(stats.get("vector_candidate_latency_ms") or 0.0),
        "vector_candidate_count": float(stats.get("vector_candidate_count") or 0.0),
        "token_candidate_latency_ms": float(stats.get("token_candidate_latency_ms") or 0.0),
        "token_candidate_count": float(stats.get("token_candidate_count") or 0.0),
        "union_candidate_count": float(stats.get("union_candidate_count") or 0.0),
        "determinism_mismatches": float(stats.get("determinism_mismatches") or 0.0),
        "deterministic": 1.0 if float(stats.get("determinism_mismatches") or 0.0) == 0.0 else 0.0,
    }
    out.update({f"retrieval_{key}": value for key, value in evaluate_ranked(row, ranked, args.top_k, args.budget).items()})
    out.update({f"multihop_{key}": value for key, value in multihop_metrics(row, ranked, args.top_k, args.budget).items()})
    out.update({f"context_{key}": value for key, value in context_decoy_metrics(row, ranked, args.budget).items()})
    return out


def evaluate_growth(base: Ranked, grown: Ranked, grown_row: dict[str, Any], grown_stats: dict[str, float], args: argparse.Namespace) -> dict[str, float]:
    out = evaluate_one(grown_row, grown, grown_stats, args)
    out["growth_relevant_retention"] = relevant_retention(base, grown, grown_row)
    out["growth_topn_retention"] = top_retention(base, grown, args.stability_top_n)
    out["growth_existing_position_change_rate"] = changed_existing_positions(base, grown, args.stability_top_n)
    return out


def metric(result: dict[str, Any], section: str, key: str, stat: str = "avg", scenario: str = "") -> float:
    if section == "baseline":
        return float(result["baseline"].get(key, {}).get(stat, 0.0))
    return float(result["growth"].get(scenario, {}).get(key, {}).get(stat, 0.0))


def regression_failures(result: dict[str, Any], args: argparse.Namespace) -> list[str]:
    failures = []
    if metric(result, "baseline", "deterministic") < 1.0:
        failures.append("baseline determinism mismatch")
    if metric(result, "baseline", "retrieval_context_recall") < args.min_recall:
        failures.append("baseline recall below threshold")
    if metric(result, "baseline", "retrieval_context_precision") < args.min_precision:
        failures.append("baseline precision below threshold")
    if metric(result, "baseline", "latency_ms", "p95") > args.max_p95_ms:
        failures.append("baseline p95 latency above threshold")
    for scenario in args.growth_scenarios:
        if metric(result, "growth", "deterministic", scenario=scenario) < 1.0:
            failures.append(f"{scenario} determinism mismatch")
        if metric(result, "growth", "retrieval_context_recall", scenario=scenario) < args.min_recall:
            failures.append(f"{scenario} recall below threshold")
        if metric(result, "growth", "retrieval_context_precision", scenario=scenario) < args.min_precision:
            failures.append(f"{scenario} precision below threshold")
        if metric(result, "growth", "growth_relevant_retention", scenario=scenario) < args.min_growth_retention:
            failures.append(f"{scenario} relevant retention below threshold")
        if metric(result, "growth", "growth_topn_retention", scenario=scenario) < args.min_topn_retention:
            failures.append(f"{scenario} topN retention below threshold")
        if metric(result, "growth", "latency_ms", "p95", scenario=scenario) > args.max_p95_ms:
            failures.append(f"{scenario} p95 latency above threshold")
    return failures


def run(args: argparse.Namespace) -> dict[str, Any]:
    backend = build_embedding_backend(args)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    baseline_rows: list[dict[str, float]] = []
    growth_rows: dict[str, list[dict[str, float]]] = {scenario: [] for scenario in args.growth_scenarios}
    for case_index in range(args.cases):
        seed = args.seed + case_index
        case_dir = work_dir / f"case_{seed}"
        row = ensure_backend_embeddings(build_large_pool_case(seed, args.pool_size), backend)
        base_ranked, base_stats = determinism_check(row, backend, case_dir, args)
        baseline_rows.append(evaluate_one(row, base_ranked, base_stats, args))
        for scenario in args.growth_scenarios:
            grown = grow_row(row, scenario, args.growth_count, seed)
            grown = ensure_backend_embeddings(grown, backend)
            grown_ranked, grown_stats = determinism_check(grown, backend, case_dir / f"growth_{scenario}", args)
            growth_rows[scenario].append(evaluate_growth(base_ranked, grown_ranked, grown, grown_stats, args))
        if (case_index + 1) % max(1, args.log_every) == 0:
            print(
                json.dumps(
                    {
                        "case": case_index + 1,
                        "baseline_recall": average([row["retrieval_context_recall"] for row in baseline_rows]),
                        "baseline_precision": average([row["retrieval_context_precision"] for row in baseline_rows]),
                        "baseline_latency_p95": quantile([row["latency_ms"] for row in baseline_rows], 0.95),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    result = {
        "benchmark": "rope_delta_grid",
        "embedding_backend": backend.name,
        "cases": args.cases,
        "pool_size": args.pool_size,
        "growth_count": args.growth_count,
        "growth_scenarios": args.growth_scenarios,
        "determinism_repeats": args.determinism_repeats,
        "layers": args.layers,
        "layer_schedule": args.layer_schedule,
        "dim_count": args.dim_count,
        "cell_width": args.cell_width,
        "radius": args.radius,
        "min_layer_delta": args.min_layer_delta,
        "max_cell_scan": args.max_cell_scan,
        "min_query_layers": args.min_query_layers,
        "max_query_layers": args.max_query_layers,
        "min_node_layer_delta": args.min_node_layer_delta,
        "min_node_layers": args.min_node_layers,
        "max_node_layers": args.max_node_layers,
        "edge_seed_count": args.edge_seed_count,
        "graph_depth": args.graph_depth,
        "final_fetch": args.final_fetch,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "baseline": aggregate(baseline_rows),
        "growth": {scenario: aggregate(rows) for scenario, rows in growth_rows.items()},
    }
    failures = regression_failures(result, args)
    result["failures"] = failures
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_md:
        write_markdown(result, Path(args.output_md))
    if args.fail_on_regression and failures:
        raise SystemExit("regression thresholds failed: " + "; ".join(failures))
    return result


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# Rope Delta Grid",
        "",
        f"- backend: `{result['embedding_backend']}`",
        f"- cases: `{result['cases']}`",
        f"- pool_size: `{result['pool_size']}`",
        f"- growth_count: `{result['growth_count']}`",
        f"- layers: `{result['layers']}`",
        f"- layer_schedule: `{result['layer_schedule']}`",
        f"- dim_count: `{result['dim_count']}`",
        f"- cell_width: `{result['cell_width']}`",
        f"- radius: `{result['radius']}`",
        f"- min_layer_delta: `{result['min_layer_delta']}`",
        f"- max_cell_scan: `{result['max_cell_scan']}`",
        f"- min_query_layers: `{result['min_query_layers']}`",
        f"- max_query_layers: `{result['max_query_layers']}`",
        f"- min_node_layer_delta: `{result['min_node_layer_delta']}`",
        f"- min_node_layers: `{result['min_node_layers']}`",
        f"- max_node_layers: `{result['max_node_layers']}`",
        f"- edge_seed_count: `{result['edge_seed_count']}`",
        f"- graph_depth: `{result['graph_depth']}`",
        f"- final_fetch: `{result['final_fetch']}`",
        f"- elapsed_seconds: `{result['elapsed_seconds']}`",
        "",
        "| section | recall | precision | latency p95 ms | deterministic | payload reads p95 | nodes read p95 | raw candidates p95 | relevant retention | topN retention |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    def add_row(name: str, values: dict[str, dict[str, float]]) -> None:
        lines.append(
            f"| {name} | "
            f"{values.get('retrieval_context_recall', {}).get('avg', 0.0):.4f} | "
            f"{values.get('retrieval_context_precision', {}).get('avg', 0.0):.4f} | "
            f"{values.get('latency_ms', {}).get('p95', 0.0):.2f} | "
            f"{values.get('deterministic', {}).get('avg', 0.0):.4f} | "
            f"{values.get('payload_reads', {}).get('p95', 0.0):.2f} | "
            f"{values.get('node_records_read', {}).get('p95', 0.0):.2f} | "
            f"{values.get('raw_final_candidate_count', {}).get('p95', 0.0):.2f} | "
            f"{values.get('growth_relevant_retention', {}).get('avg', 0.0):.4f} | "
            f"{values.get('growth_topn_retention', {}).get('avg', 0.0):.4f} |"
        )

    add_row("baseline", result["baseline"])
    for scenario, values in result["growth"].items():
        add_row(scenario, values)
    if result.get("failures"):
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- {failure}" for failure in result["failures"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=5)
    parser.add_argument("--pool-size", type=int, default=5000)
    parser.add_argument("--growth-count", type=int, default=1000)
    parser.add_argument("--growth-scenarios", type=parse_scenarios, default=list(GROWTH_SCENARIOS))
    parser.add_argument("--seed", type=int, default=62000)
    parser.add_argument("--work-dir", default="artifacts/rope_delta_grid")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--budget", type=int, default=90)
    parser.add_argument("--compact-limit", type=int, default=3)
    parser.add_argument("--stability-top-n", type=int, default=64)
    parser.add_argument("--determinism-repeats", type=int, default=2)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--layer-schedule", choices=["auto", "consecutive", "spread"], default="auto")
    parser.add_argument("--dim-count", type=int, default=64)
    parser.add_argument("--cell-width", type=float, default=0.03125)
    parser.add_argument("--radius", type=int, default=0)
    parser.add_argument("--min-layer-delta", type=float, default=0.02)
    parser.add_argument("--max-cell-scan", type=int, default=0)
    parser.add_argument("--min-query-layers", type=int, default=1)
    parser.add_argument("--max-query-layers", type=int, default=0)
    parser.add_argument("--min-node-layer-delta", type=float, default=0.0)
    parser.add_argument("--min-node-layers", type=int, default=1)
    parser.add_argument("--max-node-layers", type=int, default=0)
    parser.add_argument("--edge-seed-count", type=int, default=48)
    parser.add_argument("--graph-depth", type=int, default=2)
    parser.add_argument("--final-fetch", type=int, default=96)
    parser.add_argument("--embedding-backend", choices=["hash", "hippo"], default="hash")
    parser.add_argument("--hippo-checkpoint", default="")
    parser.add_argument("--hippo-encoder-src", default="")
    parser.add_argument("--hippo-max-length", type=int, default=128)
    parser.add_argument("--hippo-batch-size", type=int, default=128)
    parser.add_argument("--device", default="")
    parser.add_argument("--min-recall", type=float, default=0.80)
    parser.add_argument("--min-precision", type=float, default=0.30)
    parser.add_argument("--min-growth-retention", type=float, default=0.90)
    parser.add_argument("--min-topn-retention", type=float, default=0.80)
    parser.add_argument("--max-p95-ms", type=float, default=200.0)
    parser.add_argument("--fail-on-regression", action="store_true")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
