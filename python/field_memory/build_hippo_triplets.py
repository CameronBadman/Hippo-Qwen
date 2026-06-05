from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def prefixed(text: str, prefix: str) -> str:
    text = " ".join(str(text or "").split())
    if prefix and not text.startswith(prefix):
        return f"{prefix}{text}"
    return text


def candidate_text(item: dict[str, Any]) -> str:
    return " ".join(str(item.get("text") or "").split())


def build_triplets(args: argparse.Namespace) -> dict[str, Any]:
    rng = random.Random(args.seed)
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    row_count = 0
    triplet_count = 0
    skipped = 0
    with input_path.open("r", encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as sink:
        for row_index, line in enumerate(source):
            if not line.strip():
                continue
            row_count += 1
            row = json.loads(line)
            query = prefixed(str(row.get("query") or ""), args.query_prefix)
            positives = [candidate_text(item) for item in row.get("positives") or []]
            negatives = [candidate_text(item) for item in row.get("hard_negatives") or []]
            if args.include_random_negatives:
                negatives.extend(candidate_text(item) for item in row.get("random_negatives") or [])
            positives = [item for item in positives if item]
            negatives = [item for item in negatives if item]
            if not query or not positives or not negatives:
                skipped += 1
                continue
            local_rng = random.Random(args.seed + row_index)
            local_rng.shuffle(positives)
            local_rng.shuffle(negatives)
            positives = positives[: max(1, int(args.max_positives))]
            negatives = negatives[: max(1, int(args.max_negatives))]
            per_positive = max(1, int(args.negatives_per_positive))
            for positive_index, positive in enumerate(positives):
                for offset in range(per_positive):
                    negative = negatives[(positive_index * per_positive + offset) % len(negatives)]
                    payload = {
                        "anchor": query,
                        "positive": prefixed(positive, args.passage_prefix),
                        "negative": prefixed(negative, args.passage_prefix),
                    }
                    sink.write(json.dumps(payload, sort_keys=True) + "\n")
                    triplet_count += 1
            if args.shuffle_output_buffer > 0 and triplet_count >= args.shuffle_output_buffer:
                sink.flush()

    if args.shuffle:
        lines = output_path.read_text(encoding="utf-8").splitlines()
        rng.shuffle(lines)
        output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    result = {
        "input": str(input_path),
        "output": str(output_path),
        "rows": row_count,
        "triplets": triplet_count,
        "skipped_rows": skipped,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="artifacts/token_field_encoder/train.jsonl")
    parser.add_argument("--output", default="artifacts/hippo_encoder/memorycraft_triplets.jsonl")
    parser.add_argument("--max-positives", type=int, default=8)
    parser.add_argument("--max-negatives", type=int, default=32)
    parser.add_argument("--negatives-per-positive", type=int, default=2)
    parser.add_argument("--include-random-negatives", action="store_true")
    parser.add_argument("--query-prefix", default="query: ")
    parser.add_argument("--passage-prefix", default="passage: ")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--shuffle-output-buffer", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    build_triplets(args)


if __name__ == "__main__":
    main()
