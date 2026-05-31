from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/synthetic/memories.jsonl")
    parser.add_argument("--output-dir", default="artifacts/librarian")
    args = parser.parse_args()
    print("training stub is deprecated; use python -m python.training.train_librarian")
    print(f"dataset={args.dataset}")
    print(f"output_dir={args.output_dir}")
    print("replace this with the PyTorch neighborhood transformer training loop")


if __name__ == "__main__":
    main()
