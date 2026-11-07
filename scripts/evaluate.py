"""检索评测：python -m scripts.evaluate [--k 5]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from rag import config  # noqa: E402
from rag.index import Index  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=config.TOP_K)
    parser.add_argument("--file", default=str(ROOT / "eval" / "questions.jsonl"))
    args = parser.parse_args()

    index = Index.load()
    cases = [json.loads(line) for line in Path(args.file).read_text(encoding="utf-8").splitlines() if line.strip()]

    hit = 0
    detail = []
    for case in cases:
        results = index.search(case["q"], args.k)
        texts = [c.text for c, _ in results]
        matched = any(any(kw in t for kw in case["expect"]) for t in texts)
        hit += int(matched)
        top_source = f"{results[0][0].source} · {results[0][0].section}" if results else "-"
        detail.append((case["q"], matched, top_source))

    print(f"Hit@{args.k} = {hit}/{len(cases)} = {hit / len(cases):.0%}\n")
    for q, ok, src in detail:
        print(f"[{'OK ' if ok else 'MISS'}] {q}  ->  {src}")


if __name__ == "__main__":
    main()
