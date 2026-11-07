"""建索引：python -m scripts.build_index [--dense]"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from rag import config  # noqa: E402
from rag.index import Index  # noqa: E402
from rag.loader import load_docs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense", action="store_true", help="同时生成向量（需要 SILICONFLOW_API_KEY）")
    args = parser.parse_args()

    t0 = time.time()
    chunks = load_docs()
    print(f"切片完成：{len(chunks)} 块，用时 {time.time() - t0:.1f}s")
    for c in chunks[:5]:
        print(f"  {c.id} [{c.source}] {c.section} ({len(c.text)} 字)")

    dense = args.dense or config.has_dense()
    if dense and not config.has_dense():
        print("未配置 SILICONFLOW_API_KEY，跳过向量，仅建 BM25")
        dense = False

    t1 = time.time()
    index = Index.build(chunks, dense=dense)
    path = index.save()
    print(f"索引已保存：{path}（{'BM25 + 向量' if dense else '仅 BM25'}，用时 {time.time() - t1:.1f}s）")


if __name__ == "__main__":
    main()
