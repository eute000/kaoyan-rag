"""检索 + 生成的编排层。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from . import config, llm
from .index import Index
from .loader import Chunk


@dataclass
class Hit:
    chunk: Chunk
    score: float

    @property
    def label(self) -> str:
        return f"[{self._n}] {self.chunk.source} · {self.chunk.section}"

    _n: int = 0


class Pipeline:
    def __init__(self, index: Index):
        self.index = index

    @classmethod
    def load(cls) -> "Pipeline":
        return cls(Index.load())

    def retrieve(self, question: str, k: int | None = None) -> list[Hit]:
        hits = []
        for n, (chunk, score) in enumerate(self.index.search(question, k), start=1):
            hit = Hit(chunk=chunk, score=score)
            hit._n = n
            hits.append(hit)
        return hits

    def stream(self, question: str, k: int | None = None) -> tuple[list[Hit], Iterator[str]]:
        hits = self.retrieve(question, k)
        if not hits:
            return hits, iter(["资料库里没有检索到相关内容。"])
        contexts = [h.chunk.text for h in hits]
        return hits, llm.stream_answer(question, contexts)

    def answer(self, question: str, k: int | None = None) -> tuple[list[Hit], str]:
        hits, gen = self.stream(question, k)
        return hits, "".join(gen)


def build_default() -> Pipeline:
    """没有索引就现场建一个（仅 BM25），有 key 才顺带建向量。"""
    if config.INDEX_PATH.exists():
        return Pipeline.load()
    from .loader import load_docs

    chunks = load_docs()
    index = Index.build(chunks, dense=config.has_dense())
    index.save()
    return Pipeline(index)
