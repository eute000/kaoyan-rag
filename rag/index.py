"""索引与检索：中文二元组 BM25 + 可选稠密向量，用 RRF 融合。零第三方依赖。"""
from __future__ import annotations

import json
import math
import re
import urllib.request
from collections import Counter
from pathlib import Path

from . import config
from .loader import Chunk

_CJK = re.compile(r"[\u4e00-\u9fff]+")
_ASCII = re.compile(r"[a-zA-Z0-9_]+")


def tokenize(text: str) -> list[str]:
    """中文取二元组、英文数字取整词。不依赖分词词典，对术语召回更稳。"""
    text = text.lower()
    tokens: list[str] = []
    for run in _CJK.findall(text):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i:i + 2] for i in range(len(run) - 1))
    tokens.extend(_ASCII.findall(text))
    return tokens


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1, self.b = k1, b
        self.n = len(docs)
        self.avgdl = sum(len(d) for d in docs) / max(1, self.n)
        self.tf = [Counter(d) for d in docs]
        df: Counter = Counter()
        for d in docs:
            df.update(set(d))
        self.idf = {t: math.log((self.n - c + 0.5) / (c + 0.5) + 1) for t, c in df.items()}

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for i, tf in enumerate(self.tf):
            dl = len(self.docs[i]) or 1
            s = 0.0
            for t in query:
                f = tf.get(t)
                if not f:
                    continue
                idf = self.idf.get(t, 0.0)
                s += idf * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            out.append(s)
        return out


def embed_texts(texts: list[str], batch_size: int = 16) -> list[list[float]]:
    """调用 SiliconFlow 的 bge-m3 生成向量；没配 key 时抛错由上层降级。"""
    if not config.SILICONFLOW_API_KEY:
        raise RuntimeError("未配置 SILICONFLOW_API_KEY")

    vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        body = json.dumps({"model": config.EMBED_MODEL, "input": batch, "encoding_format": "float"}).encode()
        req = urllib.request.Request(
            f"{config.EMBED_BASE_URL}/embeddings",
            data=body,
            headers={"Authorization": f"Bearer {config.SILICONFLOW_API_KEY}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())["data"]
        vectors.extend(item["embedding"] for item in data)
    return vectors


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec)) or 1e-8
    return [v / norm for v in vec]


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class Index:
    def __init__(self, chunks: list[Chunk], vectors: list[list[float]] | None = None):
        self.chunks = chunks
        self.tokens = [tokenize(c.text) for c in chunks]
        self.bm25 = BM25(self.tokens)
        self.vectors = [_normalize(v) for v in vectors] if vectors else None

    @classmethod
    def build(cls, chunks: list[Chunk], dense: bool = False) -> "Index":
        vectors = None
        if dense:
            vectors = embed_texts([c.text for c in chunks])
        return cls(chunks, vectors)

    def save(self, path: Path | None = None) -> Path:
        path = path or config.INDEX_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "chunks": [c.to_dict() for c in self.chunks],
            "vectors": self.vectors,
            "meta": {"chunk_size": config.CHUNK_SIZE, "overlap": config.CHUNK_OVERLAP},
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "Index":
        path = path or config.INDEX_PATH
        payload = json.loads(path.read_text(encoding="utf-8"))
        chunks = [Chunk(**c) for c in payload["chunks"]]
        return cls(chunks, payload.get("vectors"))

    def _dense_scores(self, query: str) -> list[float] | None:
        if not self.vectors:
            return None
        try:
            qv = _normalize(embed_texts([query])[0])
        except Exception:
            return None
        return [_cosine(qv, v) for v in self.vectors]

    def search(self, query: str, k: int | None = None) -> list[tuple[Chunk, float]]:
        k = k or config.TOP_K
        sparse = self.bm25.scores(tokenize(query))
        dense = self._dense_scores(query)

        rankings = [sorted(range(len(self.chunks)), key=lambda i: -sparse[i])]
        if dense is not None:
            rankings.append(sorted(range(len(self.chunks)), key=lambda i: -dense[i]))

        # RRF 融合：只看名次不看原始分，避免 BM25 和余弦各自量纲不一致
        fused: dict[int, float] = {}
        for ranking in rankings:
            for rank, idx in enumerate(ranking[: max(k * 4, 20)]):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (config.RRF_K + rank + 1)

        top = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
        return [(self.chunks[i], round(s, 6)) for i, s in top]
