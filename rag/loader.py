"""文档加载与切片：按 Markdown 标题分节，再把长节按段落合并成带重叠的块。"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path

from . import config


@dataclass
class Chunk:
    id: str
    source: str          # 文件名
    section: str         # 所属标题路径
    text: str
    start: int           # 在原文中的起始偏移（便于回溯）

    def to_dict(self) -> dict:
        return asdict(self)


_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


def _split_sections(text: str) -> list[tuple[str, str, int]]:
    """返回 [(标题路径, 正文, 正文起始偏移)]，按出现的标题切分。"""
    lines = text.splitlines(keepends=True)
    sections: list[tuple[str, str, int]] = []
    stack: list[tuple[int, str]] = []
    buf: list[str] = []
    buf_start = 0
    offset = 0

    def flush() -> None:
        nonlocal buf
        body = "".join(buf).strip()
        if body:
            # 一级标题当文档标题，不进章节路径，避免每块都带一长串前缀
            path = " / ".join(t for lv, t in stack if lv > 1)
            if not path:
                path = " / ".join(t for lv, t in stack if lv == 1) or "（无标题）"
            sections.append((path, body, buf_start))
        buf = []

    for line in lines:
        m = _HEADING.match(line.rstrip("\n"))
        if m:
            flush()
            level = len(m.group(1))
            title = m.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            buf_start = offset + len(line)
        else:
            if not buf:
                buf_start = offset
            buf.append(line)
        offset += len(line)
    flush()
    return sections


def _chunk_section(section: str, body: str, base: int) -> list[tuple[str, int]]:
    """段落优先切分：先按空行分块，再按 CHUNK_SIZE 合并，块间保留 CHUNK_OVERLAP 重叠。"""
    paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    if not paras:
        return []
    out: list[tuple[str, int]] = []
    cur = ""
    cur_start = base
    for para in paras:
        if cur and len(cur) + len(para) + 2 > config.CHUNK_SIZE:
            out.append((cur.strip(), cur_start))
            tail = cur[-config.CHUNK_OVERLAP:] if config.CHUNK_OVERLAP else ""
            cur = (tail + "\n\n" + para) if tail else para
            cur_start = base + max(0, body.find(para))
        else:
            cur = (cur + "\n\n" + para) if cur else para
    if cur.strip():
        out.append((cur.strip(), cur_start))
    return out


def load_docs(docs_dir: Path | None = None) -> list[Chunk]:
    docs_dir = docs_dir or config.DOCS_DIR
    chunks: list[Chunk] = []
    files = sorted([p for p in docs_dir.glob("**/*") if p.suffix.lower() in {".md", ".txt"}])
    for path in files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for section, body, base in _split_sections(text):
            for i, (piece, start) in enumerate(_chunk_section(section, body, base)):
                chunks.append(Chunk(
                    id=f"{path.stem}-{len(chunks):04d}",
                    source=path.name,
                    section=section,
                    text=piece,
                    start=start,
                ))
    for idx, chunk in enumerate(chunks):
        chunk.id = f"c{idx:04d}"
    return chunks


if __name__ == "__main__":
    cs = load_docs()
    print(f"共 {len(cs)} 块")
    for c in cs[:3]:
        print("-", c.id, c.section, len(c.text))
