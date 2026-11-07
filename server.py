"""零依赖 HTTP 服务：/ 给前端页面，/api/ask 流式返回检索结果与回答。

启动：python server.py   （默认 http://127.0.0.1:8000）
"""
from __future__ import annotations

import json
import mimetypes
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from rag import config  # noqa: E402
from rag.pipeline import build_default  # noqa: E402

WEB_DIR = ROOT / "web"
_pipe = None
_lock = threading.Lock()


def get_pipe():
    global _pipe
    if _pipe is None:
        with _lock:
            if _pipe is None:
                _pipe = build_default()
    return _pipe


class Handler(BaseHTTPRequestHandler):
    server_version = "kaoyan-rag/0.1"

    def log_message(self, fmt, *args):  # 静音默认日志
        pass

    # ---------- 静态 ----------
    def _serve_file(self, rel: str) -> None:
        path = (WEB_DIR / rel).resolve()
        if not str(path).startswith(str(WEB_DIR.resolve())) or not path.is_file():
            self.send_error(404)
            return
        data = path.read_bytes()
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", f"{ctype}; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self._serve_file("index.html")
        elif parsed.path == "/api/stats":
            pipe = get_pipe()
            body = json.dumps({
                "chunks": len(pipe.index.chunks),
                "dense": bool(pipe.index.vectors),
                "llm": config.has_llm(),
                "model": config.CHAT_MODEL,
            }, ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self._serve_file(parsed.path.lstrip("/"))

    # ---------- 问答 ----------
    def do_POST(self):
        if urlparse(self.path).path != "/api/ask":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self.send_error(400)
            return

        question = (payload.get("question") or "").strip()
        k = int(payload.get("k") or config.TOP_K)
        if not question:
            self.send_error(400)
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        def write_event(event: str, data: dict) -> None:
            chunk = f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
            self.wfile.write(chunk.encode("utf-8"))
            self.wfile.flush()

        pipe = get_pipe()
        hits = pipe.retrieve(question, k)
        write_event("sources", {
            "hits": [
                {
                    "n": h._n,
                    "source": h.chunk.source,
                    "section": h.chunk.section,
                    "score": h.score,
                    "text": h.chunk.text[:400],
                }
                for h in hits
            ]
        })

        for piece in pipe.stream(question, k)[1]:
            write_event("token", {"text": piece})

        write_event("done", {"ok": True})


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"kaoyan-rag 已启动： http://127.0.0.1:{port}")
    print(f"语料块数：{len(get_pipe().index.chunks)}｜稠密检索：{'开' if get_pipe().index.vectors else '关'}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")


if __name__ == "__main__":
    main()
