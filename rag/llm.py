"""DeepSeek 调用（OpenAI 兼容 HTTP 接口），用 urllib 实现流式，零第三方依赖。"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Iterator

from . import config

SYSTEM_PROMPT = (
    "你是考研资料问答助手。只能依据【资料】作答，不要编造资料之外的内容。\n"
    "规则：\n"
    "1. 每个结论后面用 [编号] 标注来源，编号对应资料片段的序号；\n"
    "2. 资料里没有的，直接回答“资料中没有提到”，并说明可能相关的方向；\n"
    "3. 用中文回答，条理清晰，涉及概念时先给定义再解释；\n"
    "4. 不要复述整段资料，只答用户问到的部分。"
)


def build_messages(question: str, contexts: list[str]) -> list[dict]:
    blocks = [f"[{i}] {ctx}" for i, ctx in enumerate(contexts, start=1)]
    user = (
        "【资料】\n" + "\n\n".join(blocks) + f"\n\n【问题】\n{question}\n\n"
        "请依据资料回答，并在结论后用 [编号] 标注来源。"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def stream_answer(question: str, contexts: list[str]) -> Iterator[str]:
    if not config.DEEPSEEK_API_KEY:
        yield "（未配置 DEEPSEEK_API_KEY，无法生成回答。把 key 写进项目根目录 .env 再问。）"
        return

    body = json.dumps({
        "model": config.CHAT_MODEL,
        "messages": build_messages(question, contexts),
        "temperature": config.TEMPERATURE,
        "stream": True,
    }).encode()

    req = urllib.request.Request(
        f"{config.DEEPSEEK_BASE_URL.rstrip('/')}/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {config.DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                choices = obj.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                piece = delta.get("content")
                if piece:
                    yield piece
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "ignore")[:200]
        yield f"\n\n[生成失败] HTTP {e.code}: {detail}"
    except Exception as e:  # 网络/超时
        yield f"\n\n[生成失败] {type(e).__name__}: {e}"


def answer(question: str, contexts: list[str]) -> str:
    return "".join(stream_answer(question, contexts))
