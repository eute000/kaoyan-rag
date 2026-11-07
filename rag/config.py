"""全局配置：路径、模型、检索参数。所有密钥优先读环境变量，其次读项目根目录 .env。"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "docs"
DATA_DIR = ROOT / "data"
INDEX_PATH = DATA_DIR / "index.json"


def _load_dotenv() -> None:
    """极简 .env 加载器：KEY=VALUE，忽略注释和空行，不覆盖已有环境变量。"""
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv()

# ---- 生成模型（DeepSeek，OpenAI 兼容协议）----
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
CHAT_MODEL = os.getenv("CHAT_MODEL", "deepseek-chat")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
MAX_CONTEXT_CHUNKS = int(os.getenv("MAX_CONTEXT_CHUNKS", "5"))

# ---- 可选稠密检索（SiliconFlow bge-m3）；不给 key 就只跑 BM25 ----
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
EMBED_BASE_URL = os.getenv("EMBED_BASE_URL", "https://api.siliconflow.cn/v1")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")

# ---- 切片参数 ----
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "400"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))

# ---- 检索参数 ----
TOP_K = int(os.getenv("TOP_K", "5"))
RRF_K = 60


def has_llm() -> bool:
    return bool(DEEPSEEK_API_KEY)


def has_dense() -> bool:
    return bool(SILICONFLOW_API_KEY)
