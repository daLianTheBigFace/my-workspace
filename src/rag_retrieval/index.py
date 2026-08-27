"""分块索引：启动时读盘，供各召回器共享。

build_index.py 一次性产出：
- chunks.json      [{id, page, text}]  id 全局自增，page 从 1 开始
- pages.json       [整页文本, ...]      1-based，供问答上下文用
- embeddings.npy   稠密向量，行序与 chunks 一致（bge 稠密召回用）
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .config import RagConfig


@dataclass(frozen=True)
class Chunk:
    id: int
    page: int  # 1-based 页码
    text: str


class RagIndex:
    """知识库的内存形态：分块 + 整页 + 稠密向量。"""

    def __init__(
        self,
        chunks: list[Chunk],
        pages: list[str],
        embeddings: np.ndarray | None,
    ):
        self.chunks = chunks
        self.pages = pages
        self.embeddings = embeddings

    @property
    def texts(self) -> list[str]:
        return [c.text for c in self.chunks]

    def page_text(self, page: int) -> str:
        """取整页文本（1-based）。"""
        return self.pages[page - 1]


@lru_cache(maxsize=1)
def load_index() -> RagIndex:
    """读盘加载索引（进程内单例，重复调用返回同一对象）。"""
    cfg = RagConfig()
    index_dir = Path(cfg.index_dir)
    with open(index_dir / "chunks.json", encoding="utf-8") as f:
        raw_chunks = json.load(f)
    chunks = [
        Chunk(id=c["id"], page=c["page"], text=c["text"]) for c in raw_chunks
    ]
    with open(index_dir / "pages.json", encoding="utf-8") as f:
        pages = json.load(f)
    emb_path = index_dir / "embeddings.npy"
    embeddings = np.load(emb_path) if emb_path.exists() else None
    return RagIndex(chunks=chunks, pages=pages, embeddings=embeddings)


def index_info() -> dict:
    """索引元信息（供 GET /rag/index_info）。"""
    index = load_index()
    return {
        "n_chunks": len(index.chunks),
        "n_pages": len(index.pages),
        "has_dense": index.embeddings is not None,
    }
