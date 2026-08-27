"""一次性构建分块索引：PDF → 分块 → 稠密向量 → 写盘。

运行：uv run python -m rag_retrieval.build_index
算法照搬 week06（RAG101_05：定长 40 字分块 + bge 编码），week06 文件不改。
产物落在 data/rag_index/（git 忽略）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pdfplumber

from sentence_bert import get_model

from .config import RagConfig


def split_text_fixed_size(text: str, chunk_size: int) -> list[str]:
    """定长 chunk 划分（照搬 week06 RAG101_05）。"""
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def build(config: RagConfig | None = None) -> None:
    cfg = config or RagConfig()
    pdf_path = Path(cfg.pdf_path)
    if not pdf_path.exists():
        sys.exit(f"PDF 不存在：{pdf_path}")

    # 1. 逐页提取文本
    pages: list[str] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")

    # 2. 每页定长分块
    chunks: list[dict] = []
    for page_idx, text in enumerate(pages, start=1):
        for chunk_text in split_text_fixed_size(text, cfg.chunk_size):
            chunks.append({"id": len(chunks), "page": page_idx, "text": chunk_text})

    # 3. bge 稠密编码（与 RAG101_05 一致：normalize_embeddings=True）
    model = get_model()
    embeddings = model.encode(
        [c["text"] for c in chunks], normalize_embeddings=True, show_progress_bar=True
    )

    # 4. 写盘
    index_dir = Path(cfg.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    with open(index_dir / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=1)
    with open(index_dir / "pages.json", "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=1)
    np.save(index_dir / "embeddings.npy", embeddings)

    print(f"完成：{len(chunks)} 个 chunk，{len(pages)} 页 → {index_dir}")


if __name__ == "__main__":
    build()
