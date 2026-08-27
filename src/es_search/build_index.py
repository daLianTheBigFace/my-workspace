"""把汽车知识手册灌进 ES（python -m es_search.build_index）。

数据来源（二选一）：
1. 优先复用 rag_retrieval 已建好的 data/rag_index/（chunks.json + embeddings.npy，
   3743 chunk 的 512 维 bge 向量已算好）——秒级建库，不重复编码。
2. 缺失则自行 pdfplumber 解析 + sentence_bert 编码（自包含兜底，稍慢）。

建完可 curl http://localhost:9200/car_manual/_count 验证。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from elasticsearch.helpers import bulk

from .config import EsConfig
from .es_client import ensure_index, get_client


def _load_rag_data(cfg: EsConfig) -> tuple[list[dict], np.ndarray] | None:
    """读 data/rag_index 的 chunks + 向量；文件不全返回 None。"""
    index_dir = Path(cfg.rag_index_dir)
    chunks_path = index_dir / "chunks.json"
    emb_path = index_dir / "embeddings.npy"
    if chunks_path.exists() and emb_path.exists():
        with open(chunks_path, encoding="utf-8") as f:
            chunks = json.load(f)
        embeddings = np.load(emb_path)
        return chunks, embeddings
    return None


def _parse_pdf(cfg: EsConfig) -> tuple[list[dict], np.ndarray]:
    """兜底：直接解析 PDF + 用 bge 编码 chunk。"""
    import pdfplumber
    from sentence_bert import get_model

    chunks: list[dict] = []
    texts: list[str] = []
    cid = 0
    with pdfplumber.open(cfg.pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            for i in range(0, len(text), cfg.chunk_size):
                chunk_text = text[i : i + cfg.chunk_size]
                chunks.append({"id": cid, "page": page_idx + 1, "text": chunk_text})
                texts.append(chunk_text)
                cid += 1
    model = get_model()
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    return chunks, embeddings


def build() -> int:
    """建索引并灌数据，返回写入条数。"""
    cfg = EsConfig()
    ensure_index()

    data = _load_rag_data(cfg)
    if data is None:
        print(f"未找到 {cfg.rag_index_dir}，自行解析 PDF + 编码…", flush=True)
        chunks, embeddings = _parse_pdf(cfg)
    else:
        chunks, embeddings = data
        print(f"复用 {cfg.rag_index_dir}（{len(chunks)} chunk）", flush=True)

    n = len(chunks)
    if embeddings.shape[0] != n:
        raise RuntimeError(f"chunk {n} 条但向量 {embeddings.shape[0]} 条，索引数据不一致")

    docs = (
        {
            "_index": cfg.index_name,
            "_source": {
                "chunk_id": chunks[i]["id"],
                "page": chunks[i]["page"],
                "text": chunks[i]["text"],
                "text_vector": embeddings[i].astype(float).tolist(),
            },
        }
        for i in range(n)
    )

    es = get_client()
    success, errors = bulk(es, docs, chunk_size=500)
    if errors:
        raise RuntimeError(f"bulk 写入 {len(errors)} 条失败，示例：{errors[0]}")
    return success


if __name__ == "__main__":
    cfg = EsConfig()
    count = build()
    print(f"已写入 {count} 条文档到 ES 索引 [{cfg.index_name}]", flush=True)
