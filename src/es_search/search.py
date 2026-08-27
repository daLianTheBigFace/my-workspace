"""三大检索能力：全文检索 / 条件过滤 / 向量检索。

对外只暴露 full_text / filtered / vector 三个函数，API 层调用。
向量检索复用 sentence_bert.get_model()（bge-small-zh-v1.5），不重复加载。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from . import queries
from .config import EsConfig
from .es_client import ensure_index, get_client

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EsHit:
    chunk_id: int
    page: int  # 1-based 页码
    text: str  # 命中的 chunk 文本
    score: float  # 相关分


def _to_result(resp: dict) -> tuple[list[EsHit], int]:
    """把 ES 响应拆成 (hits, total)。"""
    total = int(resp.get("hits", {}).get("total", {}).get("value", 0))
    hits: list[EsHit] = []
    for h in resp.get("hits", {}).get("hits", []):
        src = h.get("_source", {})
        hits.append(
            EsHit(
                chunk_id=src.get("chunk_id"),
                page=src.get("page"),
                text=src.get("text", ""),
                score=float(h.get("_score", 0.0)),
            )
        )
    return hits, total


def _search(query_body: dict) -> tuple[list[EsHit], int]:
    """统一执行入口：确保索引存在后发查询。"""
    cfg = EsConfig()
    ensure_index()
    resp = get_client().search(index=cfg.index_name, **query_body)
    return _to_result(resp)


def full_text(
    query: str,
    size: int | None = None,
    match_type: str = "match",
) -> tuple[list[EsHit], int]:
    """全文检索（IK 分词）。match_type: match/phrase/fuzzy。"""
    size = size or EsConfig().default_size
    return _search(queries.full_text_query(query, size, match_type))


def filtered(
    query: str,
    page_from: int | None = None,
    page_to: int | None = None,
    size: int | None = None,
    match_type: str = "match",
) -> tuple[list[EsHit], int]:
    """条件过滤：全文 match + 页码范围。"""
    size = size or EsConfig().default_size
    return _search(queries.filter_query(query, page_from, page_to, size, match_type))


def vector(query: str, size: int | None = None) -> tuple[list[EsHit], int]:
    """向量检索：query 先用 bge 编码，再 knn。"""
    from sentence_bert import get_model

    size = size or EsConfig().default_size
    model = get_model()
    q_vec = model.encode([query], normalize_embeddings=True)[0].tolist()
    return _search(queries.vector_query(q_vec, size))
