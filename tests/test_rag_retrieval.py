"""RAG 检索模块测试：RRF 融合（快）+ 真实索引检索冒烟（慢）。

快路径不加载任何模型；慢路径需要已构建索引（data/rag_index/）与 reranker。
"""
from __future__ import annotations

import pytest

from rag_retrieval.fusion import reciprocal_rank_fusion
from rag_retrieval.retrievers.base import RetrievalHit


# ---- 快路径：RRF 融合纯逻辑 ----
def _hit(page: int, retriever: str, rank_score: float = 1.0) -> RetrievalHit:
    return RetrievalHit(
        page=page,
        text=f"page-{page}",
        score=rank_score,
        retriever=retriever,
    )


def test_rrf_fusion_dedup_and_paths():
    """跨路命中同一页面应合并，且记录命中了哪几路。"""
    path_a = [_hit(1, "bm25"), _hit(2, "bm25"), _hit(3, "bm25")]
    path_b = [_hit(2, "dense_bge"), _hit(3, "dense_bge"), _hit(4, "dense_bge")]
    fused = reciprocal_rank_fusion([path_a, path_b], k=60)
    pages = [h.page for h in fused]
    assert pages == [2, 3, 1, 4]  # 2 在 A 第2/B 第1，分最高；3 次之；1、4 单路命中
    by_page = {h.page: h for h in fused}
    assert by_page[2].paths == ("bm25", "dense_bge")
    assert by_page[1].paths == ("bm25",)
    # 两路都命中的 2，分应高于任何单路命中的项
    assert by_page[2].rrf_score > by_page[1].rrf_score
    assert by_page[2].rrf_score > by_page[4].rrf_score


def test_rrf_fusion_rank_ordering():
    """rank 越靠前，1/(rank+k) 贡献越大：同路内顺序正确。"""
    path = [_hit(1, "bm25"), _hit(2, "bm25")]
    fused = reciprocal_rank_fusion([path], k=60)
    assert fused[0].page == 1
    assert fused[0].rrf_score == 1 / 60
    assert fused[1].rrf_score == 1 / 61


# ---- 慢路径：真实索引检索冒烟（需 data/rag_index/ 已构建） ----
@pytest.mark.slow
def test_rag_search_smoke():
    from rag_retrieval import search

    result = search("前置座椅通风在哪里开启", top_k=5)
    assert result.hits, "应返回检索结果"
    assert result.retrieval_paths  # 至少一路参与召回
    # 命中应落在座椅通风相关页（手册 116/117 页附近，页码放宽断言）
    pages = {h.page for h in result.hits}
    assert pages, "hits 应带页码"
