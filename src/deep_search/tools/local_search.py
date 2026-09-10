"""local_search：复用 rag_retrieval.pipeline 查本地手册。

deep_search 的本地 tool：问题涉及车机手册时自动查本地。仅在 cfg.local_rounds 指定的
轮次（默认 round 1）与 web 并行跑，结果标注 source="local"。
"""
from __future__ import annotations

import logging

from ..schemas import SearchHit

logger = logging.getLogger(__name__)


def search(query: str, *, top_k: int = 5) -> list[SearchHit]:
    """复用 rag_retrieval.pipeline.search 查本地手册，转 SearchHit(source="local")。

    同步（rag 检索是 CPU/IO 绑定），节点里用 asyncio.to_thread 包裹。索引缺失 /
    未匹配（matched=False）时返回空列表，不打断 web 路。
    """
    try:
        from rag_retrieval.pipeline import search as rag_search
    except Exception as e:  # noqa: BLE001
        logger.warning("rag_retrieval 不可用，跳过本地检索：%s", e)
        return []

    try:
        result = rag_search(query, top_k=top_k)
    except Exception as e:  # noqa: BLE001 —— 索引缺失等
        logger.warning("本地检索失败 query=%r: %s", query, e)
        return []

    if not result.matched or not result.hits:
        return []

    hits: list[SearchHit] = []
    for h in result.hits:
        hits.append(
            SearchHit(
                title=f"手册 p{h.page}",
                url=f"local://manual#page{h.page}",
                snippet=h.text,
                source="local",
            )
        )
    return hits
