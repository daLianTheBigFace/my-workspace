"""web_search：Tavily 联网检索。

deep_search 的联网主力：把 planner/补搜的 query 交给 Tavily，返回 title/url/snippet
列表，供 read 节点抓取阅读。
"""
from __future__ import annotations

import logging

from ..config import DeepSearchConfig
from ..schemas import SearchHit

logger = logging.getLogger(__name__)

_client = None


def _get_client():
    global _client
    if _client is None:
        from tavily import AsyncTavilyClient

        _client = AsyncTavilyClient(api_key=DeepSearchConfig().tavily_api_key)
    return _client


async def search(query: str, *, max_results: int = 5) -> list[SearchHit]:
    """Tavily 联网检索，返回 SearchHit(source="web") 列表；无 key/失败返回 []。"""
    cfg = DeepSearchConfig()
    if not cfg.web_ready:
        logger.warning("未配置 TAVILY_API_KEY，联网检索跳过（query=%r）", query)
        return []
    try:
        resp = await _get_client().search(query, max_results=max_results)
    except Exception as e:  # noqa: BLE001 —— 联网失败不拖垮整轮
        logger.warning("Tavily 检索失败 query=%r: %s", query, e)
        return []

    hits: list[SearchHit] = []
    for r in resp.get("results") or []:
        hits.append(
            SearchHit(
                title=r.get("title", ""),
                url=r.get("url", ""),
                snippet=r.get("content") or r.get("snippet", ""),
                source="web",
            )
        )
    return hits
