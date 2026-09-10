"""reader：抓网页 → trafilatura 抽正文。

read 节点用它把 search 命中的 URL 抓下来、抽成纯文本，再交给 judge.filter 逐篇质量
过滤。
"""
from __future__ import annotations

import asyncio
import logging

import httpx

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
}


async def fetch(url: str, *, timeout: int = 10) -> str:
    """抓网页 → trafilatura 抽正文，返回纯文本；失败抛异常（节点捕获标 status="fail"）。"""
    import trafilatura

    async with httpx.AsyncClient(
        headers=_HEADERS, follow_redirects=True, timeout=timeout
    ) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        html = resp.text

    # trafilatura 是 CPU 绑定，丢线程池避免阻塞事件循环
    text = await asyncio.to_thread(
        trafilatura.extract, html, url=url, include_comments=False, include_tables=False
    )
    if not text or not text.strip():
        raise RuntimeError("正文抽取为空")
    return text.strip()
