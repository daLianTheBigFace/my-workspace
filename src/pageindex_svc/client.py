"""PageIndex 本地引擎客户端单例。

lazy 构建：真到第一次用到才 `from pageindex import PageIndexLocalClient`，
避免 litellm 的多秒导入拖慢服务启动和无关请求。构造本身不调 LLM，
只有 submit/query 时才真正连 DeepSeek。
"""
from __future__ import annotations

import logging
from pathlib import Path

from .config import PageIndexConfig

logger = logging.getLogger(__name__)

_client = None


def get_client():
    """进程内单例 PageIndexLocalClient（本地引擎 + DeepSeek，索引落 E 盘）。"""
    global _client
    if _client is None:
        cfg = PageIndexConfig()
        Path(cfg.storage_path).mkdir(parents=True, exist_ok=True)
        # key 有没有不拦构造；缺 key 时 submit/query 再报明确错误（见 api.app._require_llm）。
        from pageindex import PageIndexLocalClient

        _client = PageIndexLocalClient(
            index_model=cfg.model,
            summary_model=cfg.summary_model,
            chat_model=cfg.model,
            storage_path=cfg.storage_path,
        )
        logger.info(
            "PageIndex 本地引擎就绪：model=%s storage=%s",
            cfg.model,
            cfg.storage_path,
        )
    return _client
