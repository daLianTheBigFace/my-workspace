"""ES 客户端单例 + 索引管理。

用 elasticsearch-py 9.x 的现代参数写法（indices.create(index=, settings=, mappings=)、
search(query=)），服务端是本地 ES 8.17（无认证 HTTP，week06 同款连接方式）。
"""
from __future__ import annotations

import logging

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from .config import EsConfig

logger = logging.getLogger(__name__)

_client: Elasticsearch | None = None

# 文本字段用 IK 中文分词（ik_max_word 索引最细粒度 / ik_smart 查询智能粒度），
# 向量字段用 dense_vector(512, cosine) 支持 HNSW knn 检索。
MAPPINGS = {
    "properties": {
        "chunk_id": {"type": "integer"},
        "page": {"type": "integer"},
        "text": {
            "type": "text",
            "analyzer": "ik_max_word",
            "search_analyzer": "ik_smart",
        },
        "text_vector": {
            "type": "dense_vector",
            "dims": 512,
            "index": True,
            "similarity": "cosine",
        },
    }
}

SETTINGS = {"number_of_shards": 1, "number_of_replicas": 0}


def get_client() -> Elasticsearch:
    """进程内单例：连接本地 ES。"""
    global _client
    if _client is None:
        _client = Elasticsearch(EsConfig().es_url, request_timeout=30)
    return _client


def ping() -> bool:
    """ES 是否可连（挂了不致命，返回 False）。"""
    try:
        return bool(get_client().ping())
    except Exception:
        logger.warning("ES 连接失败：%s", EsConfig().es_url, exc_info=True)
        return False


def ensure_index() -> None:
    """索引不存在则创建（含 mapping）。已存在则跳过。"""
    cfg = EsConfig()
    es = get_client()
    if es.indices.exists(index=cfg.index_name):
        return
    es.indices.create(
        index=cfg.index_name,
        settings=SETTINGS,
        mappings=MAPPINGS,
    )
    logger.info("已创建 ES 索引 %s", cfg.index_name)


def count_docs() -> int | None:
    """索引文档数（不存在返回 None）。"""
    cfg = EsConfig()
    es = get_client()
    if not es.indices.exists(index=cfg.index_name):
        return None
    return int(es.count(index=cfg.index_name)["count"])
