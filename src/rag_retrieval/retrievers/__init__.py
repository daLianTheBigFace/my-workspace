"""召回器注册入口：import 本模块即触发全部注册。"""
from . import bm25, dense_bge, tfidf  # noqa: F401
from .base import (
    BaseRetriever,
    RetrievalHit,
    RetrieverLoadError,
    RetrieverNotFoundError,
    available_retrievers,
    get_retriever,
)

__all__ = [
    "BaseRetriever",
    "RetrievalHit",
    "RetrieverLoadError",
    "RetrieverNotFoundError",
    "available_retrievers",
    "get_retriever",
]
