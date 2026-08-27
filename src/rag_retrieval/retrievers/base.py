"""召回器抽象层：统一接口 + 注册表 + 单例惰性加载。

思路照搬 intent_recognition/predictors/base.py：所有召回方案
（稠密 bge / 稀疏 bm25 / 稀疏 tfidf）实现 BaseRetriever，
通过 @register_retriever 注册进 RETRIEVERS，调用方按名取单例。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

from ..index import RagIndex, load_index


@dataclass(frozen=True)
class RetrievalHit:
    """单条召回结果：页面级（对齐 week06 RAG101_05/03 的 top10 页）。

    - page：1-based 页码
    - text：整页文本（重排、展示、问答都用它）
    - score：本路原始相关分（排序已按它降序）
    - retriever：来自哪一路
    """

    page: int
    text: str
    score: float
    retriever: str


class BaseRetriever(ABC):
    """所有召回方案的统一接口。子类实现 search(query, top_k)。"""

    name: ClassVar[str]  # 注册名，如 "dense_bge" / "bm25" / "tfidf"

    def __init__(self, index: RagIndex):
        self.index = index

    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> list[RetrievalHit]:
        """返回按相关度降序的 top_k 个不重复页面。"""


# ---- 注册表与单例缓存 ----
RETRIEVERS: dict[str, type[BaseRetriever]] = {}  # name -> 类
_instances: dict[str, BaseRetriever] = {}  # name -> 单例
_load_errors: dict[str, Exception] = {}  # name -> 最近一次加载异常


class RetrieverNotFoundError(KeyError):
    """未知召回方案。"""


class RetrieverLoadError(RuntimeError):
    """方案存在但加载失败（索引缺失等）。"""


def register_retriever(cls: type[BaseRetriever]) -> type[BaseRetriever]:
    """装饰器：把类按 cls.name 注册进 RETRIEVERS。"""
    RETRIEVERS[cls.name] = cls
    return cls


def get_retriever(name: str) -> BaseRetriever:
    """按名取单例：每个召回器只构造一次（惰性加载）。"""
    if name not in RETRIEVERS:
        raise RetrieverNotFoundError(
            f"未知召回方案 '{name}'，可用：{sorted(RETRIEVERS)}"
        )
    if name in _instances:
        return _instances[name]
    index = load_index()  # 首次取召回器时加载索引
    try:
        inst = RETRIEVERS[name](index)
    except Exception as e:
        _load_errors[name] = e
        raise RetrieverLoadError(f"召回方案 '{name}' 加载失败：{e}") from e
    _instances[name] = inst
    return inst


def available_retrievers() -> list[str]:
    """当前已注册的召回方案名。"""
    return sorted(RETRIEVERS)
