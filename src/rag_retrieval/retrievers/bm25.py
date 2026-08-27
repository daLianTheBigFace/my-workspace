"""稀疏召回路：BM25Okapi + jieba 分词（页面级）。

照搬 week06 RAG101_03：对【整页文本】建 BM25Okapi，查询分词后打分，
返回 top10 个页面（week06 的 submit_bm25_retrieval_top10.json 口径）。
"""
from __future__ import annotations

import jieba
from rank_bm25 import BM25Okapi

from .base import BaseRetriever, RetrievalHit, register_retriever


@register_retriever
class Bm25Retriever(BaseRetriever):
    name = "bm25"

    def __init__(self, index):
        super().__init__(index)
        # 对整页文本建倒排索引（首次加载耗时几秒，之后常驻内存）
        self._bm25 = BM25Okapi([jieba.lcut(t) for t in index.pages])

    def search(self, query: str, top_k: int = 10) -> list[RetrievalHit]:
        scores = self._bm25.get_scores(jieba.lcut(query))
        top = scores.argsort()[::-1][:top_k]
        return [
            RetrievalHit(
                page=int(idx) + 1,  # pages 列表下标 = 页码 - 1
                text=self.index.page_text(int(idx) + 1),
                score=float(scores[idx]),
                retriever=self.name,
            )
            for idx in top
        ]
