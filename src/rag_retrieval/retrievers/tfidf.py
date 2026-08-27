"""稀疏召回路：TF-IDF + jieba 分词 + 余弦相似度（页面级）。

照搬 week06 RAG101_02：jieba 分词后 TfidfVectorizer 拟合语料，
归一化后用点积算相似度。与稠密/BM25 一样按整页召回，返回 top10 页。

与 week06 的唯一差别：week06 用「历史问题 + 语料」一起拟合 idf，
服务场景下没有历史问题集，这里只用知识库语料拟合 idf。
"""
from __future__ import annotations

import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .base import BaseRetriever, RetrievalHit, register_retriever


@register_retriever
class TfidfRetriever(BaseRetriever):
    name = "tfidf"

    def __init__(self, index):
        super().__init__(index)
        corpus = [" ".join(jieba.lcut(t)) for t in index.pages]
        self._vectorizer = TfidfVectorizer()
        self._doc_feat = self._vectorizer.fit_transform(corpus)
        self._doc_feat = normalize(self._doc_feat)

    def search(self, query: str, top_k: int = 10) -> list[RetrievalHit]:
        q_words = " ".join(jieba.lcut(query))
        q_feat = normalize(self._vectorizer.transform([q_words]))
        scores = (q_feat @ self._doc_feat.T).toarray()[0]
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
