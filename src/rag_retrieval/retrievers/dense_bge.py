"""稠密召回路：bge-small-zh-v1.5 句向量 + 余弦相似度（页面级）。

复用 sentence_bert.get_model()（同一份模型权重，不重复加载）。
算法照搬 week06 RAG101_05：normalize_embeddings=True 点积相似度；
取 top100 chunk 后按「页面去重」取 top10 个不重复页面（周内即
remove_duplicates(pages[:100])[:10] 的口径）。
"""
from __future__ import annotations

import numpy as np

from sentence_bert import get_model

from .base import BaseRetriever, RetrievalHit, register_retriever

# week06 RAG101_05：先看 top100 chunk，再按页面去重取前 10（dedup 后再取）
_DEDUP_CHUNK_SCAN = 100


@register_retriever
class DenseBgeRetriever(BaseRetriever):
    name = "dense_bge"

    def search(self, query: str, top_k: int = 10) -> list[RetrievalHit]:
        if self.index.embeddings is None:
            raise RuntimeError(
                "索引缺少稠密向量（embeddings.npy），请重新运行 build_index"
            )
        model = get_model()
        q_emb = model.encode([query], normalize_embeddings=True)[0]
        scores = q_emb @ self.index.embeddings.T  # [N] 余弦（向量已归一化）
        order = np.argsort(scores)[::-1][:_DEDUP_CHUNK_SCAN]

        seen: set[int] = set()
        hits: list[RetrievalHit] = []
        for idx in order:
            page = self.index.chunks[idx].page
            if page in seen:
                continue
            seen.add(page)
            hits.append(
                RetrievalHit(
                    page=page,
                    text=self.index.page_text(page),
                    score=float(scores[idx]),
                    retriever=self.name,
                )
            )
            if len(hits) == top_k:
                break
        return hits
