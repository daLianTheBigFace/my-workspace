"""RRF 多路召回融合（照搬 week06 RAG101_07，k=60 的逻辑一字不改）。

把各路【页面】结果按 rank 累加 1/(rank + k) 得分，按总分降序。
rank 从 0 开始 → 排得越靠前，单路得分越高。
week06 融合的是页面（submit_bge_sgement_retrieval_top10.json 等），这里同样页面级。
"""
from __future__ import annotations

from dataclasses import dataclass

from .retrievers.base import RetrievalHit


@dataclass(frozen=True)
class FusedHit:
    page: int
    text: str
    rrf_score: float
    paths: tuple[str, ...]  # 命中了哪几路召回


def reciprocal_rank_fusion(
    hits_by_path: list[list[RetrievalHit]], k: int = 60
) -> list[FusedHit]:
    """多路召回结果融合（页面级）。

    hits_by_path：每路召回器的 top_k 页面。同一个页面在不同路里 rank 不同，
    按 1/(rank + k) 累加得到融合分，跨路都命中 → 分更高。
    """
    fusion_score: dict[int, float] = {}
    texts: dict[int, str] = {}
    paths: dict[int, set[str]] = {}

    for path_hits in hits_by_path:
        seen_in_path: set[int] = set()
        for rank, hit in enumerate(path_hits):
            page = hit.page
            if page in seen_in_path:
                continue  # 同一路内页面去重兜底（召回器本已去重）
            seen_in_path.add(page)
            fusion_score[page] = fusion_score.get(page, 0.0) + 1.0 / (rank + k)
            texts[page] = hit.text
            paths.setdefault(page, set()).add(hit.retriever)

    ordered = sorted(fusion_score.items(), key=lambda kv: kv[1], reverse=True)
    return [
        FusedHit(
            page=page,
            text=texts[page],
            rrf_score=score,
            paths=tuple(sorted(paths[page])),
        )
        for page, score in ordered
    ]
