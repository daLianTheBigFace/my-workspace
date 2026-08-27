"""RAG 检索编排：多路召回 → RRF 融合 → rerank 重排 →（可选）大模型问答。

对外只暴露 search()，API 层调用它。功能按模块隔开：
- 召回：retrievers/（dense_bge / bm25 / tfidf 可插拔）
- 融合：fusion.py（RRF）
- 重排：rerank.py（bge-reranker-base，懒加载 + 优雅降级）
- 问答：answer.py（OpenAI 兼容接口）
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from . import answer as answer_mod
from . import fusion, rerank as rerank_mod
from .config import RagConfig
from .index import load_index
from .retrievers import available_retrievers, get_retriever

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchHit:
    page: int
    text: str
    rerank_score: float | None  # rerank 不可用时为 None
    paths: tuple[str, ...]  # 命中了哪几路召回


@dataclass(frozen=True)
class SearchResult:
    query: str
    hits: list[SearchHit]
    retrieval_paths: list[str]  # 实际参与召回的路径
    reranked: bool
    matched: bool = True  # top1 重排分 >= 阈值
    message: str | None = None  # 没匹配到时的提示语
    answer: str | None = None  # 大模型返回
    prompt: str | None = None  # 实际发给大模型的完整 prompt


def _reference_pages(hits: list[SearchHit], limit: int) -> list[tuple[int, str]]:
    """取重排后 top N 的整页作为上下文（照搬 RAG101_08：用整页文本回答）。"""
    seen_pages: set[int] = set()
    reference_pages: list[tuple[int, str]] = []
    for h in hits:
        if h.page not in seen_pages:
            seen_pages.add(h.page)
            reference_pages.append((h.page, load_index().page_text(h.page)))
    return reference_pages[:limit]


def _recall(query: str, top_k: int) -> tuple[list[list], list[str]]:
    """多路召回：每路各出 top_k，返回（hit 列表的列表，实际参与的路名）。"""
    used: list[str] = []
    hits_by_path: list[list] = []
    for name in available_retrievers():
        try:
            retriever = get_retriever(name)
            hits_by_path.append(retriever.search(query, top_k=top_k))
            used.append(name)
        except Exception as e:
            logger.warning("召回器 '%s' 不可用，跳过：%s", name, e)
    if not hits_by_path:
        raise RuntimeError("所有召回路都不可用，请检查索引是否已构建")
    return hits_by_path, used


def search(query: str, top_k: int = 5, with_answer: bool = False) -> SearchResult:
    """完整检索：多路召回 → RRF 融合 → rerank → 可选问答。"""
    cfg = RagConfig()
    load_index()  # 索引缺失时在这里抛 FileNotFoundError

    hits_by_path, used_paths = _recall(query, cfg.retrieve_top_k)
    fused = fusion.reciprocal_rank_fusion(hits_by_path, k=cfg.rrf_k)

    # rerank 重排：融合后 rerank_top_n 页的【整页文本】送入交叉编码器，
    # 取与 query 最相关的页（照搬 week06 RAG101_06：top3 页整页重排取 argmax）
    top_fused = fused[: cfg.rerank_top_n]
    if rerank_mod.reranker_ready():
        scores = rerank_mod.rerank_scores(query, [h.text for h in top_fused])
        ordered = sorted(zip(top_fused, scores), key=lambda ts: ts[1], reverse=True)
        reranked = True
    else:
        # 重排模型未下载/加载失败 → 退化为按 RRF 融合分排序
        logger.warning("reranker 不可用，按 RRF 融合分返回（不重排）")
        ordered = [(h, None) for h in top_fused]
        reranked = False

    # 没匹配到：top1 重排分低于阈值 → 不返回检索结果，直接提示。
    # 只在重排可用时判断（reranked=False 降级模式拿不到可信分数，全量返回）。
    if (
        reranked
        and ordered
        and ordered[0][1] is not None
        and ordered[0][1] < cfg.min_rerank_score
    ):
        return SearchResult(
            query=query,
            hits=[],
            retrieval_paths=used_paths,
            reranked=True,
            matched=False,
            message="没有匹配到相关内容（检索置信度过低）",
        )

    hits = [
        SearchHit(
            page=h.page,
            text=h.text,
            rerank_score=score,
            paths=h.paths,
        )
        for h, score in ordered[:top_k]
    ]

    answer = None
    prompt = None
    if with_answer and cfg.answer_enabled and hits:
        reference_pages = _reference_pages(hits, cfg.answer_top_n)
        prompt = answer_mod.build_prompt(query, reference_pages)
        answer = answer_mod.call_llm(prompt)

    return SearchResult(
        query=query,
        hits=hits,
        retrieval_paths=used_paths,
        reranked=reranked,
        answer=answer,
        prompt=prompt,
    )
