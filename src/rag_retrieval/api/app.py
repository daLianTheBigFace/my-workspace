"""RAG 检索 + 重排 + 问答 API 路由（挂到主服务 /rag 前缀下）。

主服务 app.py 里 include_router 一行即可接入：
    from rag_retrieval.api import router, warm as warm_rag
    app.include_router(router)

- POST /rag/search        多路召回 + 融合 + 重排，可选大模型问答
- GET  /rag/index_info    索引元信息与召回/重排可用状态
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from .. import pipeline
from ..index import index_info as _index_info
from ..rerank import reranker_ready
from ..retrievers import available_retrievers

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rag", tags=["RAG"])


# ---- Pydantic Schemas ----
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="检索问题")
    top_k: int = Field(default=3, ge=1, le=20, description="返回条数")
    with_answer: bool = Field(default=False, description="是否调大模型生成回答")

    @field_validator("query")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query 不能为空白")
        return v.strip()


class SearchHitOut(BaseModel):
    page: int
    text: str
    rerank_score: float | None
    paths: list[str]


class SearchResponse(BaseModel):
    query: str  # 用户提问
    hits: list[SearchHitOut]  # RAG 检索出的 topK（没匹配到时为空）
    retrieval_paths: list[str]
    reranked: bool
    matched: bool = True  # top1 重排分是否 >= 阈值
    message: str | None = None  # 没匹配到时的提示语
    prompt: str | None = None  # 发给模型的完整 prompt
    answer: str | None = None  # 模型的返回


class IndexInfo(BaseModel):
    n_chunks: int
    n_pages: int
    has_dense: bool
    retrievers: list[str]
    reranker_ready: bool


# ---- Endpoints ----
@router.post("/search", response_model=SearchResponse, tags=["RAG"])
def search_endpoint(req: SearchRequest) -> SearchResponse:
    try:
        result = pipeline.search(req.query, top_k=req.top_k, with_answer=req.with_answer)
    except FileNotFoundError as e:
        # 索引未构建 → 503 + 提示
        raise HTTPException(status_code=503, detail=f"索引未构建：{e}") from e
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    return SearchResponse(
        query=result.query,
        hits=[
            SearchHitOut(
                page=h.page,
                text=h.text,
                rerank_score=h.rerank_score,
                paths=list(h.paths),
            )
            for h in result.hits
        ],
        retrieval_paths=result.retrieval_paths,
        reranked=result.reranked,
        matched=result.matched,
        message=result.message,
        prompt=result.prompt,
        answer=result.answer,
    )


@router.get("/index_info", response_model=IndexInfo, tags=["RAG"])
def index_info() -> IndexInfo:
    try:
        info = _index_info()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"索引未构建：{e}") from e
    return IndexInfo(
        n_chunks=info["n_chunks"],
        n_pages=info["n_pages"],
        has_dense=info["has_dense"],
        retrievers=available_retrievers(),
        reranker_ready=reranker_ready(),
    )


def warm() -> None:
    """预热：加载索引 + 全部召回器（重排模型懒加载，首次 /search 才加载）。

    主服务 lifespan 里 try/except 包一层调用，避免索引缺失拖垮启动。
    """
    from ..index import load_index
    from ..retrievers import get_retriever

    load_index()
    for name in available_retrievers():
        try:
            get_retriever(name)
        except Exception:
            logger.exception("预热召回器 '%s' 失败", name)
