"""ES 检索 API 路由（挂主服务 /es 前缀）。

主服务 app.py 里 include_router 一行即可接入：
    from es_search.api import router as es_router, warm as warm_es
    app.include_router(es_router)

- POST /es/full_text   全文检索（IK 分词）
- POST /es/filter      全文 + 页码范围条件过滤
- POST /es/vector      向量检索（bge 编码 + knn）
- GET  /es/index_info  ES 连接与索引状态
"""
from __future__ import annotations

import logging

from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

MatchType = Literal["match", "phrase", "fuzzy"]

from .. import search as es_search
from ..config import EsConfig
from ..es_client import count_docs, get_client, ping

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/es", tags=["ES"])


# ---- Pydantic Schemas ----
class EsQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="检索内容")
    size: int = Field(default=10, ge=1, le=50, description="返回条数")
    match_type: MatchType = Field(
        default="match",
        description="匹配模式：match=分词模糊（默认）/ phrase=精确短语 / fuzzy=容错模糊",
    )

    @field_validator("query")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query 不能为空白")
        return v.strip()


class EsFilterRequest(EsQueryRequest):
    page_from: int | None = Field(default=None, ge=1, description="页码下限（含）")
    page_to: int | None = Field(default=None, ge=1, description="页码上限（含）")


class EsHitOut(BaseModel):
    chunk_id: int
    page: int
    text: str
    score: float


class EsSearchResponse(BaseModel):
    query: str
    hits: list[EsHitOut]
    total: int  # 满足条件的总条数（size 只是返回数）


class IndexInfo(BaseModel):
    es_connected: bool
    index_exists: bool
    doc_count: int | None


def _require_es() -> None:
    """ES 不在就 503，带提示。"""
    if not ping():
        raise HTTPException(
            status_code=503,
            detail="ES 未运行（http://localhost:9200），请先启动 elasticsearch.bat",
        )


def _resp(query: str, hits, total: int) -> EsSearchResponse:
    return EsSearchResponse(
        query=query,
        hits=[
            EsHitOut(chunk_id=h.chunk_id, page=h.page, text=h.text, score=h.score)
            for h in hits
        ],
        total=total,
    )


# ---- Endpoints ----
@router.post("/full_text", response_model=EsSearchResponse, tags=["ES"])
def full_text(req: EsQueryRequest) -> EsSearchResponse:
    _require_es()
    hits, total = es_search.full_text(
        req.query, size=req.size, match_type=req.match_type
    )
    return _resp(req.query, hits, total)


@router.post("/filter", response_model=EsSearchResponse, tags=["ES"])
def filtered(req: EsFilterRequest) -> EsSearchResponse:
    _require_es()
    hits, total = es_search.filtered(
        req.query,
        page_from=req.page_from,
        page_to=req.page_to,
        size=req.size,
        match_type=req.match_type,
    )
    return _resp(req.query, hits, total)


@router.post("/vector", response_model=EsSearchResponse, tags=["ES"])
def vector(req: EsQueryRequest) -> EsSearchResponse:
    _require_es()
    try:
        hits, total = es_search.vector(req.query, size=req.size)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"向量检索失败（bge 模型或 ES）：{e}") from e
    return _resp(req.query, hits, total)


@router.get("/index_info", response_model=IndexInfo, tags=["ES"])
def index_info() -> IndexInfo:
    connected = ping()
    exists = False
    if connected:
        exists = bool(get_client().indices.exists(index=EsConfig().index_name))
    return IndexInfo(
        es_connected=connected,
        index_exists=exists,
        doc_count=count_docs() if connected else None,
    )


def warm() -> None:
    """预热：ping 一下 ES（挂了只告警，不拖垮主服务启动）。"""
    if not ping():
        logger.warning("ES 不可用：%s（可稍后再启动）", EsConfig().es_url)
