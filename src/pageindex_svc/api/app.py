"""pageindex 文档检索 API 路由（挂 /pageindex 前缀）。

主服务 app.py 里 include_router 一行即可接入；也可独立起（见 pageindex_svc.serve）：
    from pageindex_svc.api import router as pageindex_router, warm as warm_pageindex
    app.include_router(pageindex_router)

接口：
- POST   /pageindex/submit                提交 PDF 建索引（flash 默认 / standard 全 LLM 建树）
- GET    /pageindex/documents             已建索引列表
- GET    /pageindex/documents/{doc_id}    单个文档元信息
- GET    /pageindex/documents/{doc_id}/tree   文档树（可带节点摘要/整页文本）
- DELETE /pageindex/documents/{doc_id}    删除索引
- POST   /pageindex/query                 推理式检索 + 问答（DeepSeek）
- GET    /pageindex/health                key / 存储目录状态

说明：建索引与检索都要 LLM（本地引擎走 DeepSeek）。只列/查已建索引不需要 key。
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from .. import service
from ..config import PageIndexConfig

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/pageindex", tags=["pageindex"])

cfg = PageIndexConfig()


# ---- Schemas ----


class SubmitRequest(BaseModel):
    pdf_path: str | None = Field(
        default=None,
        description="要建索引的 PDF 绝对路径；缺省用汽车知识手册（assets/Week06）",
    )
    mode: Literal["flash", "standard"] = Field(
        default="flash",
        description="flash=本地版面抽取 + LLM 摘要/优化（默认，快）；standard=全 LLM 建树（慢）",
    )
    metadata: dict[str, Any] | None = Field(default=None, description="附加 JSON 标签")


class DocOut(BaseModel):
    doc_id: str
    name: str
    status: str
    page_num: int
    mode: str | None = None
    description: str | None = None


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="问题 / 检索内容")
    doc_id: str = Field(..., min_length=1, description="已建索引的文档 id")
    with_process: bool = Field(
        default=False,
        description="True 时返回检索过程（思考 / 工具调用 / 命中片段）",
    )

    @field_validator("query")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query 不能为空白")
        return v.strip()


class QueryOut(BaseModel):
    doc_id: str
    query: str
    answer: str
    steps: list[dict[str, Any]] = Field(default_factory=list, description="检索过程（with_process=True 才有）")


class HealthOut(BaseModel):
    api_key_set: bool
    model: str
    storage_path: str
    default_pdf: str
    default_pdf_exists: bool


# ---- helpers ----


def _require_llm() -> None:
    """建索引 / 检索都离不开 DeepSeek，没 key 直接 503 带提示。"""
    if not cfg.api_key_set:
        raise HTTPException(
            status_code=503,
            detail="未配置 LLM key：请在仓库根目录 .env 设 DEEPSEEK_API_KEY（或 RAG_LLM_API_KEY）"
            "，PageIndex 建索引与检索都需要它",
        )


# ---- Endpoints ----


@router.post("/submit", response_model=DocOut)
def submit(req: SubmitRequest) -> DocOut:
    _require_llm()
    pdf = req.pdf_path or cfg.default_pdf
    if not pdf or not os.path.isfile(pdf):
        raise HTTPException(status_code=400, detail=f"PDF 不存在：{pdf}")
    try:
        res = service.submit_pdf(pdf, mode=req.mode, metadata=req.metadata)
    except Exception as e:  # noqa: BLE001 —— 把 DeepSeek/索引器的错原样告诉用户
        logger.exception("pageindex 建索引失败")
        raise HTTPException(status_code=502, detail=f"建索引失败：{e}") from e
    meta = service.get_doc(res["doc_id"])
    return DocOut(
        doc_id=meta.get("id") or res["doc_id"],
        name=meta.get("name", ""),
        status=meta.get("status", "completed"),
        page_num=int(meta.get("pageNum", 0) or 0),
        mode=req.mode,
        description=meta.get("description"),
    )


@router.get("/documents")
def documents(limit: int = 50, offset: int = 0) -> dict[str, Any]:
    try:
        return service.list_docs(limit=limit, offset=offset)
    except Exception as e:  # noqa: BLE001
        logger.exception("pageindex 列文档失败")
        raise HTTPException(status_code=502, detail=f"列文档失败：{e}") from e


@router.get("/documents/{doc_id}")
def document(doc_id: str) -> dict[str, Any]:
    try:
        return service.get_doc(doc_id)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=404, detail=f"未找到文档：{e}") from e


@router.get("/documents/{doc_id}/tree")
def tree(
    doc_id: str,
    node_summary: bool = False,
    include_text: bool = False,
) -> dict[str, Any]:
    try:
        return service.get_tree(doc_id, node_summary=node_summary, include_text=include_text)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"取树失败：{e}") from e


@router.delete("/documents/{doc_id}")
def delete(doc_id: str) -> dict[str, Any]:
    try:
        return service.delete_doc(doc_id)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=404, detail=f"删除失败：{e}") from e


@router.post("/query", response_model=QueryOut)
def query(req: QueryRequest) -> QueryOut:
    _require_llm()
    try:
        answer, steps = service.query(req.doc_id, req.query, with_process=req.with_process)
    except Exception as e:  # noqa: BLE001
        logger.exception("pageindex 检索失败")
        raise HTTPException(status_code=502, detail=f"检索失败：{e}") from e
    return QueryOut(doc_id=req.doc_id, query=req.query, answer=answer, steps=steps)


@router.get("/health", response_model=HealthOut)
def health() -> HealthOut:
    return HealthOut(
        api_key_set=cfg.api_key_set,
        model=cfg.model,
        storage_path=cfg.storage_path,
        default_pdf=cfg.default_pdf,
        default_pdf_exists=os.path.isfile(cfg.default_pdf),
    )


def warm() -> None:
    """启动预热：建目录 + 检查 key/默认 PDF，缺了只告警，不拖垮启动。"""
    Path(cfg.storage_path).mkdir(parents=True, exist_ok=True)
    if not cfg.api_key_set:
        logger.warning(
            "pageindex：未配置 DEEPSEEK_API_KEY（或 RAG_LLM_API_KEY），"
            "/pageindex 的建索引与检索暂不可用"
        )
    if not os.path.isfile(cfg.default_pdf):
        logger.warning("pageindex：默认 PDF 不存在：%s", cfg.default_pdf)
