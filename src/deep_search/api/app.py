"""deep_search API 路由（挂 /deep_search 前缀）。

主服务 app.py 里 include_router 一行接入即可。
接口：
- POST /deep_search/search   流式 Deep Research（SSE）
- GET  /deep_search/health   key / 就绪状态
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from .. import graph as graph_mod
from ..config import DeepSearchConfig
from ..output import write_outputs
from ..schemas import AnswerEvent, DoneEvent, ErrorEvent, Stats, sse_frame

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/deep_search", tags=["deep_search"])

cfg = DeepSearchConfig()


# ---- Schemas ----


class SearchRequest(BaseModel):
    question: str = Field(..., min_length=1, description="要深度研究的问题")
    max_rounds: int | None = Field(
        default=None, ge=1, le=10, description="迭代轮数上限（缺省用配置）"
    )
    save_files: bool = Field(default=True, description="是否落盘 report.md + result.json")

    @field_validator("question")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question 不能为空白")
        return v.strip()


class HealthOut(BaseModel):
    llm_ready: bool
    web_ready: bool
    llm_model: str
    max_rounds: int
    output_dir: str


# ---- helpers ----


def _require_ready() -> None:
    """LLM 是 hard 依赖（planner/judge/synthesize 都要）；web 是 soft（缺了只跑本地）。"""
    if not cfg.llm_ready:
        raise HTTPException(
            status_code=503,
            detail="未配置 LLM key：请在仓库根目录 .env 设 DEEPSEEK_API_KEY（或 RAG_LLM_API_KEY），"
            "deep_search 的 planner/judge/synthesize 都需要它",
        )


# ---- SSE 流 ----


async def _stream(question: str, max_rounds: int, save_files: bool) -> AsyncIterator[str]:
    started = time.monotonic()
    trace: list[dict] = []
    answer_parts: list[str] = []
    final_state: dict | None = None

    initial = {"question": question, "round": 1, "max_rounds": max_rounds}
    try:
        async for ev in graph_mod.graph.astream_events(initial, version="v2"):
            kind = ev["event"]
            if kind == "on_chain_end":
                name = ev.get("name")
                output = ev["data"].get("output")
                if name in ("plan", "search", "read", "reflect") and isinstance(output, dict):
                    for sse in output.get("events", []):
                        trace.append({"event": name, **sse.model_dump()})
                        yield sse_frame(name, sse)
                elif name == "LangGraph" and isinstance(output, dict):
                    final_state = output
            elif kind == "on_chat_model_stream":
                if ev.get("metadata", {}).get("langgraph_node") == "synthesize":
                    c = ev["data"]["chunk"].content
                    if isinstance(c, str) and c:
                        answer_parts.append(c)
                        yield sse_frame("answer", AnswerEvent(delta=c))
    except Exception as e:  # noqa: BLE001
        logger.exception("deep_search 执行失败")
        yield sse_frame("error", ErrorEvent(message=str(e)))
        return

    if not final_state:
        yield sse_frame("error", ErrorEvent(message="图未产出结果"))
        return

    answer = final_state.get("answer") or "".join(answer_parts)
    sources = final_state.get("sources", [])
    stats = Stats(
        rounds=final_state.get("round", 1),
        searches=len(final_state.get("seen_queries", set())),
        sources=len(sources),
        elapsed_ms=int((time.monotonic() - started) * 1000),
    )

    report_path = result_path = None
    if save_files:
        try:
            report_path, result_path = write_outputs(
                question=question,
                answer=answer,
                sources=sources,
                stats=stats,
                sub_questions=final_state.get("sub_questions", []),
                stop_reason=final_state.get("stop_reason", ""),
                trace=trace,
            )
        except Exception as e:  # noqa: BLE001 —— 落盘失败不影响已流出的答案
            logger.exception("落盘失败")

    yield sse_frame(
        "done",
        DoneEvent(
            answer=answer,
            sources=sources,
            report_path=report_path,
            result_path=result_path,
            stats=stats,
        ),
    )


# ---- Endpoints ----


@router.post("/search")
async def search(req: SearchRequest) -> StreamingResponse:
    _require_ready()
    return StreamingResponse(
        _stream(req.question, req.max_rounds or cfg.max_rounds, req.save_files),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/health", response_model=HealthOut)
def health() -> HealthOut:
    return HealthOut(
        llm_ready=cfg.llm_ready,
        web_ready=cfg.web_ready,
        llm_model=cfg.llm_model,
        max_rounds=cfg.max_rounds,
        output_dir=cfg.output_dir,
    )


def warm() -> None:
    """启动预热：检查 key / 建输出目录，缺了只告警，不拖垮启动。"""
    if not cfg.llm_ready:
        logger.warning("deep_search：未配置 LLM key，/deep_search/search 暂不可用")
    if not cfg.web_ready:
        logger.warning("deep_search：未配置 TAVILY_API_KEY，联网检索将跳过（仅本地 tool）")
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
