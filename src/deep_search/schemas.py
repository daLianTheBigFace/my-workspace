"""deep_search 的 Pydantic 模型：结构化输出 + 数据模型 + SSE 事件。

三类：
1. 结构化输出（planner / judge 用 with_structured_output 逼 LLM 返回固定形状）：
   PlanOutput / FilterVerdict / CheckVerdict
2. 数据模型（state 与结果复用）：SearchHit / Note / Source / Stats
3. SSE 事件（POST /deep_search/search 流式 data 载荷）：PlanEvent / SearchEvent /
   ReadEvent / ReflectEvent / AnswerEvent / DoneEvent / ErrorEvent，配 sse_frame() 编码。
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# SSE 事件名（也用作 sse_frame 的 event 字段）
EventName = Literal["plan", "search", "read", "reflect", "answer", "done", "error"]


# --------------------------------------------------------------------------
# 结构化输出（LLM 返回形状）
# --------------------------------------------------------------------------


class PlanOutput(BaseModel):
    """planner 结构化输出：去废话后拆出的搜索 query（关键词式短句，可搜索）。

    LLM 只出 query 文本、不生成 id —— id 由代码按序号包装成 SubQuestion（LLM 编 id
    不可靠）。故字段名用 questions（list[str]），区别于带 id 的 SubQuestion。
    """

    questions: list[str] = Field(
        ..., description="拆解出的搜索 query 文本列表（去废话、关键词化）"
    )


class FilterVerdict(BaseModel):
    """judge.filter 结构化输出：单篇抽取正文值不值得信、有没有营养。"""

    useful: bool = Field(..., description="是否采纳进 notes（不相关/低质 → False）")
    summary: str = Field("", description="若采纳，浓缩成可引用的精华片段")
    reason: str = Field("", description="一句话理由（不采纳时说明为什么）")


class CheckVerdict(BaseModel):
    """judge.check 结构化输出：攒的证据够不够回答原问题。"""

    verdict: Literal["continue", "enough"] = Field(..., description="够不够")
    gap_queries: list[str] = Field(
        default_factory=list, description="不够时的补搜 query（可搜索串，避免兜圈）"
    )
    rationale: str = Field("", description="判断理由（写进 reflect 事件）")


# --------------------------------------------------------------------------
# 数据模型（state 与结果复用）
# --------------------------------------------------------------------------


class SubQuestion(BaseModel):
    """planner 拆出的一个子问题（带 id 便于前端对照追踪检索进度）。"""

    id: str
    question: str


class SearchHit(BaseModel):
    """一次检索命中的单条结果（web 与 local 统一成同构，方便合并去重）。"""

    title: str
    url: str
    snippet: str = ""
    source: Literal["web", "local"] = "web"


class Note(BaseModel):
    """judge.filter 采纳后进 notes 的精华片段（供 synthesize 引用 + report 检索过程）。"""

    text: str
    title: str
    url: str
    source: Literal["web", "local"]
    round: int  # 哪个搜索轮次采到的


class Source(BaseModel):
    """最终引用的来源（done 事件 + report.md 的 [n] 列表，已去重）。"""

    title: str
    url: str


class Stats(BaseModel):
    """done 概览的统计。"""

    rounds: int  # 实际跑的搜索轮数
    searches: int  # 实际执行的检索 query 数（= len(seen_queries)）
    sources: int  # 引用来源数
    elapsed_ms: int  # 全程耗时


# --------------------------------------------------------------------------
# SSE 事件载荷（POST /deep_search/search 的 data）
# --------------------------------------------------------------------------


class PlanEvent(BaseModel):
    round: int
    sub_questions: list[SubQuestion]


class SearchEvent(BaseModel):
    round: int
    query: str
    source: Literal["web", "local"]
    results: list[SearchHit]


class ReadEvent(BaseModel):
    round: int
    url: str
    status: Literal["ok", "fail"]
    chars: int  # 抽取正文字符数（fail 时为 0）


class ReflectEvent(BaseModel):
    round: int
    verdict: Literal["continue", "enough"]
    gap_queries: list[str]
    rationale: str


class AnswerEvent(BaseModel):
    delta: str  # 答案流式增量


class DoneEvent(BaseModel):
    answer: str
    sources: list[Source]
    report_path: str | None = None
    result_path: str | None = None
    stats: Stats


class ErrorEvent(BaseModel):
    message: str


def sse_frame(event: str, data: BaseModel) -> str:
    """把一次事件编码成 SSE 帧：`event: <name>\\ndata: <json>\\n\\n`。

    api 层用它把各事件模型序列化后逐帧 yield 给客户端。
    """
    return f"event: {event}\ndata: {data.model_dump_json()}\n\n"
