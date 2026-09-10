"""AgentState：LangGraph 各节点共享的状态（TypedDict）。

字段分两类：
- 跨轮累加（Annotated + reducer）：notes / seen_queries / seen_urls / sources
- 每轮替换（last-write-wins）：round / gap_queries / verdict / stop_reason
一次性字段：question / sub_questions / answer
"""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from .schemas import Note, SearchHit, Source, SubQuestion


class AgentState(TypedDict, total=False):
    # —— 一次性 ——
    question: str  # 原始用户问题（start 设）
    sub_questions: list[SubQuestion]  # planner 拆出的子问题（带 id，只拆一次）
    max_rounds: int  # 迭代轮数上限（initial_state 设，可请求体覆盖）

    # —— 每轮替换 ——
    round: int  # 当前搜索轮次（1 起；reflect 里 +1）
    gap_queries: list[str]  # reflect 本轮补搜 query
    verdict: str  # "continue" | "enough"
    stop_reason: str  # 收敛原因：enough / max_rounds / no_new_info
    rationale: str  # reflect 判官理由（进 reflect 事件 + result.json）
    pending_hits: list[SearchHit]  # 本轮 search 产出、待 read 抓取的命中

    # —— 跨轮累加 ——
    notes: Annotated[list[Note], operator.add]  # 采纳的精华片段（供 synthesize 引用）
    seen_queries: Annotated[set[str], operator.or_]  # 已搜过的 query（防补搜兜圈）
    seen_urls: Annotated[set[str], operator.or_]  # 已抓过的 URL（防重复抓取）
    sources: Annotated[list[Source], operator.add]  # 引用来源（由 notes 汇总去重）

    # —— 结尾 ——
    answer: str  # synthesize 产出的完整答案
