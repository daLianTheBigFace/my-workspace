"""planner 节点：去废话 → 拆搜索 query。

plan 节点（图第一个 LLM 角色）：把用户的自然语言问题，提炼成 N 个可搜索的关键词式
query（去掉寒暄/冗余，留下检索能命中的核心概念）。只跑一次（START → plan → search）。
"""
from __future__ import annotations

import logging

from ..schemas import PlanEvent, PlanOutput, SubQuestion
from ..state import AgentState
from . import get_llm

logger = logging.getLogger(__name__)

_PLAN_PROMPT = """你是搜索规划器。把用户问题拆解成若干可独立检索的关键词式 query。

要求：
- 去掉寒暄、语气词、冗余修饰，保留能命中检索的核心概念；
- 每个 query 是简短、可搜索的短语（中文关键词即可）；
- 拆 2~5 个；问题简单时 1 个也行。

用户问题：{question}"""


async def plan(state: AgentState) -> dict[str, object]:
    """plan 节点：去废话 → 拆搜索 query，返回 {"sub_questions", "events"}。

    LLM 只出 query 文本（PlanOutput.questions），id 由代码按序号生成 q0/q1…。
    结构化输出失败时降级为 [原问题]，保证链路不断。
    """
    question = state["question"]
    round_no = state.get("round", 1)

    structured = get_llm().with_structured_output(PlanOutput, method="function_calling")
    try:
        out: PlanOutput = await structured.ainvoke(_PLAN_PROMPT.format(question=question))
        queries = [q.strip() for q in (out.questions or []) if q.strip()]
    except Exception as e:  # noqa: BLE001 —— 结构化输出失败降级
        logger.warning("planner 拆解失败，降级为原问题：%s", e)
        queries = []
    if not queries:
        queries = [question]

    sub_questions = [SubQuestion(id=f"q{i}", question=q) for i, q in enumerate(queries)]
    return {
        "sub_questions": sub_questions,
        "events": [PlanEvent(round=round_no, sub_questions=sub_questions)],
    }
