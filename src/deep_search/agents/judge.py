"""判官（judge）：两段式，两个 prompt 都用 DeepSeek。

- 质量过滤 filter()：read 节点内逐篇判「值不值得信、有没有营养」→ FilterVerdict；
- 充分性判断 check()：reflect 节点判「攒的证据够不够回答原问题」→ CheckVerdict。

check() 不能只判「行/不行」，必须同时输出可搜索的 gap_queries，且 prompt 里喂
seen（已搜过的 query），避免补搜只是换个说法兜圈。
"""
from __future__ import annotations

import logging
from typing import Literal

from ..config import DeepSearchConfig
from ..schemas import CheckVerdict, FilterVerdict, Note, ReflectEvent
from ..state import AgentState
from . import get_llm

logger = logging.getLogger(__name__)

_FILTER_PROMPT = """你是信息判官。判断这篇正文是否与用户问题相关、且值得采纳进证据库。

标准：
- 与用户问题相关、有实质信息、来源可信 → useful=true，summary 里浓缩成 1~3 句可引用精华；
- 广告、导航、空壳页、低质/与问题无关的内容 → useful=false，reason 一句话说明。

用户问题：{question}
标题：{title}
来源：{source}
正文（可能截断）：
{text}"""

_CHECK_PROMPT = """你是信息判官。判断当前攒到的证据是否足以回答用户的原问题。

- 证据已覆盖问题核心要点、可给出有依据的回答 → verdict="enough"；
- 还缺关键信息 → verdict="continue"，给出 1~3 个「补搜 query」：可直接检索的关键词短语，
  且不要与「已搜过的 query」重复。

用户原问题：{question}

已搜过的 query：{seen}

已采纳证据：
{notes}"""


async def filter(
    text: str, *, question: str, title: str, url: str, source: Literal["web", "local"]
) -> FilterVerdict:
    """质量过滤（read 节点内逐篇调用）：这篇是否与问题相关、值不值得信。"""
    structured = get_llm().with_structured_output(FilterVerdict, method="function_calling")
    try:
        return await structured.ainvoke(
            _FILTER_PROMPT.format(
                question=question, title=title, source=source, text=text[:4000]
            )
        )
    except Exception as e:  # noqa: BLE001 —— 失败保守采纳，不因判官丢证据
        logger.warning("judge.filter 失败，保守采纳：%s", e)
        return FilterVerdict(useful=True, summary=text[:500], reason="filter 调用失败，保守采纳")


async def check(
    question: str, notes: list[Note], seen_queries: set[str]
) -> CheckVerdict:
    """充分性判断（reflect 节点内调用）：攒的证据够不够回答原问题。"""
    structured = get_llm().with_structured_output(CheckVerdict, method="function_calling")
    notes_text = "\n".join(f"- [{n.title}] {n.text[:300]}" for n in notes) or "（暂无）"
    seen_text = "、".join(sorted(seen_queries)) or "（暂无）"
    try:
        return await structured.ainvoke(
            _CHECK_PROMPT.format(question=question, seen=seen_text, notes=notes_text)
        )
    except Exception as e:  # noqa: BLE001 —— 失败保守结束，避免死循环
        logger.warning("judge.check 失败，保守判 enough：%s", e)
        return CheckVerdict(verdict="enough", rationale="check 调用失败，保守结束")


async def reflect(state: AgentState) -> dict[str, object]:
    """reflect 节点：调 check()，决定回 search 还是进 synthesize。

    收敛三保险：
    ① round >= max_rounds → stop_reason="max_rounds"，硬停；
    ② check 判 enough → stop_reason="enough"；
    ③ 判 continue 但补搜 query 全在 seen_queries → stop_reason="no_new_info"。
    """
    cfg = DeepSearchConfig()
    round_no = state["round"]
    max_rounds = state.get("max_rounds", cfg.max_rounds)

    if round_no >= max_rounds:
        rationale = f"达到最大轮数上限（{max_rounds}）"
        return {
            "verdict": "enough",
            "gap_queries": [],
            "stop_reason": "max_rounds",
            "rationale": rationale,
            "events": [
                ReflectEvent(round=round_no, verdict="enough", gap_queries=[], rationale=rationale)
            ],
        }

    verdict = await check(
        state["question"], state.get("notes", []), state.get("seen_queries", set())
    )

    if verdict.verdict == "enough":
        return {
            "verdict": "enough",
            "gap_queries": [],
            "stop_reason": "enough",
            "rationale": verdict.rationale,
            "events": [
                ReflectEvent(round=round_no, verdict="enough", gap_queries=[], rationale=verdict.rationale)
            ],
        }

    seen = state.get("seen_queries", set())
    gap = [q for q in verdict.gap_queries if q and q not in seen]
    if not gap:
        rationale = verdict.rationale or "补搜 query 均已被搜过"
        return {
            "verdict": "enough",
            "gap_queries": [],
            "stop_reason": "no_new_info",
            "rationale": rationale,
            "events": [
                ReflectEvent(round=round_no, verdict="enough", gap_queries=[], rationale=rationale)
            ],
        }

    return {
        "verdict": "continue",
        "gap_queries": gap,
        "round": round_no + 1,
        "rationale": verdict.rationale,
        "events": [
            ReflectEvent(round=round_no, verdict="continue", gap_queries=gap, rationale=verdict.rationale)
        ],
    }
