"""synthesize 节点：带引用答案。

synthesize 节点（图最后一个 LLM 角色）：把 judge 采纳的 notes 合成一篇带 [n] 引用的
答案。「检索过程」一节不在这里生成，而是由 output.py 从事件 trace 结构化提炼
（更精确：拆了什么 / 搜了什么 / 读了哪些源 / 判了几轮）。
"""
from __future__ import annotations

import logging

from ..schemas import Source
from ..state import AgentState
from . import get_llm

logger = logging.getLogger(__name__)

_PROMPT = """你是深度研究助手。根据下面编号的证据，回答用户问题。

要求：
- 用 [n] 标注引用来源（n 对应证据编号）；
- 覆盖问题核心要点，客观、不编造证据里没有的内容。

用户问题：{question}

证据：
{evidence}"""


def _build_prompt(state: AgentState) -> str:
    notes = state.get("notes", [])
    seen: dict[str, int] = {}
    lines: list[str] = []
    for n in notes:
        idx = seen.setdefault(n.url, len(seen) + 1)
        lines.append(f"[{idx}] （{n.source}）{n.title}：{n.text}")
    evidence = "\n".join(lines) or "（无证据）"
    return _PROMPT.format(question=state["question"], evidence=evidence)


async def synthesize(state: AgentState) -> dict[str, object]:
    """synthesize 节点：走流式 astream，token 经 astream_events 映射成 answer(delta)。

    返回 {"answer": 完整答案, "sources": 去重后的引用来源}。
    """
    llm = get_llm()
    parts: list[str] = []
    async for chunk in llm.astream(_build_prompt(state)):
        c = chunk.content
        if isinstance(c, str) and c:
            parts.append(c)

    # 汇总引用来源（按 url 去重，顺序 = notes 首次出现顺序）
    sources: list[Source] = []
    seen_urls: set[str] = set()
    for n in state.get("notes", []):
        if n.url not in seen_urls:
            seen_urls.add(n.url)
            sources.append(Source(title=n.title, url=n.url))

    return {"answer": "".join(parts), "sources": sources}
