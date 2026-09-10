"""graph：StateGraph 编排核心。

5 节点 + 1 条条件回边：
    START → plan → search → read → reflect ──(不够)→ search
                                       └──(够了)→ synthesize → END

search / read 是两个非 LLM 胶水节点（横跨多个 tool），放这里；plan / reflect /
synthesize 是 LLM 角色，分别来自 agents/planner、agents/judge、agents/synthesizer。

SSE 事件传递：每个节点把要发给前端的事件塞进返回 dict 的 "events" 字段（schema 之外
的 key，不进 state），api 层在 astream_events 的 on_chain_end 里取出逐帧发射。
"""
from __future__ import annotations

import asyncio
import logging

from langgraph.graph import END, START, StateGraph

from .agents import judge, planner, synthesizer
from .config import DeepSearchConfig
from .schemas import Note, ReadEvent, SearchEvent, SearchHit
from .state import AgentState
from .tools import fetch, local_search, web_search

logger = logging.getLogger(__name__)


# ---- search 节点：web + local 并行检索 ----

async def _search(state: AgentState) -> dict[str, object]:
    cfg = DeepSearchConfig()
    round_no = state["round"]
    seen_queries = state.get("seen_queries", set())
    seen_urls = state.get("seen_urls", set())

    # 本轮要搜的 query：round 1 用 sub_questions，之后用 reflect 补搜的 gap_queries
    if round_no == 1:
        queries = [sq.question for sq in state.get("sub_questions", [])]
    else:
        queries = state.get("gap_queries", [])

    todo = [q for q in queries if q and q not in seen_queries]
    if not todo:
        return {"events": []}

    run_local = cfg.enable_local and round_no in cfg.local_rounds
    sem = asyncio.Semaphore(cfg.search_concurrency)

    async def one(q: str):
        async with sem:
            web_hits = await web_search(q, max_results=cfg.results_per_query)
            if run_local:
                local_hits = await asyncio.to_thread(
                    local_search, q, top_k=cfg.results_per_query
                )
            else:
                local_hits = []
            return q, web_hits, local_hits

    results = await asyncio.gather(*(one(q) for q in todo))

    new_queries: set[str] = set()
    new_urls: set[str] = set()
    pending: list[SearchHit] = []
    events: list = []

    for q, web_hits, local_hits in results:
        new_queries.add(q)
        for h in web_hits + local_hits:
            if h.url and h.url not in seen_urls and h.url not in new_urls:
                pending.append(h)  # 同轮 + 跨轮按 url 去重，避免重复抓取/重复 filter
            new_urls.add(h.url)
        if web_hits:
            events.append(SearchEvent(round=round_no, query=q, source="web", results=web_hits))
        if local_hits:
            events.append(SearchEvent(round=round_no, query=q, source="local", results=local_hits))

    return {
        "seen_queries": new_queries,
        "seen_urls": new_urls,
        "pending_hits": pending,
        "events": events,
    }


# ---- read 节点：抓取 + judge.filter 质量过滤 ----

async def _read(state: AgentState) -> dict[str, object]:
    cfg = DeepSearchConfig()
    round_no = state["round"]
    hits = state.get("pending_hits", [])
    remaining = max(0, cfg.max_sources - len(state.get("notes", [])))

    new_notes: list[Note] = []
    events: list[ReadEvent] = []

    # 串行抓取（避免并发打爆目标站点），逐篇 fetch + filter
    for h in hits:
        if remaining <= 0:
            break
        try:
            if h.source == "local":
                text = h.snippet  # 本地命中文本已就绪，无需抓网页
            else:
                text = await fetch(h.url, timeout=cfg.fetch_timeout)
        except Exception as e:  # noqa: BLE001 —— 单篇失败不拖垮整轮
            logger.warning("抓取失败 %s: %s", h.url, e)
            events.append(ReadEvent(round=round_no, url=h.url, status="fail", chars=0))
            continue

        chars = len(text)
        if text.strip():
            verdict = await judge.filter(
                text, question=state["question"], title=h.title, url=h.url, source=h.source
            )
            if verdict.useful:
                new_notes.append(
                    Note(
                        text=verdict.summary or text[:500],
                        title=h.title,
                        url=h.url,
                        source=h.source,
                        round=round_no,
                    )
                )
        events.append(ReadEvent(round=round_no, url=h.url, status="ok", chars=chars))
        remaining -= 1

    return {"notes": new_notes, "events": events}


# ---- 条件路由 + 建图 ----

def _route_after_reflect(state: AgentState) -> str:
    return "synthesize" if state.get("verdict") == "enough" else "search"


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("plan", planner.plan)
    g.add_node("search", _search)
    g.add_node("read", _read)
    g.add_node("reflect", judge.reflect)
    g.add_node("synthesize", synthesizer.synthesize)

    g.add_edge(START, "plan")
    g.add_edge("plan", "search")
    g.add_edge("search", "read")
    g.add_edge("read", "reflect")
    g.add_conditional_edges(
        "reflect",
        _route_after_reflect,
        {"search": "search", "synthesize": "synthesize"},
    )
    g.add_edge("synthesize", END)
    return g.compile()


graph = build_graph()
