"""deep_search 深度搜索服务：Deep Research 智能体（多步自主检索 + 合成）。

仓库第 6 个能力包，形态与前 5 个「一次检索出结果」不同：LangGraph 编排
plan → search → read → reflect → synthesize，联网（Tavily）+ 本地手册双 tool，
对外暴露 SSE 流式接口。对外能力在 deep_search.api（POST /deep_search/search）。
"""
from __future__ import annotations

__all__ = []
