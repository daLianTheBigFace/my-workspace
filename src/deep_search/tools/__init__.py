"""tools 层：三个非 LLM 工具（联网检索 / 本地检索 / 网页抓取）。"""
from __future__ import annotations

from .web_search import search as web_search
from .local_search import search as local_search
from .reader import fetch

__all__ = ["web_search", "local_search", "fetch"]
