"""RAG 检索 + 重排 + 问答（多路召回）独立包。

与 intent_recognition / sentence_bert 平级。对外暴露 pipeline.search，
API 路由在 api/app.py（挂到主服务 /rag 前缀）。
"""
from .pipeline import SearchHit, SearchResult, search

__all__ = ["SearchHit", "SearchResult", "search"]
