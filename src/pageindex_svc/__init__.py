"""pageindex 文档检索：包装 VectifyAI PageIndex 本地引擎（推理式无向量 RAG）。

本地包装包刻意不用 `pageindex` 做顶层名，避免与 pip/git 依赖 `pageindex` 同名冲突。
对外能力在 `pageindex_svc.api`（FastAPI /pageindex 路由），核心逻辑在 service/config。
"""
from __future__ import annotations

__all__ = []
