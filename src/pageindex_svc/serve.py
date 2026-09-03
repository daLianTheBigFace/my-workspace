"""pageindex 文档检索独立服务入口。

两种起法：
- uv run python -m pageindex_svc.serve        （默认 127.0.0.1:8010）
- uv run uvicorn pageindex_svc.serve:app --port 8010

也可以不单起：主服务（intent_recognition.api.app）已 include 本包路由，
挂 /pageindex 前缀即可一起用。
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from .api import router, warm


@asynccontextmanager
async def lifespan(_: FastAPI):
    # 启动预热：检查 key / 建索引目录，缺了只告警（warm 内部有 try 语义）
    warm()
    yield


app = FastAPI(
    title="pageindex 文档检索服务",
    description=(
        "包装 VectifyAI PageIndex 本地引擎（推理式无向量 RAG，DeepSeek）。"
        "把 PDF 建成树索引，再对索引做推理式检索 + 问答。"
    ),
    version="0.1.0",
    lifespan=lifespan,
)
app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8010)
