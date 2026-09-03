"""pageindex 文档检索的自包含配置。

PageIndex（VectifyAI）是向量无关 / 推理式 RAG：用 LLM 把长文档建成「目录树」索引，
检索时模型对着精简树推理、再读命中的页回答。本包装复用仓库 .env 里的 DeepSeek key，
把索引与索引落点都钉在 E 盘。

约定：
- LLM：litellm 模型名（provider/model）。默认 deepseek/deepseek-chat，用
  PAGEINDEX_MODEL / PAGEINDEX_SUMMARY_MODEL 覆盖（如 anthropic/claude-sonnet-4-6）。
- key：DEEPSEEK_API_KEY 优先，缺省回落 RAG_LLM_API_KEY（与 /rag 问答同一个 key）。
- storage_path：本地索引目录，默认项目 data/pageindex（E 盘、gitignore）。
- 独立包约定：拷走本文件夹 + 配好 .env 即可在别处单跑。

注：任何新缓存/大文件都走 E 盘。uv 缓存已在 pyproject 指到 E:\\uv-cache。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# src/pageindex_svc/config.py -> 项目根目录（E:\project\my-workspace）
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 自动读项目根目录 .env（RAG_LLM_* / DEEPSEEK_API_KEY）。已存在的环境变量优先。
from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

# PageIndex 本地引擎经 litellm 调模型，litellm 认 DEEPSEEK_API_KEY；
# 若用户只配了 RAG_LLM_API_KEY，则注入给 litellm 用。
if not os.getenv("DEEPSEEK_API_KEY") and os.getenv("RAG_LLM_API_KEY"):
    os.environ["DEEPSEEK_API_KEY"] = os.getenv("RAG_LLM_API_KEY", "")


@dataclass
class PageIndexConfig:
    """PageIndex 文档检索配置。

    - model：建树索引用的模型（litellm 名）。默认 deepseek/deepseek-chat。
    - summary_model：节点摘要 / 文档描述 / flash 优化用的模型，默认同上。
    - chat_model：检索问答（推理式检索 + 回答）用的模型，默认同 model。
    - storage_path：本地索引目录（E 盘），PageIndex 每个文档一个子目录落盘。
    - default_pdf：冒烟 / 缺省演示 PDF（Week06 汽车知识手册）。
    """

    model: str = field(
        default_factory=lambda: os.getenv("PAGEINDEX_MODEL", "deepseek/deepseek-chat")
    )
    summary_model: str = field(
        default_factory=lambda: os.getenv(
            "PAGEINDEX_SUMMARY_MODEL",
            os.getenv("PAGEINDEX_MODEL", "deepseek/deepseek-chat"),
        )
    )
    storage_path: str = field(
        default_factory=lambda: str(PROJECT_ROOT / "data" / "pageindex")
    )
    default_pdf: str = field(
        default_factory=lambda: str(PROJECT_ROOT / "assets" / "Week06" / "汽车知识手册.pdf")
    )

    @property
    def index_dir(self) -> Path:
        return Path(self.storage_path)

    @property
    def api_key_set(self) -> bool:
        """是否有可用的 DeepSeek key（建索引与检索都离不开 LLM）。"""
        return bool(os.getenv("DEEPSEEK_API_KEY"))
