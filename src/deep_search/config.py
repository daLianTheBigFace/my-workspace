"""deep_search 深度搜索服务的自包含配置。

沿用仓库 config 惯例（PROJECT_ROOT = parents[2]、load_dotenv、路径钉 E 盘）：
- Tavily key 从环境变量 TAVILY_API_KEY 读，缺了联网不可用；
- LLM 全部复用仓库 .env 的 DeepSeek（RAG_LLM_* 优先，回落 DEEPSEEK_API_KEY）；
- 输出目录 outputs/deep_search 钉在项目根（E 盘、gitignore）。

约定：本包拷走 + 配好 .env 即可单跑。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# src/deep_search/config.py -> 项目根目录（E:\project\my-workspace）
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 自动读项目根目录 .env（RAG_LLM_* / DEEPSEEK_API_KEY / TAVILY_API_KEY）。
# 已存在的环境变量优先，不覆盖。
from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class DeepSearchConfig:
    """Deep Research 智能体的连接与预算配置。

    - max_rounds：迭代搜索轮数上限（收敛硬停保险 1）。
    - max_sources：总抓取页面数上限（防止越搜越多）。
    - results_per_query：Tavily 每 query 返回条数。
    - search_concurrency：子问题/补搜 query 并行检索度。
    - fetch_timeout：单页抓取超时（秒）。
    - enable_local / local_rounds：是否启用本地手册 tool，及哪些轮次跑（默认仅 round 1）。
    - output_dir：落盘目录（E 盘，gitignore），流结束后写 report.md + result.json。
    """

    max_rounds: int = 3
    max_sources: int = 12
    results_per_query: int = 5
    search_concurrency: int = 3
    fetch_timeout: int = 10
    enable_local: bool = True
    local_rounds: frozenset[int] = frozenset({1})  # round 1 起算，默认只在首轮跑本地
    output_dir: str = field(
        default_factory=lambda: str(PROJECT_ROOT / "outputs" / "deep_search")
    )

    # ---- 联网检索（Tavily）----
    @property
    def tavily_api_key(self) -> str | None:
        return os.environ.get("TAVILY_API_KEY")

    @property
    def web_ready(self) -> bool:
        """联网检索可用性：有 Tavily key 就算就绪。"""
        return bool(self.tavily_api_key)

    # ---- LLM（DeepSeek，OpenAI 兼容）----
    # 复用 rag_retrieval 的 RAG_LLM_*；未配置时回落 DeepSeek 默认（.env 里直接可用）。
    @property
    def llm_base_url(self) -> str:
        return os.environ.get("RAG_LLM_BASE_URL") or "https://api.deepseek.com"

    @property
    def llm_api_key(self) -> str | None:
        return os.environ.get("RAG_LLM_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")

    @property
    def llm_model(self) -> str:
        return os.environ.get("RAG_LLM_MODEL") or "deepseek-chat"

    @property
    def llm_ready(self) -> bool:
        """planner / judge / synthesizer 都离不开 LLM，拿不到 key 算没就绪。"""
        return bool(self.llm_api_key)
