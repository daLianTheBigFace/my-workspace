"""RAG 检索 + 重排 + 问答模块的自包含配置。

与 intent_recognition / sentence_bert 相互独立：拷走本文件夹即可在别处使用。
算法参数照搬 week06（chunk=40、每路 top10、RRF k=60、bge-reranker 重排），
week06 原文件不改。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# src/rag_retrieval/config.py -> 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 保险：HF 下载缓存固定到 E 盘项目内（默认会写 C 盘）。
# 须在任何 transformers/huggingface import 之前生效。
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "models" / ".hf_cache"))

# 自动读取项目根目录的 .env（RAG_LLM_* 等配置）。已存在的环境变量优先，不覆盖。
from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class RagConfig:
    """RAG 检索 + 重排 + 问答配置。

    - pdf_path：知识库 PDF（汽车知识手册）
    - index_dir：build_index 产出的分块索引目录
    - chunk_size：定长分块字符数（照搬 week06 RAG101_05：40）
    - retrieve_top_k：每路召回返回的页面数（照搬 week06：top10 页）
    - rrf_k：RRF 融合平滑系数（照搬 week06 RAG101_07：60）
    - rerank_top_n：送入重排的融合候选页数（照搬 week06 RAG101_06：top3 页，
      用整页文本重排后取最优）
    - min_rerank_score：top1 重排分低于此值视为「没匹配到」，不返回检索结果
      （手册外查询约 0~0.1，手册内相关约 0.7~1.0，0.3 能干净切开）
    - answer_top_n：送入大模型上下文的最多文档页数
    - rerank_model_dir / device：bge-reranker-base 本地路径与计算设备
    - LLM：OpenAI 兼容接口，key / base_url / model 从环境变量读，不硬编码
    """

    pdf_path: str = field(
        default_factory=lambda: str(
            PROJECT_ROOT / "assets" / "Week06" / "汽车知识手册.pdf"
        )
    )
    index_dir: str = field(
        default_factory=lambda: str(PROJECT_ROOT / "data" / "rag_index")
    )
    chunk_size: int = 40
    retrieve_top_k: int = 10
    rrf_k: int = 60
    rerank_top_n: int = 3
    min_rerank_score: float = 0.3
    answer_top_n: int = 3
    rerank_model_dir: str = field(
        default_factory=lambda: str(
            PROJECT_ROOT / "assets" / "models" / "bge-reranker-base"
        )
    )
    device: str = "cuda"  # "cuda" | "cpu"

    # ---- LLM（OpenAI 兼容接口）----
    # 优先读 RAG_LLM_*；未配置时回落 DeepSeek 默认：
    # base_url 与模型名有默认值，key 兼容 DEEPSEEK_API_KEY（.env 里直接可用）。
    @property
    def llm_base_url(self) -> str:
        return os.environ.get("RAG_LLM_BASE_URL") or "https://api.deepseek.com"

    @property
    def llm_api_key(self) -> str | None:
        return os.environ.get("RAG_LLM_API_KEY") or os.environ.get(
            "DEEPSEEK_API_KEY"
        )

    @property
    def llm_model(self) -> str:
        return os.environ.get("RAG_LLM_MODEL") or "deepseek-chat"

    @property
    def answer_enabled(self) -> bool:
        """只要拿到 key 就算启用问答。"""
        return bool(self.llm_api_key)
