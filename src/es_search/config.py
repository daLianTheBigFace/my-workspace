"""ES 检索模块的自包含配置（对齐 rag_retrieval/config.py 的模式）。

与 intent_recognition / sentence_bert / rag_retrieval 相互独立：
拷走本文件夹即可在别处使用（连自己的 ES、灌自己的索引）。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# src/es_search/config.py -> 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 保险：HF 下载缓存固定到 E 盘项目内（默认会写 C 盘）。
# 须在任何 transformers/huggingface import 之前生效。
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "models" / ".hf_cache"))

# 自动读取项目根目录的 .env。已存在的环境变量优先，不覆盖。
from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class EsConfig:
    """ES 连接 + 索引配置。

    - es_url：ES 服务地址（本地默认无认证 http://localhost:9200，week06 同款）
    - index_name：存放汽车手册的索引名
    - default_size：检索默认返回条数
    - rag_index_dir：rag_retrieval 建好的分块索引目录（chunks.json + embeddings.npy），
      存在则直接复用（秒级建库），缺失时 build_index 自行解析 PDF 兜底
    - pdf_path / chunk_size：兜底解析 PDF 用的参数
    """

    es_url: str = field(
        default_factory=lambda: os.environ.get("ES_URL") or "http://localhost:9200"
    )
    index_name: str = "car_manual"
    default_size: int = 10
    rag_index_dir: str = field(
        default_factory=lambda: str(PROJECT_ROOT / "data" / "rag_index")
    )
    pdf_path: str = field(
        default_factory=lambda: str(
            PROJECT_ROOT / "assets" / "Week06" / "汽车知识手册.pdf"
        )
    )
    chunk_size: int = 40
