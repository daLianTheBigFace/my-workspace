"""sentence-BERT 句向量模块的自包含配置。

与 intent_recognition 相互独立：拷走本文件夹即可在别处使用。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# src/sentence_bert/config.py -> 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 保险：HF 下载缓存固定到 E 盘项目内（默认会写 C 盘）。
# 须在任何 transformers/huggingface import 之前生效。
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "models" / ".hf_cache"))


@dataclass
class SentenceBertConfig:
    """句向量（sentence-BERT）推理配置：模型路径 + 相似度判定阈值。

    - sentence_dir：assets/models 下的自包含句向量模型（bge 已下载就位）
    - device：句向量计算设备，"auto" 自动选 GPU/CPU
    - similarity_threshold：余弦相似度 ≥ 此值视为"匹配"
      （冒烟测试参考：同类 0.79 / 异类 ~0.25，0.5 能干净切开）
    """

    sentence_dir: str = field(
        default_factory=lambda: str(PROJECT_ROOT / "assets" / "models" / "bge-small-zh-v1.5")
    )
    device: str = "cuda"  # "auto" | "cuda" | "cpu"
    similarity_threshold: float = 0.5
