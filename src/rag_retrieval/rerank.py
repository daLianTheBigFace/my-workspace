"""bge-reranker-base 交叉编码重排（照搬 week06 RAG101_06）。

cross-encoder 直接对 (query, doc) 对打分，比双塔句向量的
「先各自编码再比对」更能捕捉细粒度相关。本地模型在 assets/models/bge-reranker-base。

模型懒加载：未下载时 get_reranker() 抛异常，pipeline 会优雅降级为纯 RRF。
"""
from __future__ import annotations

import logging

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import RagConfig

logger = logging.getLogger(__name__)

_model = None
_tokenizer = None


def get_reranker():
    """单例惰性加载交叉编码重排模型（有就用，没有才加载）。"""
    global _model, _tokenizer
    if _model is None:
        cfg = RagConfig()
        _tokenizer = AutoTokenizer.from_pretrained(cfg.rerank_model_dir)
        _model = AutoModelForSequenceClassification.from_pretrained(
            cfg.rerank_model_dir
        )
        if cfg.device.startswith("cuda"):
            _model = _model.to(cfg.device)
        _model.eval()
    return _model, _tokenizer


def rerank_scores(query: str, texts: list[str]) -> list[float]:
    """对 query 与每个候选文本打分，返回 sigmoid 归一化到 0~1 的相关分。

    照搬 week06 RAG101_06：tokenizer 一次打包所有 (query, doc) 对，
    max_length=512 截断，取 logits。差别：用 sigmoid 归一化便于排序展示。
    """
    if not texts:
        return []
    model, tokenizer = get_reranker()
    pairs = [[query, text] for text in texts]
    inputs = tokenizer(
        pairs, padding=True, truncation=True, return_tensors="pt", max_length=512
    )
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs, return_dict=True).logits.view(-1).float()
    return torch.sigmoid(logits).cpu().tolist()


def reranker_ready() -> bool:
    """重排模型是否可用（尝试加载，失败返回 False 不致命）。"""
    try:
        get_reranker()
        return True
    except Exception:
        logger.exception("reranker 加载失败（模型未下载？）")
        return False
