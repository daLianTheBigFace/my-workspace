"""BERT 意图识别预测器：加载 models/final/（原地不动的微调产物）。"""
from __future__ import annotations

import logging
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..config import LABELS, ModelConfig, PredictorConfig
from .base import BasePredictor, Prediction, register_predictor

logger = logging.getLogger(__name__)


@register_predictor
class BertPredictor(BasePredictor):
    """基于微调后中文 BERT 的序列分类预测器。"""

    name = "bert"

    def __init__(self, final_dir: str | Path | None = None) -> None:
        # final_dir 可注入，方便测试用替代目录；生产走 config
        cfg = PredictorConfig()
        final = Path(final_dir) if final_dir else cfg.final_dir
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 隐患(a)：final 的 config.json 无 num_labels 键 —— 显式传入，规避输出维度不匹配
        self._model = AutoModelForSequenceClassification.from_pretrained(
            str(final), num_labels=len(LABELS)
        )

        # 隐患(b)：final 缺 vocab.txt / special_tokens_map.json —— 优先 final，失败回退原始目录
        self._tokenizer = self._load_tokenizer(final, ModelConfig().pretrained_model)

        # 多标签模式检查：predict 用 sigmoid；旧单标签产物（softmax 头）不阻断，仅告警
        if (
            getattr(self._model.config, "problem_type", "single_label_classification")
            != "multi_label_classification"
        ):
            logger.warning(
                "models/final 的 problem_type=%r，非 multi_label_classification。"
                "多标签头需用 train/bert.py 重训；当前 sigmoid 输出在旧单标签头上不准确。",
                self._model.config.problem_type,
            )

        self._model.to(self._device).eval()
        logger.info("BertPredictor 加载完成，device=%s", self._device)

    @staticmethod
    def _load_tokenizer(final_dir: Path, source_dir: Path) -> AutoTokenizer:
        """优先从 models/final 加载（tokenizer.json 走 fast path）；
        失败则回退到原始 bert-base-chinese（同一分词器，保证预测一致）。"""
        try:
            return AutoTokenizer.from_pretrained(str(final_dir))
        except Exception:
            logger.warning("models/final 分词器加载失败，回退到原始 bert-base-chinese")
            return AutoTokenizer.from_pretrained(str(source_dir))

    @torch.inference_mode()
    def predict(self, text: str, top_k: int = 3) -> list[Prediction]:
        # 已核实 models/final 的 id2label 顺序 == config.LABELS，直接用 LABELS 索引
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=ModelConfig().max_length,
        ).to(self._device)
        logits = self._model(**inputs).logits
        # 多标签头：sigmoid 独立激活（单标签 softmax 头在此数学上也成立，各概率独立不过量纲可跨类比较）
        probs = torch.sigmoid(logits)[0].float()
        k = min(top_k, len(LABELS))
        values, indices = torch.topk(probs, k)
        return [
            Prediction(intent=LABELS[i], probability=float(p))
            for p, i in zip(values.cpu(), indices.cpu())
        ]
