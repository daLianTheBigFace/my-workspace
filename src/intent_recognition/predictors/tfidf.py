"""TFIDF + LogisticRegression 意图识别预测器：加载 models/tfidf/。"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib

from ..config import LABELS, PredictorConfig
from .base import BasePredictor, Prediction, register_predictor

logger = logging.getLogger(__name__)


@register_predictor
class TfidfPredictor(BasePredictor):
    """基于 TFIDF 特征 + 逻辑回归分类器（predict_proba 输出概率）。"""

    name = "tfidf"

    def __init__(self, tfidf_dir: str | Path | None = None) -> None:
        # tfidf_dir 可注入，方便测试用迷你产物目录；生产走 config
        cfg = PredictorConfig()
        self._dir = Path(tfidf_dir) if tfidf_dir else cfg.tfidf_dir
        self._vectorizer = joblib.load(self._dir / "vectorizer.joblib")
        self._classifier = joblib.load(self._dir / "classifier.joblib")

        # 标签顺序取自 labels.json（自包含），并与 config.LABELS 交叉校验
        labels_path = self._dir / "labels.json"
        self._labels: list[str] = (
            json.loads(labels_path.read_text(encoding="utf-8"))["labels"]
            if labels_path.exists()
            else list(LABELS)
        )
        if list(self._labels) != list(LABELS):
            logger.warning("models/tfidf/labels.json 与 config.LABELS 不一致")

    def predict(self, text: str, top_k: int = 3) -> list[Prediction]:
        X = self._vectorizer.transform([text])
        probs = self._classifier.predict_proba(X)[0]  # 类别顺序 = 训练时 labels.json
        k = min(top_k, len(self._labels))
        order = probs.argsort()[::-1][:k]
        return [
            Prediction(intent=self._labels[i], probability=float(probs[i]))
            for i in order
        ]
