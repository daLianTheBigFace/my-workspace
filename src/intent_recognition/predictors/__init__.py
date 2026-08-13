"""预测器包：统一入口 + 触发各方案注册。

import 子模块会执行 @register_predictor 装饰器，把方案填进 PREDICTORS。
llm.py 是预留扩展点模板，不在此引入（见 llm.py 的接入步骤）。
"""
from __future__ import annotations

from .base import (
    PREDICTORS,
    BasePredictor,
    Prediction,
    PredictorLoadError,
    PredictorNotFoundError,
    available_predictors,
    get_predictor,
    predictor_status,
    register_predictor,
    split_intents,
    split_main_sub,
)
from . import bert, tfidf  # noqa: F401  (注册副作用)

__all__ = [
    "PREDICTORS",
    "BasePredictor",
    "Prediction",
    "PredictorLoadError",
    "PredictorNotFoundError",
    "available_predictors",
    "get_predictor",
    "predictor_status",
    "register_predictor",
    "split_intents",
    "split_main_sub",
]
