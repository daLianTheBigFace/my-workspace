"""预测器抽象层：统一推理接口 + 注册表 + 单例惰性加载。

所有推理方案（BERT / TFIDF / 未来的 LLM）实现 BasePredictor，
通过 @register_predictor 注册进 PREDICTORS，调用方按名字取单例。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class Prediction:
    """单条预测结果：意图名 + 概率。"""

    intent: str
    probability: float


class BasePredictor(ABC):
    """所有推理方案的统一接口。子类实现 predict(text, top_k)。"""

    name: ClassVar[str]  # 注册名，如 "bert" / "tfidf"

    @abstractmethod
    def predict(self, text: str, top_k: int = 3) -> list[Prediction]:
        """返回按概率降序的 top_k 候选，概率在 [0,1]。"""


# ---- 注册表与单例缓存 ----
PREDICTORS: dict[str, type[BasePredictor]] = {}  # name -> 类
_instances: dict[str, BasePredictor] = {}  # name -> 单例
_load_errors: dict[str, Exception] = {}  # name -> 最近一次加载异常


class PredictorNotFoundError(KeyError):
    """未知推理方案。"""


class PredictorLoadError(RuntimeError):
    """方案存在但加载失败（模型缺失等）。"""


def split_intents(
    ranked: list[Prediction], threshold: float
) -> list[Prediction]:
    """平等多意图：返回所有概率 ≥ threshold 的意图，各自独立置信度，保持降序。

    不分主次——一句话里每个满足阈值的意图都独立返回。
    """
    return [p for p in ranked if p.probability >= threshold]


def split_main_sub(
    ranked: list[Prediction], sub_threshold: float
) -> tuple[Prediction, Prediction | None]:
    """从按概率降序的候选里拆出主、次意图（旧版主次模式，向后兼容保留）。

    - 主意图 = ranked[0]（概率最高）
    - 次意图 = ranked[1]，且概率 ≥ sub_threshold，否则 None
    - 兼容单标签模型（ranked 可能只有 1 条 → 次意图恒为 None）
    """
    if not ranked:
        raise ValueError("ranked 不能为空")
    main = ranked[0]
    sub = (
        ranked[1]
        if len(ranked) > 1 and ranked[1].probability >= sub_threshold
        else None
    )
    return main, sub


def register_predictor(cls: type[BasePredictor]) -> type[BasePredictor]:
    """装饰器：把类按 cls.name 注册进 PREDICTORS。"""
    PREDICTORS[cls.name] = cls
    return cls


def get_predictor(name: str) -> BasePredictor:
    """按名取单例：每个 Predictor 只构造一次（惰性加载）。"""
    if name not in PREDICTORS:
        raise PredictorNotFoundError(
            f"未知推理方案 '{name}'，可用：{sorted(PREDICTORS)}"
        )
    if name in _instances:
        return _instances[name]
    try:
        inst = PREDICTORS[name]()
    except Exception as e:
        _load_errors[name] = e
        raise PredictorLoadError(f"推理方案 '{name}' 加载失败：{e}") from e
    _instances[name] = inst
    return inst


def available_predictors() -> list[str]:
    """当前已注册的方案名（供 GET /models 使用）。"""
    return sorted(PREDICTORS)


def predictor_status(name: str) -> str:
    """返回 'ready' | 'error' | 'not_loaded'。

    只读缓存状态，不触发加载——避免健康检查拉起重型模型。
    """
    if name in _instances:
        return "ready"
    if name in _load_errors:
        return "error"
    return "not_loaded"
