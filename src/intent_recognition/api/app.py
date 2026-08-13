"""FastAPI 推理服务：可插拔多方案意图识别。

启动：uv run uvicorn intent_recognition.api.app:app --reload --port 8000
浏览器测试：http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from ..config import PredictorConfig
from ..predictors import (
    PredictorLoadError,
    PredictorNotFoundError,
    available_predictors,
    get_predictor,
    predictor_status,
    split_intents,
)

logger = logging.getLogger(__name__)


# ---- Pydantic Schemas ----
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="待识别文本")
    model: str = Field(default="bert", description="推理方案，见 GET /models")

    @field_validator("text")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text 不能为空白")
        return v.strip()


class Candidate(BaseModel):
    intent: str
    probability: float


class PredictResponse(BaseModel):
    model: str
    intent: str
    confidence: float
    intents: list[Candidate]  # 平等多意图：所有 ≥ 阈值的意图，各自独立置信度
    sub_intent: str | None = None  # 兼容保留：intents[1]（若有）；无可选 None
    sub_confidence: float | None = None
    top3: list[Candidate]


class HealthResponse(BaseModel):
    status: str
    default_model: str
    models: dict[str, str]


class ModelsResponse(BaseModel):
    models: list[dict[str, str]]


# ---- lifespan 预热：加载默认方案，失败不致命 ----
@asynccontextmanager
async def lifespan(_: FastAPI):
    cfg = PredictorConfig()
    try:
        get_predictor(cfg.default_model)  # 预热 bert，避免首个请求卡顿
    except Exception:
        logger.exception("默认方案 '%s' 预热失败", cfg.default_model)
    yield


app = FastAPI(
    title="意图识别推理服务",
    description="可插拔多方案（BERT / TFIDF / 预留 LLM）意图识别",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["系统"])
def health() -> HealthResponse:
    cfg = PredictorConfig()
    return HealthResponse(
        status="ok",
        default_model=cfg.default_model,
        models={m: predictor_status(m) for m in available_predictors()},
    )


@app.get("/models", response_model=ModelsResponse, tags=["系统"])
def list_models() -> ModelsResponse:
    return ModelsResponse(
        models=[
            {"name": m, "status": predictor_status(m)}
            for m in available_predictors()
        ]
    )


@app.post("/predict", response_model=PredictResponse, tags=["推理"])
def predict(req: PredictRequest) -> PredictResponse:
    try:
        predictor = get_predictor(req.model)
    except PredictorNotFoundError as e:  # 未知方案 → 404 + 可用列表
        raise HTTPException(status_code=404, detail=str(e)) from e
    except PredictorLoadError as e:  # 方案存在但加载失败 → 500
        raise HTTPException(status_code=500, detail=str(e)) from e

    result = predictor.predict(req.text, top_k=3)
    intents = split_intents(result, PredictorConfig().multi_intent_threshold)
    # 主输出 intents 是核心；以下字段为兼容保留
    main = intents[0] if intents else result[0]
    sub = intents[1] if len(intents) > 1 else None
    return PredictResponse(
        model=req.model,
        intent=main.intent,
        confidence=main.probability,
        intents=[
            Candidate(intent=i.intent, probability=i.probability) for i in intents
        ],
        sub_intent=sub.intent if sub else None,
        sub_confidence=sub.probability if sub else None,
        top3=[Candidate(intent=r.intent, probability=r.probability) for r in result],
    )
