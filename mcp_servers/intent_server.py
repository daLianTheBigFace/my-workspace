"""意图识别 MCP Server（stdio）。

让 Claude Code 直接调用 intent_recognition 的预测器——不起 uvicorn、不碰 HTTP。
工具：
- predict_intent(text, model="bert", top_k=3)  等价 POST /predict
- list_models()                                 等价 GET /models

由 Claude Code 通过仓库根目录 .mcp.json 拉起本脚本。
"""
from __future__ import annotations

import sys
from pathlib import Path

# 只把 src/ 加进 sys.path（不把项目根加进去，避免 mcp_servers 之类顶层目录撞名）
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mcp.server import MCPServer  # noqa: E402

from intent_recognition.config import PredictorConfig  # noqa: E402
from intent_recognition.predictors import (  # noqa: E402
    PredictorLoadError,
    PredictorNotFoundError,
    available_predictors,
    get_predictor,
    predictor_status,
    split_intents,
)

mcp = MCPServer(
    "intent-recognition",
    title="意图识别推理",
    description="可插拔多方案（BERT / TFIDF）意图识别，直接调用本地模型",
)


@mcp.tool()
def predict_intent(text: str, model: str = "bert", top_k: int = 3) -> dict:
    """识别一句话的意图（等价 POST /predict）。

    Args:
        text: 待识别文本，如「今天天气怎么样」。
        model: 推理方案，"bert"（默认，准确）或 "tfidf"（快，但需 models/tfidf 产物）。
        top_k: 返回 top_k 候选（默认 3）。
    """
    cfg = PredictorConfig()
    try:
        predictor = get_predictor(model)
    except PredictorNotFoundError as e:
        return {"error": str(e), "available": available_predictors()}
    except PredictorLoadError as e:
        return {"error": str(e), "available": available_predictors()}

    ranked = predictor.predict(text, top_k=top_k)
    intents = split_intents(ranked, cfg.multi_intent_threshold)
    main = intents[0] if intents else (ranked[0] if ranked else None)

    return {
        "model": model,
        "text": text,
        "intent": main.intent if main else None,
        "confidence": main.probability if main else None,
        "intents": [
            {"intent": p.intent, "probability": p.probability} for p in intents
        ],
        "top3": [{"intent": p.intent, "probability": p.probability} for p in ranked],
    }


@mcp.tool()
def list_models() -> dict:
    """列出可用推理方案及加载状态（等价 GET /models）。"""
    cfg = PredictorConfig()
    return {
        "default_model": cfg.default_model,
        "models": [
            {"name": m, "status": predictor_status(m)} for m in available_predictors()
        ],
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
