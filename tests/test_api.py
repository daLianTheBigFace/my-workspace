"""API 层测试：TestClient 路由 + 校验 + fake predictor 验证路由逻辑。"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from intent_recognition.api.app import app
from intent_recognition.config import PROJECT_ROOT
from intent_recognition.predictors import BasePredictor, Prediction
from intent_recognition.predictors.base import PREDICTORS, _instances


@pytest.fixture(autouse=True)
def _clean_instances():
    """每个测试后清空单例缓存，避免跨测试污染。"""
    yield
    _instances.clear()


class FakePredictor(BasePredictor):
    name = "fake"

    def predict(self, text: str, top_k: int = 3) -> list[Prediction]:
        return [Prediction(intent="Weather-Query", probability=0.9)]


# 不进 with 上下文 → 不触发 lifespan，不拉 BERT 模型
client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["default_model"] == "bert"
    assert "bert" in body["models"]


def test_models():
    resp = client.get("/models")
    assert resp.status_code == 200
    names = [m["name"] for m in resp.json()["models"]]
    assert "bert" in names
    assert "tfidf" in names


def test_predict_unknown_model():
    resp = client.post("/predict", json={"text": "你好", "model": "nope"})
    assert resp.status_code == 404


def test_predict_empty_text():
    resp = client.post("/predict", json={"text": ""})
    assert resp.status_code == 422


def test_predict_blank_text():
    resp = client.post("/predict", json={"text": "   "})
    assert resp.status_code == 422


def test_predict_with_fake_model(monkeypatch):
    monkeypatch.setitem(PREDICTORS, "fake", FakePredictor)
    resp = client.post("/predict", json={"text": "今天天气怎么样", "model": "fake"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["model"] == "fake"
    assert body["intent"] == "Weather-Query"
    assert body["confidence"] == 0.9
    assert len(body["top3"]) == 1


FINAL_DIR = PROJECT_ROOT / "models" / "final"


@pytest.mark.slow
@pytest.mark.skipif(
    not (FINAL_DIR / "config.json").exists(), reason="需要已训练的 BERT 模型产物"
)
def test_predict_bert_real():
    resp = client.post("/predict", json={"text": "今天天气怎么样"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["model"] == "bert"
    assert len(body["top3"]) == 3
