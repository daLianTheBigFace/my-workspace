"""predictors 层测试：注册表、单例、迷你 TFIDF 加载链路、真实 BERT（门控）。"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from intent_recognition.config import LABELS, PROJECT_ROOT
from intent_recognition.predictors import (
    BasePredictor,
    Prediction,
    PredictorNotFoundError,
    available_predictors,
    get_predictor,
    predictor_status,
    register_predictor,
    split_intents,
    split_main_sub,
)
from intent_recognition.predictors.base import PREDICTORS, _instances
from intent_recognition.predictors.tfidf import TfidfPredictor


@pytest.fixture(autouse=True)
def _clean_instances():
    """每个测试后清空单例缓存，避免跨测试污染。"""
    yield
    _instances.clear()


class FakePredictor(BasePredictor):
    """轻量 fake：不加载任何模型，只验证注册表/路由逻辑。"""

    name = "fake"

    def predict(self, text: str, top_k: int = 3) -> list[Prediction]:
        return [Prediction(intent="Weather-Query", probability=0.9)]


def test_register_and_singleton():
    assert "fake" not in PREDICTORS
    register_predictor(FakePredictor)
    try:
        a = get_predictor("fake")
        b = get_predictor("fake")
        assert a is b  # 单例：两次获取同一实例
        assert predictor_status("fake") == "ready"
    finally:
        PREDICTORS.pop("fake", None)


def test_unknown_predictor():
    with pytest.raises(PredictorNotFoundError):
        get_predictor("不存在")


def test_status_not_loaded_without_loading():
    """未加载时 status=not_loaded，且不触发构造（不抛错、不拉模型）。"""
    register_predictor(FakePredictor)
    try:
        assert predictor_status("fake") == "not_loaded"
    finally:
        PREDICTORS.pop("fake", None)


def test_available_predictors_registered():
    names = available_predictors()
    assert "bert" in names
    assert "tfidf" in names


def test_split_intents_all_above_threshold():
    """平等多意图：所有 ≥ 阈值的意图都返回，保持降序。"""
    ranked = [
        Prediction(intent="Music-Play", probability=0.9),
        Prediction(intent="HomeAppliance-Control", probability=0.6),
        Prediction(intent="Radio-Listen", probability=0.2),
    ]
    intents = split_intents(ranked, threshold=0.3)
    assert [i.intent for i in intents] == ["Music-Play", "HomeAppliance-Control"]


def test_split_intents_below_threshold_empty():
    """全部 < 阈值 → 空列表。"""
    ranked = [
        Prediction(intent="Music-Play", probability=0.1),
        Prediction(intent="Radio-Listen", probability=0.2),
    ]
    assert split_intents(ranked, threshold=0.3) == []


def test_split_intents_preserves_order():
    """输入乱序也应保持概率降序（调用方保证）。"""
    ranked = [
        Prediction(intent="A", probability=0.9),
        Prediction(intent="B", probability=0.4),
        Prediction(intent="C", probability=0.8),
    ]
    intents = split_intents(ranked, threshold=0.3)
    assert [i.intent for i in intents] == ["A", "B", "C"]  # 按原顺序，不重排


def test_split_main_sub():
    """主 = ranked[0]；次 = ranked[1] 且 ≥ 阈值。"""
    ranked = [
        Prediction(intent="Music-Play", probability=0.9),
        Prediction(intent="HomeAppliance-Control", probability=0.6),
    ]
    main, sub = split_main_sub(ranked, sub_threshold=0.3)
    assert main.intent == "Music-Play"
    assert sub.intent == "HomeAppliance-Control"
    assert sub.probability == 0.6


def test_split_main_sub_below_threshold():
    """次意图概率低于阈值 → None。"""
    ranked = [
        Prediction(intent="Music-Play", probability=0.9),
        Prediction(intent="HomeAppliance-Control", probability=0.2),
    ]
    main, sub = split_main_sub(ranked, sub_threshold=0.3)
    assert main.intent == "Music-Play"
    assert sub is None


def test_split_main_sub_single_candidate():
    """单条候选（单标签产物）→ 次意图恒 None。"""
    main, sub = split_main_sub(
        [Prediction(intent="Weather-Query", probability=0.9)], 0.3
    )
    assert main.intent == "Weather-Query"
    assert sub is None


def _train_mini_tfidf(target: Path) -> None:
    """训一个 3 类迷你 TFIDF 产物，供加载/预测链路测试。"""
    texts = [
        "今天天气", "明天天气", "下雨了",
        "播放音乐", "来首歌", "听歌",
        "你好", "再见", "随便聊聊",
    ]
    label_names = ["weather", "music", "other"]
    labels = ["weather"] * 3 + ["music"] * 3 + ["other"] * 3
    label2id = {n: i for i, n in enumerate(label_names)}
    y = [label2id[n] for n in labels]

    vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 2))
    clf = LogisticRegression()
    clf.fit(vec.fit_transform(texts), y)

    joblib.dump(vec, target / "vectorizer.joblib")
    joblib.dump(clf, target / "classifier.joblib")
    (target / "labels.json").write_text(
        json.dumps({"labels": label_names}, ensure_ascii=False), encoding="utf-8"
    )


def test_tfidf_predict_top3(tmp_path):
    _train_mini_tfidf(tmp_path)
    predictor = TfidfPredictor(tmp_path)
    result = predictor.predict("播放一首歌", top_k=3)

    assert len(result) == 3
    assert all(r.intent in {"weather", "music", "other"} for r in result)
    probs = [r.probability for r in result]
    assert probs == sorted(probs, reverse=True)  # 降序
    assert all(0.0 <= p <= 1.0 for p in probs)


FINAL_DIR = PROJECT_ROOT / "models" / "final"


@pytest.mark.slow
@pytest.mark.skipif(
    not (FINAL_DIR / "config.json").exists(), reason="需要已训练的 BERT 模型产物"
)
def test_bert_predict_top3():
    from intent_recognition.predictors.bert import BertPredictor

    predictor = BertPredictor()
    result = predictor.predict("今天天气怎么样", top_k=3)

    assert len(result) == 3
    assert all(r.intent in LABELS for r in result)
    probs = [r.probability for r in result]
    assert probs == sorted(probs, reverse=True)
