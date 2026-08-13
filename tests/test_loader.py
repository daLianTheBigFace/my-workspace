"""data/loader.py 多标签函数测试。"""
from __future__ import annotations

from datasets import ClassLabel, Dataset

from intent_recognition.config import DataConfig, LABELS, ModelConfig
from intent_recognition.data.loader import (
    load_multi_raw_data,
    multi_label_distribution,
    to_multi_hot,
    to_multi_label_dataset,
    to_primary_id,
)


def test_to_multi_hot_basic():
    hot = to_multi_hot(LABELS, "Music-Play,HomeAppliance-Control")
    assert len(hot) == len(LABELS)
    assert hot[LABELS.index("Music-Play")] == 1.0
    assert hot[LABELS.index("HomeAppliance-Control")] == 1.0
    assert sum(hot) == 2.0
    # 必须 float 字面量：供 BERT collator 推断 float32 标签
    assert all(isinstance(v, float) for v in hot)


def test_to_multi_hot_single_label():
    hot = to_multi_hot(LABELS, "Weather-Query")
    assert sum(hot) == 1.0
    assert hot[LABELS.index("Weather-Query")] == 1.0


def test_to_multi_hot_unknown_label_ignored():
    hot = to_multi_hot(LABELS, "Weather-Query,不存在的类")
    assert sum(hot) == 1.0


def test_to_primary_id():
    assert (
        to_primary_id(LABELS, "Music-Play,HomeAppliance-Control")
        == LABELS.index("Music-Play")
    )
    assert to_primary_id(LABELS, "Weather-Query") == LABELS.index("Weather-Query")


def test_load_multi_raw_data(tmp_path):
    csv_ = tmp_path / "m.csv"
    csv_.write_text(
        "播放音乐并调大音量\tMusic-Play,HomeAppliance-Control\n"
        "今天天气怎么样\tWeather-Query\n",
        encoding="utf-8",
    )
    ds = load_multi_raw_data(DataConfig(multi_csv=csv_))
    assert ds.num_rows == 2
    assert ds["label"][0] == "Music-Play,HomeAppliance-Control"
    assert ds["label"][1] == "Weather-Query"


def test_multi_label_distribution():
    ds = Dataset.from_dict({"label": ["A,B", "A", "B,C"]})
    dist = multi_label_distribution(ds)
    assert dist["A"] == 2
    assert dist["B"] == 2
    assert dist["C"] == 1


def test_to_multi_label_dataset(tmp_path):
    csv_ = tmp_path / "m.csv"
    csv_.write_text(
        "播放音乐并调大音量\tMusic-Play,HomeAppliance-Control\n"
        "今天天气怎么样\tWeather-Query\n",
        encoding="utf-8",
    )
    ds = to_multi_label_dataset(ModelConfig(), DataConfig(multi_csv=csv_))

    assert set(ds.column_names) == {"text", "label", "primary"}
    assert isinstance(ds.features["primary"], ClassLabel)
    assert ds.features["primary"].num_classes == len(LABELS)
    assert ds["primary"][0] == LABELS.index("Music-Play")
    assert sum(ds["label"][0]) == 2.0
    assert sum(ds["label"][1]) == 1.0
