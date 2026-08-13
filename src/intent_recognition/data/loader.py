"""数据加载与预处理。

原始数据：assets/dataset/dataset.csv —— TSV（tab 分隔）、无表头、
每行两列：text（文本）\t label（意图标签），共 12100 条、12 类。
"""
from __future__ import annotations

import csv
from collections import Counter

from datasets import ClassLabel, Dataset

from ..config import DataConfig, ModelConfig


def load_raw_data(data_config: DataConfig | None = None) -> Dataset:
    """读取 TSV 原始数据，返回含 text / label 字符串的 Dataset。"""
    cfg = data_config or DataConfig()
    texts, labels = [], []
    with open(cfg.raw_csv, "r", encoding="utf-8") as f:
        for text, label in csv.reader(f, delimiter="\t"):
            texts.append(text)
            labels.append(label)

    return Dataset.from_dict({"text": texts, "label": labels})


def to_class_label_dataset(
    model_config: ModelConfig, data_config: DataConfig | None = None
) -> Dataset:
    """把 label 字符串映射为整数 id，并挂上 ClassLabel（顺序 = config.LABELS）。"""
    label2id = {name: i for i, name in enumerate(model_config.labels)}
    ds = load_raw_data(data_config)
    ds = ds.map(lambda ex: {"label": label2id[ex["label"]]})
    ds = ds.cast_column(
        "label",
        ClassLabel(num_classes=model_config.num_labels, names=model_config.labels),
    )
    return ds


def label_distribution(ds: Dataset) -> Counter:
    """打印/返回标签分布（用于确认数据加载正确）。"""
    return Counter(ds["label"])
