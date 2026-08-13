"""数据加载与预处理。

原始数据：assets/dataset/dataset.csv —— TSV（tab 分隔）、无表头、
每行两列：text（文本）\t label（意图标签），共 12100 条、12 类。
"""
from __future__ import annotations

import csv
from collections import Counter
from typing import Sequence

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


# ---- 多标签（主次意图） ----
def load_multi_raw_data(data_config: DataConfig | None = None) -> Dataset:
    """读取多标签 TSV（text \t label1,label2），label 字段保持字符串原样。

    次意图可空（单标签行 label 就是 "Weather-Query"）。
    """
    cfg = data_config or DataConfig()
    texts, labels = [], []
    with open(cfg.multi_csv, "r", encoding="utf-8") as f:
        for text, label in csv.reader(f, delimiter="\t"):
            texts.append(text)
            labels.append(label)
    return Dataset.from_dict({"text": texts, "label": labels})


def to_multi_hot(labels: Sequence[str], label_str: str) -> list[float]:
    """把 "主,次" 字符串转成 len(labels) 维 multi-hot float 列表。

    主、次均为 1.0；必须用 float 字面量（供 BERT BCE 的 collator 推断 float32）。
    """
    label2id = {name: i for i, name in enumerate(labels)}
    hot = [0.0] * len(labels)
    for name in label_str.split(","):
        name = name.strip()
        if name in label2id:
            hot[label2id[name]] = 1.0
    return hot


def to_primary_id(labels: Sequence[str], label_str: str) -> int:
    """取逗号第一个（主意图）的整数 id，供分层划分。"""
    main = label_str.split(",")[0].strip()
    return labels.index(main)


def to_multi_label_dataset(
    model_config: ModelConfig, data_config: DataConfig | None = None
) -> Dataset:
    """多标签训练集：text / label(multi-hot float) / primary(ClassLabel)。

    primary 列 cast 成 ClassLabel，供 train_test_split 的 stratify_by_column 使用
    （datasets 只认 ClassLabel 类型的分层列）。
    """
    ds = load_multi_raw_data(data_config)
    ds = ds.map(
        lambda ex: {
            "label": to_multi_hot(model_config.labels, ex["label"]),
            "primary": to_primary_id(model_config.labels, ex["label"]),
        }
    )
    return ds.cast_column(
        "primary",
        ClassLabel(num_classes=model_config.num_labels, names=model_config.labels),
    )


def multi_label_distribution(ds: Dataset) -> Counter:
    """按单个标签统计多标签数据分布（切分逗号后逐个计数）。"""
    counts: Counter[str] = Counter()
    for label_str in ds["label"]:
        for name in label_str.split(","):
            name = name.strip()
            if name:
                counts[name] += 1
    return counts
