"""data/augment_multi_intent.py 测试。"""
from __future__ import annotations

import csv

from intent_recognition.config import DataConfig
from intent_recognition.data.augment_multi_intent import augment


def test_augment_generates_new_rows(tmp_path):
    raw = tmp_path / "raw.csv"
    raw.write_text(
        "提醒我关灯\tAlarm-Update\n"
        "今天天气怎么样\tWeather-Query\n",
        encoding="utf-8",
    )
    multi = tmp_path / "m.csv"
    cfg = DataConfig(raw_csv=raw, multi_csv=multi)

    orig, new = augment(cfg, seed=1)
    assert len(orig) == 1  # "提醒我关灯" 标出 Alarm,HomeAppliance 真标注
    assert len(new) > 0  # 生成变体
    assert all("," in label for _, label in new)  # 全是多标签

    raw_rows = {t for t, _ in csv.reader(open(raw, encoding="utf-8"), delimiter="\t")}
    assert all(x not in raw_rows for x, _ in new)  # 变体不与 raw 原文重复


def test_augment_idempotent(tmp_path):
    """重跑从 raw 全量重建，产生完全一致的文件（确定性 + seed 固定）。"""
    raw = tmp_path / "raw.csv"
    raw.write_text(
        "提醒我关灯\tAlarm-Update\n"
        "今天天气怎么样\tWeather-Query\n",
        encoding="utf-8",
    )
    multi = tmp_path / "m.csv"
    cfg = DataConfig(raw_csv=raw, multi_csv=multi)

    _, new1 = augment(cfg, seed=1)
    assert len(new1) > 0
    content1 = multi.read_text(encoding="utf-8")

    # 重跑应输出完全相同的文件（无重复、无累积）
    _, new2 = augment(cfg, seed=1)
    assert [t for t, _ in new1] == [t for t, _ in new2]
    assert multi.read_text(encoding="utf-8") == content1
