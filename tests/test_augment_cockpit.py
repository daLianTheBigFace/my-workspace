"""data/augment_cockpit.py 测试。"""
from __future__ import annotations

import csv

from intent_recognition.config import DataConfig
from intent_recognition.data.augment_cockpit import augment_cockpit


def test_cockpit_generates_triples(tmp_path):
    raw = tmp_path / "raw.csv"
    raw.write_text(
        "提醒我关灯\tAlarm-Update\n"
        "今天天气怎么样\tWeather-Query\n"
        "播放音乐\tMusic-Play\n"
        "导航到西湖\tTravel-Query\n"
        "打开空调\tHomeAppliance-Control\n",
        encoding="utf-8",
    )
    multi = tmp_path / "m.csv"
    cfg = DataConfig(raw_csv=raw, multi_csv=multi)

    orig, new = augment_cockpit(cfg, seed=1)
    assert len(new) > 0
    # 车机句应含 3 标签（含唤醒词），3 标签样本是核心缺口
    n3 = sum(1 for _, label in new if len(label.split(",")) == 3)
    assert n3 > 0
    # 所有标签必须是 12 类意图里的合法值
    from intent_recognition.config import LABELS

    for _, label in new:
        assert all(intent in LABELS for intent in label.split(","))


def test_cockpit_wake_word_and_connectors(tmp_path):
    """车机句应覆盖唤醒词句式与多指令连接词。"""
    raw = tmp_path / "raw.csv"
    raw.write_text(
        "提醒我关灯\tAlarm-Update\n"
        "今天天气怎么样\tWeather-Query\n"
        "播放音乐\tMusic-Play\n"
        "导航到西湖\tTravel-Query\n"
        "打开空调\tHomeAppliance-Control\n",
        encoding="utf-8",
    )
    cfg = DataConfig(raw_csv=raw, multi_csv=tmp_path / "m.csv")

    _, new = augment_cockpit(cfg, seed=1)
    texts = [t for t, _ in new]
    assert any("领克" in t or "小度" in t or "小欧" in t for t in texts)  # 唤醒词
    assert any(sep in t for sep in ("，", "、", "然后", "顺便", "同时", "先", "再") for t in texts)


def test_cockpit_idempotent(tmp_path):
    """重跑从基础全量重建，产生完全一致的文件（不累积车机句）。"""
    raw = tmp_path / "raw.csv"
    raw.write_text(
        "提醒我关灯\tAlarm-Update\n"
        "今天天气怎么样\tWeather-Query\n"
        "播放音乐\tMusic-Play\n"
        "导航到西湖\tTravel-Query\n"
        "打开空调\tHomeAppliance-Control\n",
        encoding="utf-8",
    )
    cfg = DataConfig(raw_csv=raw, multi_csv=tmp_path / "m.csv")

    _, new1 = augment_cockpit(cfg, seed=1)
    content1 = (tmp_path / "m.csv").read_text(encoding="utf-8")
    assert len(new1) > 0

    _, new2 = augment_cockpit(cfg, seed=1)
    assert [t for t, _ in new1] == [t for t, _ in new2]  # 生成句完全一致
    assert (tmp_path / "m.csv").read_text(encoding="utf-8") == content1
