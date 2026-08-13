"""data/build_multi_intent.py 标注规则测试。"""
from __future__ import annotations

import csv

from intent_recognition.config import DataConfig
from intent_recognition.data.build_multi_intent import (
    annotate_row,
    build_multi_csv,
    rule_alarm_wraps_action,
    rule_conjunction_two_actions,
    rule_homeappliance_wraps_scene,
)


def test_alarm_wraps_action_hit():
    hit = rule_alarm_wraps_action("立即设置一个9点提醒我关灯的闹铃", "Alarm-Update")
    assert hit is not None
    assert hit.sub_intent == "HomeAppliance-Control"


def test_alarm_wraps_action_ignores_other_main():
    # 规则只在主意图为 Alarm-Update 时生效
    assert rule_alarm_wraps_action("立即设置一个9点提醒我关灯的闹铃", "Music-Play") is None


def test_alarm_wraps_action_same_intent_skipped():
    # 补的次意图 == 主意图时不标注（"看电视" 不会给 TVProgram-Play 再标 TVProgram-Play）
    assert rule_alarm_wraps_action("提醒我看电视的闹钟", "TVProgram-Play") is None


def test_homeappliance_wraps_scene_hit():
    hit = rule_homeappliance_wraps_scene("打开音响，我想听听歌", "HomeAppliance-Control")
    assert hit is not None
    assert hit.sub_intent == "Music-Play"


def test_conjunction_same_intent_returns_none():
    # 同意图修饰："找到并播放" 是搜索+播放同一动作，不构成双意图
    assert rule_conjunction_two_actions("找到并播放快乐大本营", "Video-Play") is None


def test_false_positive_words_return_none():
    # 还有（票务剩余）/ 接着（续播）不会命中动作词表
    assert annotate_row("还有双鸭山到淮阴的汽车票吗13号的", "Travel-Query") is None
    assert annotate_row("接着给我续放海洋的下雨天电台节目", "Radio-Listen") is None


def test_annotate_row_end_to_end():
    hit = annotate_row("创建一个周六晚上8点跟朋友去看电影的闹钟", "Alarm-Update")
    assert hit is not None
    assert hit.sub_intent == "FilmTele-Play"
    assert annotate_row("今天天气怎么样", "Weather-Query") is None


def test_build_multi_csv_generates_consistent(tmp_path):
    src = tmp_path / "raw.csv"
    src.write_text(
        "立即设置一个9点提醒我关灯的闹铃\tAlarm-Update\n"
        "今天天气怎么样\tWeather-Query\n",
        encoding="utf-8",
    )
    out = tmp_path / "multi.csv"
    cfg = DataConfig(raw_csv=src, multi_csv=out)

    result = build_multi_csv(cfg)
    assert result[0][2].sub_intent == "HomeAppliance-Control"
    assert result[1][2] is None

    with open(out, encoding="utf-8") as f:
        rows = list(csv.reader(f, delimiter="\t"))
    assert rows[0][1] == "Alarm-Update,HomeAppliance-Control"
    assert rows[1][1] == "Weather-Query"
