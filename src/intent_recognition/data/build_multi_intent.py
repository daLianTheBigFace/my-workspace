"""多意图标注：为单标签数据补次意图，生成 dataset_multi.csv。

格式：text \\t main_label,sub_label（逗号分隔，第一个为主意图，次意图可空）。
次意图只补一个、宁缺毋滥；不硬补无映射的动作。

规则（基于对 dataset.csv 全部 12100 行的语义调查，2026-08-13 定稿）：
1. rule_alarm_wraps_action：主=Alarm-Update 且句内含可映射动作词 → 补对应意图
   （"立即设置一个9点提醒我关灯的闹铃"→+HomeAppliance-Control；
    "创建一个周六晚上8点跟朋友去看电影的闹钟"→+FilmTele-Play）。命中 ~44 条。
2. rule_homeappliance_wraps_scene：主=HomeAppliance-Control 且句内含播放场景 → 补场景类
   （"打开音响，让它自动播放轻快的音乐"→+Music-Play；
    "我准备看电影了，把灯调暗一些"→+FilmTele-Play）。命中 ~9 条。
3. rule_conjunction_two_actions：连接词「并/然后/同时」连接不同域动作 → 补第二动作。
   当前数据中连接词两侧几乎全是同意图修饰（"找到并播放""导演并主演"），
   该规则预期 0 命中，保留供数据扩充后使用。

被否决的模式（调查结论，避免引入噪声）：
- 查询混播放：数据中"查询播放/播放查询"是天气查询的冗余固定词块
  （"查询播放周三哈尔滨是晴天吗"），不含真实播放意图，**不补**。
- 假阳性词：还有（票务剩余）、接着（续播）、再看（重复观看）——不会出现在词表中，
  自然不触发。

脚本幂等：规则确定性，重跑结果一致。
"""
from __future__ import annotations

import csv
import random
import re
from collections import Counter
from dataclasses import dataclass

from ..config import DataConfig, PROJECT_ROOT


@dataclass
class RuleHit:
    """一次规则命中：补哪个次意图、为什么。"""

    rule: str
    sub_intent: str
    reason: str


# 动作词 → 意图 映射表。匹配时按词长降序（长词优先，避免"看电视节目"被"看电视"抢占）。
_ACTION_TO_LABEL: dict[str, str] = {
    # 电影 / 剧集
    "去看电影": "FilmTele-Play",
    "看电影": "FilmTele-Play",
    "看个电影": "FilmTele-Play",
    "看部电影": "FilmTele-Play",
    "看电视剧": "FilmTele-Play",
    "看剧": "FilmTele-Play",
    "追剧": "FilmTele-Play",
    "看片": "FilmTele-Play",
    "看戏": "FilmTele-Play",
    # 视频
    "看视频": "Video-Play",
    "看直播": "Video-Play",
    "看动画": "Video-Play",
    "看球赛": "Video-Play",
    "刷视频": "Video-Play",
    # 音乐
    "听音乐会": "Music-Play",
    "播放音乐": "Music-Play",
    "播放歌曲": "Music-Play",
    "听音乐": "Music-Play",
    "听首歌": "Music-Play",
    "听下歌": "Music-Play",
    "单曲循环": "Music-Play",
    "听歌": "Music-Play",
    "放音乐": "Music-Play",
    # 广播 / 相声评书
    "听广播": "Radio-Listen",
    "听电台": "Radio-Listen",
    "听评书": "Radio-Listen",
    "听相声": "Radio-Listen",
    # 电视节目
    "看电视节目": "TVProgram-Play",
    "看电视": "TVProgram-Play",
    "看新闻": "TVProgram-Play",
    "看央视": "TVProgram-Play",
    "看体育": "TVProgram-Play",
    # 家电
    "开空调": "HomeAppliance-Control",
    "关空调": "HomeAppliance-Control",
    "调空调": "HomeAppliance-Control",
    "开灯": "HomeAppliance-Control",
    "关灯": "HomeAppliance-Control",
    "开窗": "HomeAppliance-Control",
    "关窗": "HomeAppliance-Control",
    "开电视": "HomeAppliance-Control",
    "关电视": "HomeAppliance-Control",
    "晾衣服": "HomeAppliance-Control",
    "收衣服": "HomeAppliance-Control",
    # 天气
    "看天气预报": "Weather-Query",
    "看天气": "Weather-Query",
    # 出行
    "去机场": "Travel-Query",
    "去高铁站": "Travel-Query",
    "去火车站": "Travel-Query",
    "坐飞机": "Travel-Query",
    "坐高铁": "Travel-Query",
    "坐火车": "Travel-Query",
    "赶飞机": "Travel-Query",
    "去旅游": "Travel-Query",
    # 注意：不做「出门/去上班」→ Travel-Query 映射——它们是泛指出行，不是查交通信息，
    # 会误标（"我等会出门，帮我把电视机关机" 不应补 Travel-Query）。
}

_CONJUNCTIONS = ("并", "然后", "同时")
# 同意图修饰：动词+并播放 / 导演并主演 等，不构成双意图
_SAME_INTENT_PATTERNS = [
    re.compile(r"(找到|搜索|查找|查).{0,3}(并|并且)?播放"),
    re.compile(r"导演并主演"),
]


def _first_action(text: str, main: str) -> RuleHit | None:
    """返回文本中最长优先命中的动作词（跳过与主意图相同的词）。

    返回 RuleHit（rule 占位，由调用方改写），无命中返回 None。
    """
    for word in sorted(_ACTION_TO_LABEL, key=len, reverse=True):
        if word in text:
            intent = _ACTION_TO_LABEL[word]
            if intent != main:
                return RuleHit(
                    rule="action_word",
                    sub_intent=intent,
                    reason=f"含动作词「{word}」→{intent}",
                )
    return None


def rule_alarm_wraps_action(text: str, main: str) -> RuleHit | None:
    """规则 1：闹钟/提醒主意图里包裹的伴随动作。"""
    if main != "Alarm-Update":
        return None
    hit = _first_action(text, main)
    if hit:
        hit.rule = "alarm_wraps_action"
        hit.reason = "闹钟/提醒类主意图，" + hit.reason
    return hit


def rule_homeappliance_wraps_scene(text: str, main: str) -> RuleHit | None:
    """规则 2：家电控制主意图里包裹的播放场景。"""
    if main != "HomeAppliance-Control":
        return None
    hit = _first_action(text, main)
    if hit:
        hit.rule = "homeappliance_wraps_scene"
        hit.reason = "家电控制主意图，" + hit.reason
    return hit


def rule_conjunction_two_actions(text: str, main: str) -> RuleHit | None:
    """规则 3：连接词连接不同域动作 → 补第二动作。

    数据调查结论：当前数据连接词两侧均为同意图修饰，本规则预期 0 命中。
    """
    if not any(c in text for c in _CONJUNCTIONS):
        return None
    if any(p.search(text) for p in _SAME_INTENT_PATTERNS):
        return None
    hit = _first_action(text, main)
    if hit:
        hit.rule = "conjunction_two_actions"
        hit.reason = "连接词双动作，" + hit.reason
    return hit


_RULES = [
    rule_alarm_wraps_action,
    rule_homeappliance_wraps_scene,
    rule_conjunction_two_actions,
]


def annotate_row(text: str, main: str) -> RuleHit | None:
    """按优先级对一行数据标注次意图；命中返回 RuleHit，否则 None。"""
    for rule in _RULES:
        hit = rule(text, main)
        if hit is not None:
            return hit
    return None


def build_multi_csv(data_config: DataConfig | None = None) -> list[tuple[str, str, RuleHit | None]]:
    """读 dataset.csv，逐行标注，写 dataset_multi.csv（text\\tmain,sub）。

    返回 (text, main, hit) 列表供统计/审核复用。
    """
    cfg = data_config or DataConfig()
    result: list[tuple[str, str, RuleHit | None]] = []
    with open(cfg.raw_csv, "r", encoding="utf-8") as f:
        rows = list(csv.reader(f, delimiter="\t"))

    with open(cfg.multi_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        for text, main in rows:
            hit = annotate_row(text, main)
            label = f"{main},{hit.sub_intent}" if hit else main
            writer.writerow([text, label])
            result.append((text, main, hit))
    return result


def coverage_report(result: list[tuple[str, str, RuleHit | None]]) -> None:
    """打印多标签覆盖率：按规则、按主/次意图分布。"""
    total = len(result)
    hit_rows = [(t, m, h) for t, m, h in result if h is not None]
    multi = len(hit_rows)
    print(f"总行数: {total}")
    print(f"多标签行: {multi} ({multi / total:.2%})")
    print(f"单标签行: {total - multi}")

    by_rule: Counter[str] = Counter()
    by_sub: Counter[str] = Counter()
    by_main: Counter[str] = Counter()
    for _, main, hit in hit_rows:
        by_rule[hit.rule] += 1
        by_sub[hit.sub_intent] += 1
        by_main[main] += 1
    print("\n按规则命中:")
    for rule, n in by_rule.most_common():
        print(f"  {rule}: {n}")
    print("\n按主意图:")
    for main, n in by_main.most_common():
        print(f"  {main}: {n}")
    print("\n按次意图:")
    for sub, n in by_sub.most_common():
        print(f"  {sub}: {n}")


def main() -> None:
    """构建 dataset_multi.csv + 覆盖率报告 + 随机抽样 15 条供审核。"""
    random.seed(42)
    cfg = DataConfig()
    result = build_multi_csv(cfg)
    coverage_report(result)

    print("\n===== 随机抽样 15 条（含次意图） =====")
    hit_rows = [(t, m, h) for t, m, h in result if h is not None]
    for text, main, hit in random.sample(hit_rows, min(15, len(hit_rows))):
        print(f"  [{main} → +{hit.sub_intent}] {text}   ({hit.rule}: {hit.reason})")
    print(f"\n输出: {cfg.multi_csv}")


if __name__ == "__main__":
    main()
