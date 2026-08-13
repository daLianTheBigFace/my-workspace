"""车机（智能座舱）多指令数据生成：一句话多个并列意图，含唤醒词。

背景：智能座舱语音的核心形态是"一句话多个并列指令"——
"嗨领克，打开空调，播放周杰伦的夜曲，导航到西湖" 含 3 个意图，每个都应独立高置信。
现有数据最多 2 标签（主+次），3 标签为 0，模型对第 3 个意图只能给低概率（0.2 这种）。

本脚本生成车机风格 2/3 意图并列句：
- 唤醒词池（嗨领克/你好领克/小度小度/小欧小欧…，约 1/3 带唤醒词，避免全带过拟合）
- 每意图一个动作片段池（空调/音乐/广播/导航/天气/日程/视频…）
- 意图组合（2 意图常见组合 + 3 意图核心组合，如 空调+音乐+导航）
- 多指令模板（顿号/逗号/然后/顺便/同时/先…再…/帮我…）连接

依赖：本脚本内部先调用 augment_multi_intent.augment() 重建多标签基础
（build_multi + 变体，确定性、幂等），再在基础之上生成车机句并写回。
每次运行都从相同基础重建 → 幂等（不会重复累积车机句）。

用法：uv run python -m intent_recognition.data.augment_cockpit
"""
from __future__ import annotations

import csv
import random

from ..config import DataConfig
from .augment_multi_intent import augment as augment_multi

# ---- 唤醒词池（空串 = 不带唤醒词） ----
_WAKE = [
    "嗨领克", "你好领克", "领克你好", "小度小度", "小欧小欧",
    "理想同学", "小艺小艺", "嗨，领克", "",
]

# ---- 每意图的动作片段池（车机场景动作） ----
_POOL: dict[str, list[str]] = {
    "HomeAppliance-Control": [
        "打开空调", "空调开到26度", "把空调温度调低", "打开座椅加热",
        "把车窗降下来", "打开天窗", "关掉空调", "打开空气净化器",
        "把温度调到24度", "打开雨刷", "打开除雾", "调高空调风速",
        "打开座椅通风", "把车窗升上去", "打开氛围灯", "调低空调风速",
        "把座椅往前调", "打开后视镜加热", "把车内温度调低点",
    ],
    "Music-Play": [
        "播放周杰伦的夜曲", "放一首轻音乐", "来点周杰伦", "播放我的歌单",
        "放一首老歌", "播点轻快音乐", "放一首钢琴曲", "循环播放这首",
        "放首流行歌", "播放薛之谦的歌", "放一首民谣", "播放车载音乐",
        "放一首新歌", "随机播放歌单",
    ],
    "Radio-Listen": [
        "打开广播", "收听交通台", "调到FM103.9", "播放电台新闻", "收听音乐电台",
        "打开收音机", "调到交通广播", "收听早间新闻", "打开电台",
    ],
    "Travel-Query": [
        "导航到西湖", "导航去公司", "带我去火车站", "导航到最近的加油站",
        "导航回家", "导航去商场", "去机场怎么走", "导航到最近的充电站",
        "导航去医院", "带我去最近的医院", "导航到公司", "去高铁站",
    ],
    "Weather-Query": [
        "查一下今天的天气", "明天会不会下雨", "看看北京的天气", "今天气温多少度",
        "查一下明天下不下雨", "看看今天冷不冷", "查下目的地天气",
    ],
    "Calendar-Query": [
        "查一下明天有什么安排", "看看今天几点开会", "我下周一有什么日程",
        "查一下后天有没有会议", "看看我这周的安排",
    ],
    "Audio-Play": [
        "播放有声小说", "放个评书", "播放相声", "放一段有声书",
    ],
    "Video-Play": [
        "播放车载视频", "看会儿电影", "放一段车机视频", "播放动画片",
    ],
    "TVProgram-Play": [
        "打开车载电视", "播放体育频道", "看央视新闻", "打开电视", "播放新闻频道",
    ],
    "Alarm-Update": [
        "提醒我半小时后去接人", "定个闹钟八点", "提醒我下午三点开会",
        "提醒我明天早上加油", "设置一个明早7点的闹钟", "提醒我下班后取快递",
    ],
}

# ---- 2 意图组合（车机常见） ----
_PAIRS: list[tuple[str, str]] = [
    ("HomeAppliance-Control", "Music-Play"),   # 空调+音乐
    ("HomeAppliance-Control", "Radio-Listen"),  # 空调+广播
    ("Music-Play", "Travel-Query"),            # 音乐+导航
    ("HomeAppliance-Control", "Travel-Query"),  # 空调+导航
    ("Music-Play", "Radio-Listen"),            # 音乐+广播
    ("Travel-Query", "Weather-Query"),         # 导航+天气
    ("HomeAppliance-Control", "Weather-Query"), # 空调+天气
    ("Calendar-Query", "Travel-Query"),        # 日程+导航
    ("HomeAppliance-Control", "Audio-Play"),   # 空调+有声书
    ("Music-Play", "Video-Play"),              # 音乐+视频
    ("HomeAppliance-Control", "Alarm-Update"), # 空调+闹钟
    ("Travel-Query", "Alarm-Update"),          # 导航+提醒
    ("HomeAppliance-Control", "TVProgram-Play"), # 空调+电视
    ("Music-Play", "Calendar-Query"),          # 音乐+日程
]

# ---- 3 意图组合（核心新增，车机多指令） ----
_TRIPLES: list[tuple[str, str, str]] = [
    ("HomeAppliance-Control", "Music-Play", "Travel-Query"),   # 空调+夜曲+导航
    ("HomeAppliance-Control", "Music-Play", "Radio-Listen"),   # 空调+音乐+广播
    ("HomeAppliance-Control", "Music-Play", "Weather-Query"),  # 空调+音乐+天气
    ("HomeAppliance-Control", "Travel-Query", "Calendar-Query"),  # 空调+导航+日程
    ("Music-Play", "Travel-Query", "Weather-Query"),           # 音乐+导航+天气
    ("HomeAppliance-Control", "Radio-Listen", "Travel-Query"), # 空调+广播+导航
    ("HomeAppliance-Control", "Music-Play", "Video-Play"),     # 空调+音乐+视频
    ("Music-Play", "Travel-Query", "Calendar-Query"),          # 音乐+导航+日程
    ("HomeAppliance-Control", "Music-Play", "Alarm-Update"),   # 空调+音乐+闹钟
    ("HomeAppliance-Control", "Travel-Query", "Weather-Query"),  # 空调+导航+天气
    ("HomeAppliance-Control", "Music-Play", "TVProgram-Play"),  # 空调+音乐+电视
    ("Music-Play", "Radio-Listen", "Travel-Query"),            # 音乐+广播+导航
]

# ---- 多指令连接模板：{w}=唤醒词(带逗号) {aN}=动作片段 ----
_PAIR_TEMPLATES = [
    "{w}{a1}，{a2}",
    "{w}同时{a1}和{a2}",
    "{w}先{a1}再{a2}",
    "{w}帮我{a1}，还要{a2}",
    "{w}{a1}顺便{a2}",
    "{w}帮我{a1}，然后{a2}",
    "{w}麻烦{a1}和{a2}",
    "{w}我想{a1}，顺便{a2}",
    "{w}给我{a1}，再{a2}",
    "{w}{a1}，另外{a2}",
]
_TRIPLE_TEMPLATES = [
    "{w}{a1}，{a2}，{a3}",
    "{w}帮我{a1}、{a2}和{a3}",
    "{w}{a1}，然后{a2}，最后{a3}",
    "{w}先{a1}再{a2}，然后{a3}",
    "{w}{a1}，同时{a2}，再{a3}",
    "{w}帮我{a1}，然后{a2}，再{a3}",
    "{w}我想{a1}、{a2}还有{a3}",
    "{w}{a1}，然后顺便{a2}，最后{a3}",
    "{w}麻烦{a1}、{a2}和{a3}",
]

# 每组合目标生成量
_PAIR_COUNT = 10
_TRIPLE_COUNT = 25


def _wake_with_comma(rng: random.Random) -> str:
    """随机唤醒词；带唤醒词时返回 "嗨领克，" 形式（约 1/3）。"""
    w = rng.choice(_WAKE)
    return f"{w}，" if w else ""


def _gen_pair(comb: tuple[str, str], rng: random.Random) -> str:
    tpl = rng.choice(_PAIR_TEMPLATES)
    a1 = rng.choice(_POOL[comb[0]])
    a2 = rng.choice(_POOL[comb[1]])
    return tpl.format(w=_wake_with_comma(rng), a1=a1, a2=a2)


def _gen_triple(comb: tuple[str, str, str], rng: random.Random) -> str:
    tpl = rng.choice(_TRIPLE_TEMPLATES)
    a1 = rng.choice(_POOL[comb[0]])
    a2 = rng.choice(_POOL[comb[1]])
    a3 = rng.choice(_POOL[comb[2]])
    return tpl.format(w=_wake_with_comma(rng), a1=a1, a2=a2, a3=a3)


def generate_cockpit_rows(
    seen: set[str], rng: random.Random
) -> list[tuple[str, str]]:
    """生成车机多指令句（2/3 意图），排除已见行，返回 (text, "a,b[,c]")。"""
    new: list[tuple[str, str]] = []
    # 3 意图优先（核心缺口）
    for comb in _TRIPLES:
        cnt = 0
        for _ in range(_TRIPLE_COUNT * 5):  # 多试几次避免与 seen 冲突
            s = _gen_triple(comb, rng)
            if s in seen:
                continue
            seen.add(s)
            new.append((s, ",".join(comb)))
            cnt += 1
            if cnt >= _TRIPLE_COUNT:
                break
    # 2 意图
    for comb in _PAIRS:
        cnt = 0
        for _ in range(_PAIR_COUNT * 5):
            s = _gen_pair(comb, rng)
            if s in seen:
                continue
            seen.add(s)
            new.append((s, ",".join(comb)))
            cnt += 1
            if cnt >= _PAIR_COUNT:
                break
    return new


def augment_cockpit(
    data_config: DataConfig | None = None, seed: int = 42
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """重建多标签基础 + 追加车机多指令句，全量写回 dataset_multi.csv。

    幂等：先调 augment_multi() 从 raw 全量重建基础（确定性），再在固定基础
    （seen 只含基础行，不含上次车机句）上生成车机句 → 每次输出完全一致。
    返回 (基础多标签, 新增车机句)。
    """
    cfg = data_config or DataConfig()
    rng = random.Random(seed)

    # 1. 重建多标签基础（build_multi + 变体，确定性、幂等）
    augment_multi(cfg, seed=seed)

    # 2. 基础行作 seen（不含任何车机句 → 重跑不累积）
    with open(cfg.multi_csv, "r", encoding="utf-8") as f:
        rows = [(text, label) for text, label in csv.reader(f, delimiter="\t")]
    original_multi = [(t, l) for t, l in rows if "," in l]
    seen = {t for t, _ in rows}

    # 3. 生成车机句
    new = generate_cockpit_rows(seen, rng)

    # 4. 全量写回 = 基础行 + 车机句
    with open(cfg.multi_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        for text, label in rows:
            writer.writerow([text, label])
        for text, label in new:
            writer.writerow([text, label])
    return original_multi, new


def main() -> None:
    cfg = DataConfig()
    orig, new = augment_cockpit(cfg)

    with open(cfg.multi_csv, "r", encoding="utf-8") as f:
        rows = [(t, l) for t, l in csv.reader(f, delimiter="\t")]
    multi = [(t, l) for t, l in rows if "," in l]
    n3 = sum(1 for _, l in multi if len(l.split(",")) >= 3)
    print(f"总行数: {len(rows)}  多标签: {len(multi)} ({len(multi)/len(rows):.2%})  3标签: {n3}")
    print(f"新增车机句: {len(new)}")

    rng = random.Random(7)
    sample = rng.sample(new, min(15, len(new)))
    print("\n===== 车机句抽样 15 条 =====")
    for text, label in sample:
        print(f"  [{label}] {text}")


if __name__ == "__main__":
    main()
