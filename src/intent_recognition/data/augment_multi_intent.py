"""多标签数据扩充：缓解次意图覆盖率过低（0.36% → 目标 5%+）。

背景：build_multi_intent.py 只能从真实单标签数据标出 43 条多标签句，
占比太低导致 BERT 多标签头几乎学不到次意图（sigmoid 第二意图概率全部 <0.1）。
本脚本基于真句骨架 + 槽位生成**受控自然变体**，并补充训练数据里缺失的
典型组合（如 "播放音乐并调大音量"→Music-Play,HomeAppliance-Control，原数据 0 条）。

设计原则：
- 模板锚定真句语感（闹钟类句式、设备类句式），不做纯随机拼接
- 时间/动作槽位领域匹配，避免"凌晨3点提醒我晾衣服"式不自然句
- 生成结果去重、与已有行不冲突；重跑幂等（seen 初始化读入现有全部行）
- 主意图集中在 Alarm-Update / HomeAppliance-Control，与真标注分布一致

用法：uv run python -m intent_recognition.data.augment_multi_intent
输出：覆盖写 assets/dataset/dataset_multi.csv（原 43 条真标注 + 新增变体）
"""
from __future__ import annotations

import csv
import itertools
import random
from collections import Counter

from ..config import DataConfig
from .build_multi_intent import build_multi_csv

# ---- 通用时间槽（自然中文时间短语，全部通用场景成立） ----
_TIMES = [
    "7点", "8点半", "9点", "10点", "下午3点", "晚上9点",
    "明天上午9点", "明天下午4点", "周六晚上8点", "周五晚上11点",
    "下周一下午4点", "3号下午4点", "8月15号上午", "1月1号10点",
    "6月10号", "后天下午2点", "每天10点", "中午12点半", "每晚8点",
    "周日上午7点", "明天中午",
]

# ---- 各 (主,次) 组合的生成规格：句式模板 × 动作槽 ----
_GENERATORS: dict[tuple[str, str], dict] = {
    # ===== Alarm-Update 主意图（闹钟/提醒 包裹伴随动作） =====
    ("Alarm-Update", "HomeAppliance-Control"): {
        "templates": [
            "提醒我{time}{action}",
            "{time}提醒我{action}",
            "设置一个{time}{action}的闹铃",
            "帮我定一个{time}{action}的提醒",
            "记得在{time}提醒我{action}",
            "明天{time}记得提醒我{action}",
        ],
        "actions": ["关灯", "开空调", "关空调", "调空调", "开电视", "关电视", "开窗", "关窗", "晾衣服", "收衣服"],
        "count": 120,
    },
    ("Alarm-Update", "Travel-Query"): {
        "templates": [
            "提醒我{time}{action}",
            "{time}提醒我{action}",
            "帮我定一个{time}{action}的闹钟",
            "记得在{time}提醒我{action}",
            "设置{time}{action}的备忘录",
        ],
        "actions": ["去机场", "去火车站", "去高铁站", "坐飞机", "坐高铁", "坐火车", "赶飞机", "去旅游", "坐飞机去上海", "坐高铁去北京", "赶飞机去广州", "坐火车去成都"],
        "count": 90,
    },
    ("Alarm-Update", "FilmTele-Play"): {
        "templates": [
            "提醒我{time}{action}",
            "{time}提醒我{action}",
            "帮我定一个{time}{action}的闹钟",
            "创建{time}{action}的日程",
            "记得{time}提醒我{action}",
        ],
        "actions": ["看电影", "去看电影", "看个电影", "追剧", "看电视剧", "看部电影"],
        "count": 70,
    },
    ("Alarm-Update", "Video-Play"): {
        "templates": [
            "提醒我{time}{action}",
            "{time}提醒我{action}",
            "帮我定一个{time}{action}的闹钟",
            "创建{time}{action}的日程",
            "请为我设置一个{time}的闹钟我要{action}",
        ],
        "actions": ["看直播", "看球赛", "看视频", "看动画", "刷视频", "看比赛"],
        "count": 60,
    },
    ("Alarm-Update", "TVProgram-Play"): {
        "templates": [
            "提醒我{time}{action}",
            "{time}叫我{action}",
            "帮我定一个{time}{action}的闹钟",
            "记得{time}提醒我{action}",
        ],
        "actions": ["看电视", "看新闻", "看央视", "看体育", "看电视节目"],
        "count": 50,
    },
    ("Alarm-Update", "Music-Play"): {
        "templates": [
            "提醒我{time}{action}",
            "{time}提醒我{action}",
            "帮我定一个{time}{action}的闹钟",
            "记得{time}提醒我{action}",
        ],
        "actions": ["听歌", "听音乐", "听音乐会", "听首歌"],
        "count": 30,
    },
    ("Alarm-Update", "Radio-Listen"): {
        "templates": [
            "提醒我{time}{action}",
            "{time}提醒我{action}",
            "帮我定一个{time}{action}的闹铃",
            "记得{time}提醒我{action}",
        ],
        "actions": ["听广播", "听电台", "听评书", "听相声"],
        "count": 25,
    },
    # ===== HomeAppliance-Control 主意图（设备控制 包裹播放场景） =====
    ("HomeAppliance-Control", "Music-Play"): {
        "templates": [
            "打开{dev}，播放{music}",
            "打开{dev}我想{listen}",
            "{play}把{ctrl}调{dir}",
            "{listen}帮我把{ctrl}{dir}",
        ],
        "actions": [],  # 不用 actions 槽，用模板专用槽
        "slots": {
            "dev": ["音响", "音箱", "智能音箱"],
            "music": ["音乐", "一首歌", "轻快的音乐", "喜欢的歌"],
            "listen": ["听歌", "听音乐", "听首歌", "我想听歌"],
            "play": ["放音乐", "播放音乐", "放首歌", "来点音乐"],
            "ctrl": ["音量", "空调温度", "暖风"],
            "dir": ["调大", "调高", "调低", "调小"],
        },
        "count": 60,
    },
    ("HomeAppliance-Control", "FilmTele-Play"): {
        "templates": [
            "我想{watch}帮我把{dev}调{dir}",
            "{watch}帮我把{dev}调{dir}",
            "{watch}给我打开{dev}",
            "打开{dev}我要{watch}",
            "我想{watch}把{dev2}打开",
        ],
        "slots": {
            "watch": ["看电影", "看一场电影", "看部电影"],
            "dev": ["灯", "窗帘", "灯光"],
            "dev2": ["投影仪", "电视"],
            "dir": ["暗一些", "调暗", "暗点"],
        },
        "count": 40,
    },
    ("HomeAppliance-Control", "TVProgram-Play"): {
        "templates": [
            "我想{watch}帮我把{dev}打开",
            "{watch}帮我把{dev}打开",
            "我想{watch}帮我开下{dev}",
        ],
        "slots": {
            "watch": ["看新闻", "看电视", "看体育节目", "看央视"],
            "dev": ["电视", "电视机"],
        },
        "count": 25,
    },
    ("HomeAppliance-Control", "Travel-Query"): {
        "templates": [
            "我要{travel}帮我把{dev}设置为{mode}模式",
            "我{travel}把{dev}设成{mode}模式",
        ],
        "slots": {
            "travel": ["出去旅游几天", "出门几天", "去外地几天"],
            "dev": ["灯", "空调", "热水器"],
            "mode": ["离家", "外出"],
        },
        "count": 20,
    },
    # ===== 播放主意图 + 控制伴随（原数据缺失，用户最典型场景） =====
    ("Music-Play", "HomeAppliance-Control"): {
        "templates": [
            "{play}并调{dir}音量",
            "{play}然后把{ctrl}调{dir}",
            "{play}顺便{act}",
            "一边{play}一边{act}",
        ],
        "slots": {
            "play": ["播放音乐", "放歌", "来点音乐", "放首歌", "播放一首歌"],
            "ctrl": ["空调", "灯", "温度"],
            "act": ["把灯调暗", "开灯", "关灯", "调空调", "打开空调", "把窗帘拉上"],
            "dir": ["大", "高"],
        },
        "count": 40,
    },
    ("Video-Play", "HomeAppliance-Control"): {
        "templates": [
            "我{watch}帮我把{dev}调{dir}",
            "看{what}时把{dev}打开",
            "想看{what}顺便把{dev}打开",
        ],
        "slots": {
            "watch": ["想看球赛", "想看直播", "想看电影"],
            "what": ["球赛", "直播", "动画片"],
            "dev": ["灯", "空调", "窗帘"],
            "dir": ["暗些", "暗一点", "低一点"],
        },
        "count": 20,
    },
}


def _generate_from_spec(spec: dict, seen: set[str], rng: random.Random) -> list[str]:
    """按规格生成全部候选变体句（去重、排除已见）。截断由调用方做。"""
    cands: list[str] = []
    templates = spec["templates"]
    if spec.get("actions"):
        for tpl, time, action in itertools.product(templates, _TIMES, spec["actions"]):
            # 模板带"明天"时，时间必须是相对时间，不能是具体日期（避免"明天3号""明天明天"）
            if "明天" in tpl and any(ch in time for ch in ("明天", "号", "月")):
                continue
            s = tpl.format(time=time, action=action)
            if s not in seen:
                cands.append(s)
    else:
        slots = spec["slots"]
        keys = list(slots.keys())
        for combo in itertools.product(*[slots[k] for k in keys]):
            tpl = rng.choice(templates)
            s = tpl.format(**dict(zip(keys, combo)))
            if s not in seen:
                cands.append(s)
    rng.shuffle(cands)
    return cands


def augment(data_config: DataConfig | None = None, seed: int = 42) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """从原始单标签数据全量重建多标签集（真标注 + 固定 seed 变体），覆盖写 dataset_multi.csv。

    幂等：每次从 raw 重建，seed 固定 → 结果确定，重跑产生完全一致的文件。
    """
    cfg = data_config or DataConfig()
    rng = random.Random(seed)

    # 1. 重建真标注（build_multi_csv 覆盖写 12100 行：单标签 + 43 条真多标签）
    result = build_multi_csv(cfg)
    original_multi = [
        (t, f"{main},{hit.sub_intent}") for t, main, hit in result if hit is not None
    ]
    seen = {t for t, _, _ in result}

    # 2. 固定 seed 生成变体
    new: list[tuple[str, str]] = []
    for (main, sub), spec in _GENERATORS.items():
        all_cands = _generate_from_spec(spec, seen, rng)
        seen.update(all_cands)
        for s in all_cands[: spec["count"]]:
            new.append((s, f"{main},{sub}"))

    # 3. 全量写回（单标签 + 真多标签 + 变体）
    rows = [
        (t, main if hit is None else f"{main},{hit.sub_intent}")
        for t, main, hit in result
    ]
    with open(cfg.multi_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        for text, label in rows:
            writer.writerow([text, label])
        for text, label in new:
            writer.writerow([text, label])
    return original_multi, new


def coverage_report(original_multi: list[tuple[str, str]], new: list[tuple[str, str]], total_rows: int) -> None:
    """打印扩充前后的覆盖率与组合分布。"""
    def pairs(rows):
        return Counter(l for _, l in rows)

    all_multi = original_multi + new
    n = len(all_multi)
    print(f"总行数: {total_rows}")
    print(f"真标注多标签: {len(original_multi)} → 扩充后: {n} ({n / (total_rows + len(new)):.2%})")
    print(f"新增变体: {len(new)}")
    print("\n按(主,次)组合分布:")
    for (label, cnt) in pairs(all_multi).most_common():
        print(f"  {label}: {cnt}")


def main() -> None:
    cfg = DataConfig()
    with open(cfg.raw_csv, "r", encoding="utf-8") as f:
        total = sum(1 for _ in f)
    orig, new = augment(cfg)
    coverage_report(orig, new, total)
    print(f"\n已重建 → {cfg.multi_csv}（真标注 {len(orig)} + 变体 {len(new)}）")

    rng = random.Random(7)
    sample = rng.sample(new, min(15, len(new)))
    print("\n===== 变体随机抽样 15 条 =====")
    for text, label in sample:
        print(f"  [{label}] {text}")


if __name__ == "__main__":
    main()
