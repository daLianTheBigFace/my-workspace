"""output：落盘报告与结构化结果。

流结束后，把 Deep Research 的交付物写到 outputs/deep_search/<时间戳>-<slug>/：
- report.md：给人看（答案正文 + [n] 引用源列表 + 检索过程）
- result.json：给机器看（全量结构化 trace，对接攒 eval 集）
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from .config import DeepSearchConfig
from .schemas import Source, Stats, SubQuestion

logger = logging.getLogger(__name__)


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def slugify(text: str, max_len: int = 40) -> str:
    """把问题转成安全目录名片段：去非法字符、空白转 -、截断。"""
    text = re.sub(r"[^\w一-鿿]+", "-", text.strip())
    text = re.sub(r"-+", "-", text).strip("-")
    return text[:max_len] or "query"


def _dump(v):
    return v.model_dump() if isinstance(v, BaseModel) else v


def _build_report(
    question: str,
    answer: str,
    sources: list[Source],
    sub_questions: list[SubQuestion],
    stop_reason: str,
    trace: list[dict],
) -> str:
    search_queries: list[str] = []
    read_ok: list[str] = []
    read_fail: list[str] = []
    rounds: set[int] = set()
    seen_q: set[str] = set()
    for ev in trace:
        if ev.get("event") == "search":
            q = ev.get("query", "")
            if q not in seen_q:  # web+local 对同一 query 各发一次事件，去重
                seen_q.add(q)
                search_queries.append(q)
        elif ev.get("event") == "read":
            (read_ok if ev.get("status") == "ok" else read_fail).append(ev.get("url", ""))
        if isinstance(ev.get("round"), int):
            rounds.add(ev["round"])

    lines = ["# 深度搜索报告", ""]
    lines += [f"**问题**：{question}", ""]
    lines += ["## 答案", "", answer or "（无）", ""]
    lines += ["## 来源", ""]
    if sources:
        lines += [f"{i}. [{s.title}]({s.url})" for i, s in enumerate(sources, 1)]
    else:
        lines += ["（无）"]
    lines += ["", "## 检索过程", ""]
    sq = "；".join(s.question for s in sub_questions) or "（无）"
    lines += [f"- **拆解**：{sq}"]
    lines += [f"- **轮数**：{max(rounds) if rounds else 0}（停止原因：{stop_reason or '—'}）"]
    lines += [f"- **搜索 query（{len(search_queries)}）**：{'；'.join(search_queries) or '（无）'}"]
    lines += [f"- **成功阅读 {len(read_ok)} 篇**："]
    lines += [f"  - {u}" for u in read_ok]
    if read_fail:
        lines += [f"- **抓取失败 {len(read_fail)} 篇**：{'；'.join(read_fail)}"]
    return "\n".join(lines) + "\n"


def write_outputs(
    *,
    question: str,
    answer: str,
    sources: list[Source],
    stats: Stats,
    sub_questions: list[SubQuestion],
    stop_reason: str,
    trace: list[dict],
) -> tuple[str, str]:
    """落盘 report.md + result.json，返回 (report_path, result_path)。"""
    cfg = DeepSearchConfig()
    out_dir = Path(cfg.output_dir) / f"{_ts()}-{slugify(question)}"
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "question": question,
        "answer": answer,
        "sources": [_dump(s) for s in sources],
        "stats": stats.model_dump(),
        "sub_questions": [_dump(s) for s in sub_questions],
        "stop_reason": stop_reason,
        "trace": trace,
    }
    result_path = out_dir / "result.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report_path = out_dir / "report.md"
    report_path.write_text(
        _build_report(question, answer, sources, sub_questions, stop_reason, trace),
        encoding="utf-8",
    )

    logger.info("已落盘：%s / %s", report_path, result_path)
    return str(report_path), str(result_path)
