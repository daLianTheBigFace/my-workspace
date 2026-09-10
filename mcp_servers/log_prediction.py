"""PostToolUse hook：把 predict_intent 的每次调用落盘 CSV。

由 .claude/settings.local.json 的 hooks.PostToolUse 触发，Claude Code 在
mcp__intent__predict_intent 成功后把 JSON 灌进 stdin：

    {
      "tool_name": "mcp__intent__predict_intent",
      "tool_input":   {"text": "...", "model": "bert", "top_k": 3},
      "tool_response": {"intent": "...", "confidence": ..., "top3": [...]}
    }

输出追加到 logs/intent_predictions.csv，慢慢攒成评测集。
"""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

LOG_FILE = Path(__file__).resolve().parents[1] / "logs" / "intent_predictions.csv"
FIELDS = ["ts", "text", "model", "main_intent", "confidence", "top3"]


def _fmt_top3(entries: list[dict]) -> str:
    parts: list[str] = []
    for e in entries:
        p = e.get("probability")
        prob = f"{p:.4f}" if isinstance(p, (int, float)) else str(p)
        parts.append(f"{e.get('intent')}:{prob}")
    return " | ".join(parts)


def main() -> None:
    # 显式按 UTF-8 读字节再解码，规避 Windows 上 sys.stdin 文本层用 ANSI/UTF-8
    # + surrogateescape 解码导致的 \udcXX 代理字符问题
    raw = sys.stdin.buffer.read().decode("utf-8")
    if not raw.strip():
        return
    data = json.loads(raw)

    tool_input = data.get("tool_input") or {}
    resp = data.get("tool_response") or {}

    text = tool_input.get("text") or ""
    model = tool_input.get("model") or ""
    main_intent = resp.get("intent") or ""
    confidence = resp.get("confidence")
    top3 = _fmt_top3(resp.get("top3") or [])

    # 报错（模型未加载等）时 resp 无 top3，退化成记一条 error 行
    if not top3 and resp.get("error"):
        main_intent = "ERROR"
        confidence = resp["error"]

    row = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "text": text,
        "model": model,
        "main_intent": main_intent,
        "confidence": confidence,
        "top3": top3,
    }

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    new_file = not LOG_FILE.exists()
    with LOG_FILE.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    main()
