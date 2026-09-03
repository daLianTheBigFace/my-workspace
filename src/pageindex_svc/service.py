"""PageIndex 文档检索的核心操作。

把本地引擎对外的动作收敛到这：建索引 / 列文档 / 取树 / 检索问答。
页面上不直接碰 pageindex 客户端，全部经 `get_client()` 惰性拉起。
"""
from __future__ import annotations

from typing import Any

from .client import get_client

# ---- 索引管理 ----


def submit_pdf(pdf_path: str, mode: str = "flash", metadata: dict | None = None) -> dict[str, Any]:
    """提交 PDF 建索引（本地同步）。flash 默认；mode="standard" 走全 LLM 建树。"""
    return get_client().submit_document(pdf_path, mode=mode, metadata=metadata)


def list_docs(limit: int = 50, offset: int = 0) -> dict[str, Any]:
    return get_client().list_documents(limit=limit, offset=offset)


def get_doc(doc_id: str) -> dict[str, Any]:
    return get_client().get_document(doc_id)


def get_tree(doc_id: str, node_summary: bool = False, include_text: bool = False) -> dict[str, Any]:
    return get_client().get_tree(doc_id, node_summary=node_summary, include_text=include_text)


def delete_doc(doc_id: str) -> dict[str, Any]:
    return get_client().delete_document(doc_id)


# ---- 检索 + 问答 ----


def _cap(text: str, limit: int) -> str:
    """过程回显裁剪，避免把整页文本/超长思考塞爆响应。"""
    if len(text) <= limit:
        return text
    return f"{text[:limit]} …(已截断，原文 {len(text)} 字)"


def _fmt_step(ev: dict[str, Any]) -> dict[str, Any]:
    """把 PageIndex 推理过程的单个事件压成可回显的小 dict。"""
    typ = ev.get("type")
    if typ == "thinking":
        return {"type": "thinking", "text": _cap(ev.get("delta", ""), 300)}
    if typ == "tool_call":
        return {
            "type": "tool_call",
            "name": ev.get("name"),
            "arguments": _cap(str(ev.get("arguments", "")), 500),
        }
    if typ == "tool_result":
        out = ev.get("output", "")
        return {
            "type": "tool_result",
            "name": ev.get("name"),
            "output_len": len(out),
            "output": _cap(str(out), 600),
        }
    return {"type": str(typ)}


def query(doc_id: str, text: str, with_process: bool = False) -> tuple[str, list[dict[str, Any]]]:
    """对某个已建索引文档做推理式检索 + 问答。

    走 chat 的 answer lane（own-model，DeepSeek），stream 起来消费 events：
    既能拼出最终回答，又能（with_process=True 时）拿到模型检索命中的
    tool_call / tool_result 作为过程与引用。
    """
    client = get_client()
    stream = client.chat(text, doc_id=doc_id, stream=True, show_process=False)
    answer: list[str] = []
    steps: list[dict[str, Any]] = []
    try:
        for ev in stream.events:
            typ = ev.get("type")
            if typ == "answer":
                answer.append(ev.get("delta", ""))
            elif with_process:
                steps.append(_fmt_step(ev))
    finally:
        stream.close()
    return "".join(answer), steps
