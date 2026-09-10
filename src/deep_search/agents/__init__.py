"""agents 层：三个 LLM 角色 = 图的节点。

planner / judge / synthesizer 共用同一个 DeepSeek（get_llm 懒加载）。
结构化输出绑定由各 agent 自己做：.with_structured_output(Model, method="function_calling")
—— DeepSeek 不支持 OpenAI 的 json_schema 结构化输出，必须用 function_calling。
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ..config import DeepSearchConfig

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

_llm: "ChatOpenAI | None" = None


def get_llm() -> "ChatOpenAI":
    """懒加载共用的 DeepSeek ChatOpenAI（planner / judge / synthesizer 共用）。

    首次调用才构建（对齐仓库「重资源懒加载」惯例，import 不碰 langchain）。
    注意 langchain-openai 1.6 的字段名：model_name / openai_api_key / openai_api_base
    （不是 model / api_key / base_url）。
    """
    global _llm
    if _llm is None:
        from langchain_openai import ChatOpenAI

        cfg = DeepSearchConfig()
        _llm = ChatOpenAI(
            model_name=cfg.llm_model,
            openai_api_key=cfg.llm_api_key,
            openai_api_base=cfg.llm_base_url,
            temperature=0,  # planner/judge 要稳定；synthesize 需要时可另行实例化
        )
    return _llm


from .planner import plan  # noqa: E402
from .judge import reflect  # noqa: E402
from .synthesizer import synthesize  # noqa: E402

__all__ = ["get_llm", "plan", "reflect", "synthesize"]
