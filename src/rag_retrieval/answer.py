"""问答：把重排后的 top 页拼成 prompt，请求大模型生成回答。

照搬 week06 RAG101_08 的 prompt 模板与「无法回答」兜底；但请求大模型改用
openai 官方 SDK（OpenAI 兼容接口，DeepSeek / Ollama / GLM 等都能用）。
week06 里的 GLM/GPT key 是课程遗留，一律不碰。

三步走，接口层可拿到「发给模型的 prompt」：
- build_prompt(query, reference_pages) -> 拼好的完整 prompt
- call_llm(prompt) -> 把 prompt 发给大模型（非流式），返回回答
- generate_answer(...) = build_prompt + call_llm 一步到位

配置（见 config.RagConfig）：优先 RAG_LLM_BASE_URL / RAG_LLM_API_KEY /
RAG_LLM_MODEL，未配置时回落 DeepSeek 默认（base_url=https://api.deepseek.com，
model=deepseek-chat，key 兼容 DEEPSEEK_API_KEY）。只有拿到 key 才启用问答。
"""
from __future__ import annotations

import logging

from openai import OpenAI

from .config import RagConfig

logger = logging.getLogger(__name__)

# 照搬 week06 RAG101_08 的 prompt 模板（占位符改成命名式便于复用）
_PROMPT_TEMPLATE = (
    "你是一个汽车专家，你擅长编写和回答汽车相关的用户提问，帮我结合给定的资料，回答下面的问题。\n"
    "如果问题无法从资料中获得，或无法从资料中进行回答，请回答无法回答。如果提问不符合逻辑，请回答无法回答。\n"
    "如果问题可以从资料中获得，则请逐步回答。\n\n"
    "资料：{reference}\n\n"
    "问题：{query}"
)


def build_reference(reference_pages: list[tuple[int, str]]) -> str:
    """把 (页码, 整页文本) 列表拼成 prompt 里的"资料"段落。"""
    return "\n".join(
        text.replace("\n", " ") + f"\t上述内容在第{page}页"
        for page, text in reference_pages
    )


def build_prompt(query: str, reference_pages: list[tuple[int, str]]) -> str:
    """拼出发给大模型的完整 prompt（资料段 + 问题）。"""
    return _PROMPT_TEMPLATE.format(
        reference=build_reference(reference_pages), query=query
    )


# 模型按 prompt 指示说的"拒绝话术"才是真拒绝；正常回答里出现的"无法"不算。
# 兜底条件：命中拒绝话术 且 回答很短（说明是裸拒绝，没有实质内容）。
_REFUSAL_MARKERS = (
    "无法回答",
    "不能回答",
    "无法从资料",
    "无法从给定",
    "无法从提供的",
    "无法为您",
)
_REFUSAL_MAX_LEN = 40


def _is_bare_refusal(answer: str) -> bool:
    """是否裸拒绝：命中拒绝话术 且 足够短（没有实质内容可保留）。"""
    return len(answer) < _REFUSAL_MAX_LEN and any(
        m in answer for m in _REFUSAL_MARKERS
    )


def call_llm(prompt: str) -> str:
    """把 prompt 发给用户自己的大模型接口（非流式），返回回答文本。"""
    cfg = RagConfig()
    if not cfg.answer_enabled:
        raise RuntimeError(
            "问答未启用：请设置 RAG_LLM_API_KEY 或 DEEPSEEK_API_KEY"
        )

    client = OpenAI(api_key=cfg.llm_api_key, base_url=cfg.llm_base_url, timeout=60)
    resp = client.chat.completions.create(
        model=cfg.llm_model,
        messages=[{"role": "user", "content": prompt}],
        stream=False,  # 先不做流式，等接口稳定再开
    )
    answer = (resp.choices[0].message.content or "").strip()
    if _is_bare_refusal(answer):
        answer = "结合给定的资料，无法回答问题。"
    return answer


def generate_answer(query: str, reference_pages: list[tuple[int, str]]) -> str:
    """一步到位：拼 prompt 并请求大模型，返回回答。"""
    return call_llm(build_prompt(query, reference_pages))
