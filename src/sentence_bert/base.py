from dataclasses import dataclass

from sentence_transformers import SentenceTransformer

from .config import SentenceBertConfig

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        config = SentenceBertConfig()
        _model = SentenceTransformer(config.sentence_dir, device=config.device)
    return _model


def similarity(text1: str, text2: str) -> float:
    """两个句子的余弦相似度，返回 0~1 标量。"""
    embeddings = get_model().encode([text1, text2], convert_to_tensor=True)
    return float(get_model().similarity(embeddings[:1], embeddings[1:])[0, 0])


@dataclass(frozen=True)
class MatchResult:
    """最相似候选的检索结果。"""

    text: str
    index: int
    score: float


def match(query: str, candidates: list[str]) -> MatchResult:
    """从候选文本中找出与 query 最相似的一句（1 句 vs N 句检索）。

    query 和候选库各编码一次，一个相似度矩阵算全部分数，argmax 取最高。
    """
    if not candidates:
        raise ValueError("candidates 不能为空")
    model = get_model()
    q_emb = model.encode([query], convert_to_tensor=True)      # [1, dim]
    c_emb = model.encode(candidates, convert_to_tensor=True)   # [N, dim]
    scores = model.similarity(q_emb, c_emb)[0]                 # [N]
    best = int(scores.argmax())
    return MatchResult(text=candidates[best], index=best, score=float(scores[best]))