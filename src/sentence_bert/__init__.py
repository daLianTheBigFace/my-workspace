"""sentence-BERT 句向量相似度模块（独立包，与 intent_recognition 平级）。

对外只暴露相似度与检索接口，实现细节在 base.py。
"""
from .base import MatchResult, get_model, match, similarity

__all__ = ["MatchResult", "get_model", "match", "similarity"]
