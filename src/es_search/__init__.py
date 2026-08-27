"""ES 检索包：基于本地 Elasticsearch 的全文检索 / 条件过滤 / 向量检索。

三大能力（search.py）：
- full_text(query, size)   全文检索（IK 中文分词 match）
- filtered(query, ...)     条件过滤（全文 + 页码 range）
- vector(query, size)      向量检索（bge 编码 + knn）

构建索引：python -m es_search.build_index
挂到主服务：from es_search.api import router, warm
"""
from .search import EsHit, filtered, full_text, vector

__all__ = ["EsHit", "full_text", "filtered", "vector"]
