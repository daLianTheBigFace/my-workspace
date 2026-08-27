"""构造 ES 查询体（纯函数，不连 ES，方便单测）。

三个能力对应三种查询：
- 全文检索：match（text 字段，IK 中文分词）
- 条件过滤：bool(must=match) + filter(range on page)
- 向量检索：knn（text_vector 字段，HNSW 近似检索）

全文检索支持 match_type 三档：
- match:   分词后任一命中（默认，宽松）
- phrase:  match_phrase，分词后按顺序连续出现（精确）
- fuzzy:   match + fuzziness AUTO，容忍错别字/编辑距离
"""
from __future__ import annotations

MATCH_TYPES = ("match", "phrase", "fuzzy")


def _match_clause(query: str, match_type: str = "match") -> dict:
    """按匹配模式构造 text 字段的查询子句。"""
    if match_type == "phrase":
        return {"match_phrase": {"text": query}}
    if match_type == "fuzzy":
        # 用显式 fuzziness=1 而非 AUTO：中文词多为 2 字符，AUTO 对 2 字符只允许
        # 0 次编辑（等于无容错），fuzziness=1 才能容忍「座椅/坐椅」这类错别字
        return {"match": {"text": {"query": query, "fuzziness": 1}}}
    if match_type == "match":
        return {"match": {"text": query}}
    raise ValueError(f"未知 match_type：{match_type}，可选 {MATCH_TYPES}")


def full_text_query(query: str, size: int = 10, match_type: str = "match") -> dict:
    """全文检索：IK 分词后的 match 查询。"""
    return {"query": _match_clause(query, match_type), "size": size}


def filter_query(
    query: str,
    page_from: int | None = None,
    page_to: int | None = None,
    size: int = 10,
    match_type: str = "match",
) -> dict:
    """条件过滤：全文 match + 页码 range 过滤（可只给一端或两端）。"""
    if page_from is None and page_to is None:
        return full_text_query(query, size, match_type)
    filters = []
    if page_from is not None:
        filters.append({"range": {"page": {"gte": page_from}}})
    if page_to is not None:
        filters.append({"range": {"page": {"lte": page_to}}})
    return {
        "query": {
            "bool": {
                "must": _match_clause(query, match_type),
                "filter": filters,
            }
        },
        "size": size,
    }


def vector_query(
    query_vector: list[float],
    size: int = 10,
    num_candidates: int = 100,
) -> dict:
    """向量检索：bge 编码后的 query 向量做 knn 近邻检索。"""
    return {
        "query": {
            "knn": {
                "field": "text_vector",
                "query_vector": query_vector,
                "k": size,
                "num_candidates": num_candidates,
            }
        },
        "size": size,
    }
