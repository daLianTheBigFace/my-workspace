"""ES 检索模块测试：查询构造（快）+ 真实 ES 冒烟（慢）。

快路径不连 ES，只验 query body 纯函数；慢路径需要本地 ES 运行且已 build_index。
"""
from __future__ import annotations

import pytest

from es_search import queries


# ---- 快路径：查询体构造纯函数 ----
def test_full_text_query_body():
    body = queries.full_text_query("空调怎么开", size=5)
    assert body["query"] == {"match": {"text": "空调怎么开"}}
    assert body["size"] == 5


def test_phrase_query_body():
    """精确短语：match_phrase，分词后按顺序连续出现。"""
    body = queries.full_text_query("空调怎么开", size=5, match_type="phrase")
    assert body["query"] == {"match_phrase": {"text": "空调怎么开"}}


def test_fuzzy_query_body():
    """容错模糊：match + fuzziness 1，容忍错别字（座椅/坐椅 这类 2 字符词）。"""
    body = queries.full_text_query("空调怎开", size=5, match_type="fuzzy")
    assert body["query"] == {
        "match": {"text": {"query": "空调怎开", "fuzziness": 1}}
    }


def test_unknown_match_type_raises():
    import pytest

    with pytest.raises(ValueError):
        queries.full_text_query("空调", size=5, match_type="regex")


def test_filter_query_with_range():
    body = queries.filter_query("座椅", page_from=110, page_to=130, size=5)
    q = body["query"]
    assert q["bool"]["must"] == {"match": {"text": "座椅"}}
    assert q["bool"]["filter"] == [
        {"range": {"page": {"gte": 110}}},
        {"range": {"page": {"lte": 130}}},
    ]
    assert body["size"] == 5


def test_filter_query_only_upper_bound():
    body = queries.filter_query("空调", page_to=250, size=3)
    assert body["query"]["bool"]["filter"] == [{"range": {"page": {"lte": 250}}}]


def test_filter_query_no_bounds_falls_back_to_full_text():
    """不传页码范围时等价全文检索。"""
    assert queries.filter_query("空调", size=3) == queries.full_text_query("空调", 3)


def test_vector_query_body():
    body = queries.vector_query([0.1, 0.2, 0.3], size=5, num_candidates=100)
    knn = body["query"]["knn"]
    assert knn["field"] == "text_vector"
    assert knn["query_vector"] == [0.1, 0.2, 0.3]
    assert knn["k"] == 5
    assert knn["num_candidates"] == 100


# ---- 慢路径：真实 ES（需本地 9200 已启动 + 已 build_index） ----
@pytest.mark.slow
def test_es_full_text_smoke():
    from es_search import full_text

    hits, total = full_text("空调", size=3)
    assert hits, "应返回空调相关 chunk"
    assert total >= len(hits)


@pytest.mark.slow
def test_es_filter_smoke():
    from es_search import filtered

    hits, total = filtered("座椅", page_from=110, page_to=130, size=3)
    assert hits, "110-130 页内应有座椅相关内容"
    assert all(110 <= h.page <= 130 for h in hits)
    assert total >= len(hits)


@pytest.mark.slow
def test_es_vector_smoke():
    from es_search import vector

    hits, total = vector("怎么开空调", size=3)
    assert hits, "向量检索应返回语义相关 chunk"
    # 语义检索一般能命中空调章节（240-250 页附近），但只宽松断言有结果
    assert total >= len(hits)
