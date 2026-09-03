"""pageindex_svc 包装包的快速测试。

快路径不建客户端（litellm 导入较重）、不调 DeepSeek；真实建索引/检索用 slow 标记。
"""
from __future__ import annotations

import os

import pytest

from pageindex_svc.api import router
from pageindex_svc.api.app import health
from pageindex_svc.config import PROJECT_ROOT, PageIndexConfig

# 期望暴露的路由（router 只含 /pageindex 前缀）
_EXPECTED_PATHS = {
    "/pageindex/documents",
    "/pageindex/documents/{doc_id}",
    "/pageindex/documents/{doc_id}/tree",
    "/pageindex/health",
    "/pageindex/query",
    "/pageindex/submit",
}


def test_config_storage_and_model():
    cfg = PageIndexConfig()
    # 索引一定落 E 盘（项目在 E:\project\my-workspace 下）
    assert cfg.storage_path == str(PROJECT_ROOT / "data" / "pageindex")
    assert cfg.model == "deepseek/deepseek-chat"
    assert isinstance(cfg.api_key_set, bool)


def test_default_pdf_exists():
    # assets/Week06 是 git 忽略的大文件，机器上没有就跳过，不算失败
    cfg = PageIndexConfig()
    if not os.path.isfile(cfg.default_pdf):
        pytest.skip(f"缺少默认 PDF：{cfg.default_pdf}")


def test_router_exposes_expected_paths():
    paths = {route.path for route in router.routes}
    assert _EXPECTED_PATHS <= paths


def test_health_reports_state():
    out = health()
    assert out.model == "deepseek/deepseek-chat"
    assert out.storage_path.endswith("data\\pageindex") or out.storage_path.endswith("data/pageindex")
    assert isinstance(out.api_key_set, bool)


@pytest.mark.slow
def test_local_client_list_documents(tmp_path):
    """慢：构建本地客户端（导入 litellm，数秒）列已建索引，不调 LLM。"""
    from pageindex_svc.client import get_client

    client = get_client()
    assert client.__class__.__name__ == "PageIndexLocalClient"
    resp = client.list_documents(limit=5)
    assert "documents" in resp and "total" in resp
