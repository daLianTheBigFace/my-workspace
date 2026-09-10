---
name: add-api-package
description: 往主服务接入一个新的能力包（新检索/推理/处理能力）。当需要新增一个 src/<name> 顶层包、并把它挂到 intent_recognition.api.app 的某个 /<prefix> 路由下时使用。触发词：新增能力、加个接口、接入新包、扩展服务、加一个检索/推理模块。
---

# 新增能力包

把一个全新的能力接进主服务（入口 `src/intent_recognition/api/app.py:app`）的标准套路。

项目里已有的 5 个能力包全部长同一个骨架，照它们抄即可：
`sentence_bert`（句子相似度）、`rag_retrieval`（RAG）、`es_search`（ES 检索）、`pageindex_svc`（文档检索）、`intent_recognition`（主包本体）。

## 约定总览（先看这个）

每个能力包 = `src/<name>/` 一个顶层包，内部结构：

```
src/<name>/
├── __init__.py          # 空或导出公共 API
├── config.py            # 路径/超参（可选，有配置就放这，参考 EsConfig / PageIndexConfig）
├── <业务模块>.py         # 纯逻辑（如 search.py / service.py），不依赖 FastAPI
└── api/
    ├── __init__.py      # from .app import router, warm  再导出  ← 主服务只认这个
    └── app.py           # router = APIRouter(...) + 各接口 + def warm()
```

主服务 `intent_recognition/api/app.py` 是**唯一入口**：`include_router` 各包 router，`lifespan` 里调各包 `warm()`。

核心约定一句话：**`api/app.py` 必须暴露 `router` 和 `warm` 两个名字，`warm()` 只告警不抛异常。**

## 步骤

### 1. 建包骨架

在 `src/` 下建目录，命名用 `snake_case`（`es_search`、`pageindex_svc` 这种）。别用会跟 pip/git 依赖撞名的顶层名（教训：`pageindex` 撞了 PyPI 包，改叫 `pageindex_svc`）。

### 2. 写 `api/app.py`（router + warm）

照 `src/es_search/api/app.py` 抄。模板（`<...>` 是占位符，替换成你的实际名字；其余是可抄的真代码）：

```python
"""<能力名> API 路由（挂 /<prefix> 前缀）。

主服务 app.py 里 include_router 一行即可接入：
    from <name>.api import router as <name>_router, warm as warm_<name>
    app.include_router(<name>_router)

接口：
- ...
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from ..config import <Name>Config
from .. import <业务模块>

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/<prefix>", tags=["<tag>"])


# ---- Pydantic Schemas ----（请求/响应体，参考其它包；query 用 field_validator 去空白）

# ---- Endpoints ----（@router.post / @router.get；依赖没就绪时抛 HTTPException 带提示，参考 es 的 _require_es、pageindex 的 _require_llm）


def warm() -> None:
    """启动预热：只告警，绝不抛异常（否则会拖垮主服务启动）。"""
    if <依赖没就绪>:
        logger.warning("<name>：<缺什么>，/<prefix> 暂不可用")
```

要点：
- `prefix` 用一个短的路径段（如 `/es`、`/pageindex`、`/rag`），tags 用包名或领域名。
- 依赖（ES / LLM key / 索引 / 模型）没就绪时，**接口里抛 `HTTPException`（503/502）带中文提示**，而不是让异常裸奔；参考 `es_search` 的 `_require_es()`、`pageindex_svc` 的 `_require_llm()`。
- `warm()` 内部自己 try/except 或只判断后 `logger.warning`，**不要 raise**。

### 3. 写 `api/__init__.py`

```python
"""API 层：对外暴露 router 与 warm。"""
from .app import router, warm

__all__ = ["router", "warm"]
```

主服务只 import 这两个名字，少了哪个都会挂。

### 4. 接入主服务 `app.py`

在 `src/intent_recognition/api/app.py` 里做 3 件事：

```python
# 顶部 import（跟现有 3 个包并排）
from <name>.api import router as <name>_router
from <name>.api import warm as warm_<name>

# lifespan 里加预热
try:
    warm_<name>()
except Exception:
    logger.exception("<name> 预热失败")

# 底部挂路由
app.include_router(<name>_router)
```

`lifespan` 里 `warm` 的写法分两档，看你的 `warm()` 会不会 raise：
- `warm()` 已保证不抛异常（只 warning）→ 直接 `warm_<name>()`（参考 `warm_es()`）。
- `warm()` 可能加载模型/索引会抛 → 用 `try/except` 包一层（参考 `warm_rag()`、`warm_pageindex()`）。

### 5. 登记 `pyproject.toml`

把新包加进 `[tool.hatch.build.targets.wheel].packages`，否则打包时它被漏掉：

```toml
packages = [
    "src/intent_recognition",
    "src/sentence_bert",
    "src/rag_retrieval",
    "src/es_search",
    "src/pageindex_svc",
    "src/<name>",          # ← 加这行
]
```

### 6. 加依赖

新包用到的库写进 `pyproject.toml` 的 `[project].dependencies`，然后 `uv sync`。

### 7. 写测试

建 `tests/test_<name>.py`。纯逻辑（查询构造、业务模块）走**不加载真实模型/不连外部服务**的快路径；需要真实模型的用例打 `@pytest.mark.slow` 标记（见 `pyproject.toml` 的 markers）。

跑：
```bash
uv run pytest -m "not slow"    # 快路径
uv run pytest                  # 全量（含真实模型/ES 冒烟）
```

## 坑（务必）

- **`warm()` 不许抛异常**：ES 没起 / 没配 key / 索引没建都是正常状态，只能 `logger.warning`。否则一个缺依赖就拖垮整个服务启动。
- **用 `lifespan`，别用 `on_event` / `add_event_handler`**：仓库 fastapi 0.141 / starlette 1.6 已移除这些，预热一律放 lifespan（跟主包 `app.py` 一致）。
- **缓存/索引落 E 盘**（用户 C 盘敏感）：`config.py` 里路径写 `E:\...`，并确认 `.gitignore` 覆盖。
- **顶层包名别撞 pip/git 依赖**：起名前先想清楚会不会跟已装库重名（`pageindex` 的教训）。
- **可独立起服务的才加 `serve.py`**：多数包只并入主服务；需要单独部署的参考 `pageindex_svc/serve.py`。
- **响应体用 `response_model`**：本仓库所有接口都显式声明 Pydantic 响应体，保持 Swagger 文档一致。
