# deep_search — 深度搜索服务

一个「Deep Research」智能体：把用户问题拆解 → 联网（Tavily）+ 本地手册**双 tool** 检索 → 抓取阅读 → 判官把关/补搜 → 迭代 → 合成带引用的答案，全程 **SSE 流式输出**，结束后落盘报告与结构化结果。

> 状态：**已落地**（2026-09-10），`/deep_search/search` SSE 流式接口已接入主服务。本文档描述最终实现。

## 定位

和现有 5 个能力包不同，deep_search 不是「一次检索出结果」，而是**多步自主**的搜索编排层：

- 复用现有本地检索包（`rag_retrieval` / `es_search` / `pageindex_svc`）作为**本地 tool**，问题涉及车机手册时自动查本地；
- 联网检索走 **Tavily**，覆盖手册外的开放问题；
- 对外只暴露 SSE 流式接口，让前端实时看到「思考 → 检索 → 阅读 → 回答」的全过程。

## 架构：LangGraph StateGraph

编排用 **LangGraph** 实现：5 个节点 + 1 条条件回边。节点分两类——**LLM 角色**（`agents/`，共用 DeepSeek）与**非 LLM 工具**（`tools/`）。

```
         START
           │
        [plan]          planner（LLM）：去废话 → 拆搜索 query
           │
        [search]        web_search + local_search（并行）
           │
        [read]          reader 抓取 + judge.filter 质量过滤
           │
        [reflect]       judge.check（LLM）：够不够 + 补搜 query
           │
        ┌─ 不够 & round<max ──┐   ← 条件回边
        │                     │
        ▼ 够了                │
   [synthesize]（LLM）         │
        │                     │
       END   ◄────────────────┘
```

| 节点 | 文件 | 职责 | LLM? |
|---|---|---|---|
| `plan` | `agents/planner.py` | 去废话 → 拆 N 个搜索 query（关键词提取） | ✅ |
| `search` | `tools/web_search.py` + `tools/local_search.py` | Tavily 联网 + 复用 `rag_retrieval.pipeline` | ❌ |
| `read` | `tools/reader.py` | 抓网页 → trafilatura 抽正文 → 判官质量过滤 | ❌（过滤调 LLM） |
| `reflect` | `agents/judge.py` | 充分性判断：够不够 → 补搜 query | ✅ |
| `synthesize` | `agents/synthesizer.py` | 带引用答案 + 检索过程段落 | ✅ |

**共享状态** `state.py`（AgentState）贯穿各节点：`question / max_rounds / sub_questions / round / gap_queries / verdict / stop_reason / rationale / pending_hits / notes / seen_queries / seen_urls / sources / answer`（其中 notes / seen_queries / seen_urls / sources 跨轮累加，其余每轮替换）。

## 判官设计（两段式）

判官有两件事，拆成两个 prompt，都用 DeepSeek：

| 段 | 回答的问题 | 输入 | 时机 | 效果 |
|---|---|---|---|---|
| **质量过滤** `judge.filter()` | 这篇值不值得信、有没有营养 | 单篇抽取正文 | read 节点内，逐篇 | 不相关的**不进 notes**，防止污染后续 |
| **充分性判断** `judge.check()` | 攒的证据够不够回答原问题 | 全部 notes 精华 | reflect 节点 | 不够则输出**补搜 query** |

要点：

- `judge.check()` 不能只判「行/不行」，必须同时输出 `gap_queries`（可搜索的 query 串）。
- prompt 里要喂 `seen`（已搜过的 query 集合），避免补搜只是换个说法兜圈。
- **收敛三保险**：① `max_rounds` 硬停；② 补搜 query 全在 `seen` 或新结果全重复 → 停；③ 判官自判「够了」→ 停。

## 结果输出（三层）

```
① 流式答案（answer 事件）  实时看，delta 打字机效果
② done 概览（done 事件）   完整答案 + 引用源列表 + 统计（轮数/搜索数/耗时）
③ 落盘文件（流结束后写）   可下载、可复盘、可复用
```

### 落盘文件（默认生成，`save_files` 可关）

每次搜索跑完，在流结束时写一个目录：

```
outputs/deep_search/<时间戳>-<问题slug>/
├── report.md      # 给人看：答案正文 + [n]引用 → 源列表 + 检索过程
└── result.json    # 给机器看：全量结构化 trace（question/plan/每步事件/answer/sources/stats）
```

- `report.md` **含「检索过程」一节**（拆了什么、搜了什么、读了哪些源、判了几轮）——这是 Deep Research 区别于普通问答的交付物。
- `result.json` 对接仓库 `logs/` 攒评测集的思路，用于复盘调试 / eval / 前端离线重建。
- 两个路径在 `done` 事件里返回（`report_path` / `result_path`），前端可做「下载报告」按钮。
- v1 **不做**来源快照缓存（体积大、版权/时效问题，对回答无直接贡献，留待复现检索过程时再补）。

## 目录结构

```
src/deep_search/
├── README.md            # 本文档
├── __init__.py
├── config.py            # DeepSearchConfig：Tavily key / LLM / 预算 / 输出路径
├── state.py             # AgentState：LangGraph 共享状态
├── graph.py             # StateGraph 构建：5 节点 + 边 + 条件回边（编排核心）
│
├── agents/              # 三个 LLM 角色 = 图的节点
│   ├── __init__.py
│   ├── planner.py       # plan 节点：去废话 → 拆搜索 query
│   ├── judge.py         # read/reflect 节点：质量过滤 + 充分性判断
│   └── synthesizer.py   # synthesize 节点：带引用答案 + 检索过程
│
├── tools/               # 三个非 LLM 工具
│   ├── __init__.py
│   ├── web_search.py    # Tavily 联网
│   ├── local_search.py  # 复用 rag_retrieval.pipeline
│   └── reader.py        # 抓网页 → trafilatura 抽正文
│
├── schemas.py           # SSE 事件 + 结构化输出（LLM 返回的 Pydantic 模型）
├── output.py            # 落盘：report.md + result.json
└── api/
    ├── __init__.py      # from .app import router, warm
    └── app.py           # POST /deep_search/search（SSE）+ GET /deep_search/health + warm()
```

## 调用关系

```
api/app.py  POST /deep_search/search
  └─► graph.astream_events(initial_state)      # LangGraph 逐事件流
        ├─ on_chain_start / on_tool_start ...  → 映射成 plan/search/read/reflect 事件
        ├─ on_chat_model_stream（LLM token）   → answer 事件（delta）
        └─ 图跑完 → 取最终 state               → done 事件
  └─► output.write(report.md, result.json)     → done 事件带 report_path/result_path
```

`graph.py` 里用 `StateGraph(AgentState)` 建图、`add_node` 加 5 个节点、`add_conditional_edges` 在 reflect 处接回边，`compile()` 后交给 `api/app.py` 调 `astream_events()`。**图负责编排，SSE 是外层传输协议，两者解耦。**

## SSE 事件 schema

`POST /deep_search/search`，请求体 `{"question": "...", ...}`，响应 `text/event-stream`。

| event | data | 说明 |
|---|---|---|
| `plan` | `{round, sub_questions:[{id,question}]}` | 拆解 / 大纲 |
| `search` | `{round, query, source:"web"\|"local", results:[{title,url,snippet}]}` | 一次检索 |
| `read` | `{round, url, status:"ok"\|"fail", chars}` | 抓取一篇 |
| `reflect` | `{round, verdict:"continue"\|"enough", gap_queries:[...], rationale}` | 判官判断 |
| `answer` | `{delta}` | 答案流式增量 |
| `done` | `{answer, sources:[{title,url}], report_path, result_path, stats:{rounds,searches,sources,elapsed_ms}}` | 收尾 |
| `error` | `{message}` | 出错 |

> **前端消费**：SSE 的 `EventSource` 只支持 GET，本服务用 POST 传 body，故前端用 `fetch` + `ReadableStream` 读流、手动拆 SSE 帧。

## 配置（`config.py`）

沿用仓库 config 惯例（`PROJECT_ROOT = parents[2]`、`load_dotenv`、路径钉 E 盘）：

| 配置项 | 默认 | 来源 | 说明 |
|---|---|---|---|
| `tavily_api_key` | — | 环境变量 `TAVILY_API_KEY` | 必填，否则 `/deep_search/search` 503 |
| `llm_base_url` | `https://api.deepseek.com` | `RAG_LLM_BASE_URL` 回落 | 复用现有 DeepSeek |
| `llm_api_key` | — | `RAG_LLM_API_KEY` 回落 `DEEPSEEK_API_KEY` | 复用现有 key |
| `llm_model` | `deepseek-chat` | `RAG_LLM_MODEL` 回落 | planner/judge/synthesizer 共用 |
| `max_rounds` | `3` | 代码默认，可请求体覆盖 | 迭代轮数上限 |
| `max_sources` | `12` | 同上 | 总抓取页面数上限 |
| `results_per_query` | `5` | 同上 | Tavily 每 query 返回条数 |
| `search_concurrency` | `3` | 同上 | 子问题并行检索度 |
| `fetch_timeout` | `10` | 同上 | 单页抓取超时（秒） |
| `enable_local` | `true` | 同上 | 是否启用本地手册 tool |
| `local_rounds` | `{1}` | 同上 | 哪些轮次跑本地（默认仅 round 1） |
| `save_files` | `true` | 请求体可关 | 是否落盘 report.md + result.json |
| `output_dir` | `outputs/deep_search` | 代码默认 | 落盘目录（E 盘，git-ignored） |

## 新增依赖

已加入 `pyproject.toml` 的 `[project].dependencies`：

- `langgraph` — StateGraph 编排（节点 + 条件回边 + 状态）
- `langchain-core` — 提示词模板 + 结构化输出
- `langchain-openai` — `ChatOpenAI` 接 DeepSeek（OpenAI 兼容，`openai_api_base` 指过去）
- `tavily-python` — Tavily 联网搜索 SDK
- `trafilatura` — 网页正文抽取（反爬/清洗）；（可选 `readability-lxml` 兜底）

## 路线图

- **阶段一（先跑通主链路）**：StateGraph 建图 + web_search + reader + 判官两段式 + SSE，先纯联网（`enable_local=false`）验证 Deep Research 流程闭环。
- **阶段二（接本地 tool）**：`local_search` 复用 `rag_retrieval.pipeline.search`，round 1 web+local 双跑，结果标注来源；视需要合并/重排两路结果。
- **阶段三（体验与评测）**：answer 流式细化、检索过程可视化、`result.json` 攒 eval 集、结果缓存与去重。
