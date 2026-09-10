# intent-recognition

一个 FastAPI 推理服务，聚合六块能力（都是独立包，通过 `src/intent_recognition/api/app.py` 一个入口暴露）：

| 包 | 能力 | 接口 |
|---|---|---|
| `intent_recognition` | 可插拔多方案意图识别（BERT / TFIDF / 预留 LLM） | `/predict` `/models` |
| `sentence_bert` | 句子相似度（bge 编码） | `/similarity` |
| `rag_retrieval` | RAG 多路召回 + 重排（对齐 week06 页面级流程） | `/rag/search` `/rag/index_info` |
| `es_search` | 基于本地 Elasticsearch 的全文 / 过滤 / 向量检索 | `/es/*` |
| `pageindex_svc` | 文档检索（VectifyAI PageIndex 本地引擎：推理式无向量 RAG，DeepSeek） | `/pageindex/*` |
| `deep_search` | 深度搜索（Deep Research 智能体：拆解→联网+本地双 tool 检索→阅读→判官→迭代→合成，SSE 流式） | `/deep_search/search` `/deep_search/health` |

## 目录结构

```
my-workspace/
├── src/
│   ├── intent_recognition/   # 主服务入口（app.py）+ 意图识别核心
│   │   ├── config.py         #   路径与超参配置（含 PredictorConfig）
│   │   ├── predictors/       #   ★ 可插拔推理核心（base/bert/tfidf/llm）
│   │   ├── train/            #   训练脚本（bert.py / tfidf.py）
│   │   └── api/              #   FastAPI 推理服务（app.py）
│   ├── sentence_bert/        # 句子相似度（bge-small-zh-v1.5）
│   ├── rag_retrieval/        # RAG 检索：dense/BM25/TFIDF + RRF 融合 + 重排
│   │   ├── retrievers/       #   dense_bge / bm25 / tfidf（页面级召回）
│   │   ├── fusion.py         #   RRF 融合
│   │   ├── rerank.py         #   bge-reranker 重排
│   │   └── api/              #   /rag 路由
│   ├── es_search/            # 本地 ES 检索：全文 / 条件过滤 / 向量
│   │   ├── queries.py        #   查询体构造（纯函数，可单测）
│   │   ├── search.py         #   三大检索能力
│   │   ├── build_index.py    #   灌数据建索引
│   │   └── api/              #   /es 路由
│   ├── pageindex_svc/        # 文档检索：包装 VectifyAI PageIndex 本地引擎
│   │   ├── config.py         #   模型/索引路径配置（DeepSeek，索引落 E 盘）
│   │   ├── client.py         #   PageIndexLocalClient 惰性单例
│   │   ├── service.py        #   建索引 / 列文档 / 取树 / 推理检索问答
│   │   ├── api/              #   /pageindex 路由（router + warm）
│   │   └── serve.py          #   独立起服务入口（uvicorn pageindex_svc.serve:app）
│   └── deep_search/          # 深度搜索（Deep Research 智能体，LangGraph 编排）
│       ├── graph.py          #   StateGraph 5 节点 + 条件回边（编排核心）
│       ├── agents/           #   planner / judge / synthesizer（三个 LLM 角色）
│       ├── tools/            #   web_search / local_search / reader（非 LLM 工具）
│       ├── output.py         #   落盘 report.md + result.json
│       └── api/              #   /deep_search 路由（SSE + warm）
├── assets/                   # 资源（大文件不入库）
│   ├── dataset/              #   原始数据 + 停用词
│   ├── models/               #   本地模型（bge、bert，git 忽略）
│   └── Week06/               #   课程文件（git 忽略）
├── data/
│   ├── rag_index/            # RAG/ES 复用：chunks.json + embeddings.npy（git 忽略）
│   └── pageindex/            # PageIndex 本地索引（git 忽略）
├── models/                   # 训练产物（git 忽略）
├── tests/                    # 测试
├── pyproject.toml            # 依赖与构建配置（uv 管理）
└── uv.lock
```

## 快速开始

环境管理用 [uv](https://docs.astral.sh/uv/)。首次同步安装依赖：

```bash
uv sync
```

> torch 已配置为从 PyTorch 官方 CUDA 源（cu130）解析，见 `pyproject.toml` 的 `[tool.uv]`。

### 启动服务

```bash
uv run uvicorn intent_recognition.api.app:app --reload --port 8000
```

打开 `http://127.0.0.1:8000/docs`（Swagger UI）可交互式调试所有接口。

### 运行测试

```bash
uv run pytest            # 全量（含真实模型 / 真实 ES 冒烟）
uv run pytest -m "not slow"   # 只跑快路径，不加载真实模型 / 不连 ES
```

## 意图识别（/predict）

- **数据**：`assets/dataset/dataset.csv`（TSV，`文本\t标签`，12 类意图），加载逻辑在 `data/loader.py`。
- **推理方案**：`GET /models` 列出可用方案；`POST /predict` 用 `model` 字段指定（默认 `bert`）。
- **训练**（可选，产物已在 `models/`）：
  ```bash
  uv run python -m intent_recognition.train.bert    # 需要 GPU，产物 models/final/
  uv run python -m intent_recognition.train.tfidf   # 产物 models/tfidf/
  ```
- **多意图**：`/predict` 返回所有 ≥ 阈值（`multi_intent_threshold`）的意图，各自独立置信度；`top3` 为兜底全排序。
- **调用示例**：
  ```bash
  curl -X POST http://127.0.0.1:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"text": "今天天气怎么样"}'
  ```

## 句子相似度（/similarity）

从候选文本列表中找出与 query 最相似的一句：

```bash
curl -X POST http://127.0.0.1:8000/similarity \
  -H "Content-Type: application/json" \
  -d '{"query": "怎么开空调", "candidates": ["打开空调", "调座椅高度", "播放音乐"]}'
```

返回 `{query, text, index, score}`。

## RAG 检索（/rag）

对齐 week06 的**页面级流程**：dense 取 top100 chunk → 页面去重 → top10 页；BM25/TFIDF 对整页建索引；页面级 RRF 融合；重排器喂 top3 页整页文本。

```bash
# 不带回答：只返回检索结果
curl -X POST http://127.0.0.1:8000/rag/search \
  -H "Content-Type: application/json" \
  -d '{"query": "空调怎么开", "top_k": 3}'

# 带回答：需要 LLM key（见 .env.example，复制为 .env 填上）
curl -X POST http://127.0.0.1:8000/rag/search \
  -H "Content-Type: application/json" \
  -d '{"query": "空调怎么开", "with_answer": true}'
```

**匹配阈值**：top1 重排分 < `min_rerank_score`（0.3）时返回 `matched: false` + `message` 提示，不再调 LLM（手册外问题不再硬编造答案）。

## 本地 ES + /es 检索

### 环境

- ES 8.17.8（本地 zip 版，自带 JDK），装在 `E:\elasticsearch-8.17.8`，无认证明文 HTTP `http://localhost:9200`。
- **重要**：elasticsearch-py 客户端必须 < 9（`elasticsearch>=8.17,<9`）。9.x 客户端连不了 8.x 服务端（`Accept: compatible-with=9` 被拒）。
- IK 中文分词插件版本必须与 ES 完全一致。

启动 ES：

```bash
E:\elasticsearch-8.17.8\bin\elasticsearch.bat
# 验证：curl http://localhost:9200  应返回集群信息（无认证）
```

### 建索引

索引 `car_manual` 复用 `data/rag_index/` 的 3743 chunk + 512 维 bge 向量，秒级建库（缺失则自动转 pdfplumber 解析兜底）：

```bash
cd E:\project\my-workspace
PYTHONPATH=src uv run python -m es_search.build_index
```

### 接口

| 接口 | 请求体 | 说明 |
|---|---|---|
| `GET /es/index_info` | — | ES 连接状态 + 索引文档数 |
| `POST /es/full_text` | `{"query","size","match_type"}` | 全文检索（IK 分词） |
| `POST /es/filter` | `{"query","page_from","page_to","size","match_type"}` | 全文 + 页码范围过滤 |
| `POST /es/vector` | `{"query","size"}` | 向量检索（bge 编码 + knn，语义相似） |

统一返回 `{query, hits: [{chunk_id, page, text, score}], total}`。

**match_type 三档**（全文/过滤都支持）：

| 值 | 含义 | 典型效果 |
|---|---|---|
| `match`（默认） | 分词后任一命中，宽松 | 「空调 座椅」→ 258 条 |
| `phrase` | 精确短语，分词后必须按顺序连续出现 | 「空调 座椅」→ 0 条 |
| `fuzzy` | 容错模糊（fuzziness=1，容忍错别字） | 「坐椅」错字 → 命中 187 条座椅内容 |

> 注意：`fuzziness` 不能用 `AUTO`——它对 2 字符中文词只允许 0 次编辑（等于无容错），所以显式用 1。

## pageindex 文档检索（/pageindex）

包装 [VectifyAI/PageIndex](https://github.com/VectifyAI/PageIndex)（v0.2.10，依赖从 git 源装，见 `pyproject.toml` `[tool.uv.sources]`）的**本地引擎**——向量无关 / 推理式 RAG：先把 PDF 建成「目录树」索引，检索时模型对着精简树推理、再读命中的页回答。

- **LLM**：建索引与检索都要模型。复用仓库 `.env` 的 DeepSeek（`deepseek/deepseek-chat`，litellm 写法），缺省回落 `RAG_LLM_API_KEY`。想换模型设 `PAGEINDEX_MODEL` / `PAGEINDEX_SUMMARY_MODEL`。
- **本地包装包命名**：`pageindex_svc`（不用 `pageindex` 做顶层名，避免和 git 依赖 `pageindex` 同名冲突）。可并入主服务（本页开头那个入口已 include），也可独立起：`uv run uvicorn pageindex_svc.serve:app --port 8010`。
- **索引落盘**：`data/pageindex/`（E 盘，git 忽略）。已建索引的文档可复用；列表/取树不需要 key，建索引/检索需要。

接口：

| 接口 | 说明 |
|---|---|
| `POST /pageindex/submit` | 提交 PDF 建索引。`mode=flash`（默认：本地版面抽结构 + LLM 写摘要，快）；`mode=standard`（全 LLM 建树，慢）。`pdf_path` 缺省用汽车手册。 |
| `GET /pageindex/documents` | 已建索引列表 |
| `GET /pageindex/documents/{doc_id}` | 文档元信息（页数/描述等） |
| `GET /pageindex/documents/{doc_id}/tree` | 文档树结构（`?node_summary=1` 带摘要，`?include_text=1` 带整页文本） |
| `POST /pageindex/query` | 推理式检索 + 问答。`with_process=true` 时返回检索过程（思考/工具调用/命中片段） |
| `DELETE /pageindex/documents/{doc_id}` | 删除索引 |
| `GET /pageindex/health` | key / 索引目录状态 |

调用示例（汽车手册）：

```bash
# 建索引（DeepSeek 写节点摘要，354 页手册约几分钟；返回 doc_id）
curl -X POST http://127.0.0.1:8000/pageindex/submit \
  -H "Content-Type: application/json" \
  -d '{"mode": "flash"}'

# 检索 + 问答，带过程
curl -X POST http://127.0.0.1:8000/pageindex/query \
  -H "Content-Type: application/json" \
  -d '{"doc_id": "<上一步的 doc_id>", "query": "怎么打开空调？", "with_process": true}'
```

> 上游引擎仍是 alpha（v0.2.10）。已知坑：flash/standard 都要求**完整连贯文档**——从大 PDF 里截出的 3 页小样会抽不出结构或触发 TOC 页码 bug，请对整本手册用。

## deep_search 深度搜索（/deep_search）

Deep Research 智能体：把问题拆解 → 联网（Tavily）+ 本地手册**双 tool** 检索 → 抓取阅读 → 判官把关/补搜 → 迭代 → 合成带引用的答案，全程 SSE 流式输出，结束落盘报告与结构化结果。与前 5 个「一次检索出结果」的包不同，它是**多步自主**的搜索编排层。

- **编排**：LangGraph `StateGraph`（5 节点 + 1 条条件回边）。
- **判官两段式**：质量过滤（read 内逐篇判可信/相关）+ 充分性判断（reflect 节点，输出可搜索的补搜 query）；收敛三保险 = `max_rounds` 硬停 + 无新增停 + 判官自判够。
- **LLM**：复用 `.env` 的 DeepSeek（`RAG_LLM_*` 回落 `DEEPSEEK_API_KEY`）；联网用 `TAVILY_API_KEY`，缺了只跑本地 tool。
- **结果三层**：SSE 流式 answer → done 概览（sources/stats）→ 落盘 `outputs/deep_search/<ts>-<slug>/` 的 `report.md` + `result.json`（`save_files` 可关）。

```bash
curl -N -X POST http://127.0.0.1:8000/deep_search/search \
  -H "Content-Type: application/json" \
  -d '{"question": "如何设置定速巡航？"}'
```

事件：`plan`（拆解）→ `search`（web/local）→ `read`（抓取）→ `reflect`（判官）→ `answer`（流式增量）→ `done`（概览 + 落盘路径）。

> SSE 的 `EventSource` 只支持 GET，本接口用 POST 传 body，前端用 `fetch` + `ReadableStream` 手动拆帧。

## 多方案设计

- 统一接口 `BasePredictor.predict(text, top_k) -> list[Prediction]`，各方案实现后通过 `@register_predictor` 注册。
- API 请求用 `model` 字段指定方案，未知方案返回 404 并附可用列表。
- LLM 接入步骤见 `predictors/llm.py` 顶部注释。
