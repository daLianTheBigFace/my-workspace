# intent-recognition

一个 FastAPI 推理服务，聚合四块能力（都是独立包，通过 `src/intent_recognition/api/app.py` 一个入口暴露）：

| 包 | 能力 | 接口 |
|---|---|---|
| `intent_recognition` | 可插拔多方案意图识别（BERT / TFIDF / 预留 LLM） | `/predict` `/models` |
| `sentence_bert` | 句子相似度（bge 编码） | `/similarity` |
| `rag_retrieval` | RAG 多路召回 + 重排（对齐 week06 页面级流程） | `/rag/search` `/rag/index_info` |
| `es_search` | 基于本地 Elasticsearch 的全文 / 过滤 / 向量检索 | `/es/*` |

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
│   └── es_search/            # 本地 ES 检索：全文 / 条件过滤 / 向量
│       ├── queries.py        #   查询体构造（纯函数，可单测）
│       ├── search.py         #   三大检索能力
│       ├── build_index.py    #   灌数据建索引
│       └── api/              #   /es 路由
├── assets/                   # 资源（大文件不入库）
│   ├── dataset/              #   原始数据 + 停用词
│   ├── models/               #   本地模型（bge、bert，git 忽略）
│   └── Week06/               #   课程文件（git 忽略）
├── data/
│   └── rag_index/            # RAG/ES 复用：chunks.json + embeddings.npy（git 忽略）
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

## week06 课程文件对照

课程文件在 `assets/Week06/`（git 忽略），全部无认证连本地 ES 8.x：

| 文件 | 演示 | 前置 |
|---|---|---|
| `04_ES测试.py` | 连接 ES + IK 分词（`_analyze`） | ES 运行中 |
| `05_ES基础.py` | 客户端建索引、写入、中文搜索 | ES 运行中 |
| `06_ES进阶.py` | multi_match + 多条件 filter | ES 运行中 |
| `07_ES向量检索.py` | SentenceTransformer + knn 向量检索 | 建了 `assets/models/BAAI` junction |

`07` 写死加载 `../models/BAAI/bge-small-zh-v1.5`，已用 junction 指向 `assets/models/bge-small-zh-v1.5`，无需改代码。

## 多方案设计

- 统一接口 `BasePredictor.predict(text, top_k) -> list[Prediction]`，各方案实现后通过 `@register_predictor` 注册。
- API 请求用 `model` 字段指定方案，未知方案返回 404 并附可用列表。
- LLM 接入步骤见 `predictors/llm.py` 顶部注释。
