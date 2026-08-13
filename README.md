# intent-recognition

可插拔多方案意图识别推理服务（BERT / TFIDF / 预留 LLM 扩展点）。

## 目录结构

```
my-workspace/
├── src/intent_recognition/   # 项目代码（标准 src 布局）
│   ├── config.py             #   路径与超参配置（含 PredictorConfig）
│   ├── data/                 #   数据加载（loader.py）
│   ├── predictors/           #   ★ 可插拔推理核心
│   │   ├── base.py           #     统一接口 + 注册表 + 单例惰性加载
│   │   ├── bert.py           #     BERT 方案 → models/final/
│   │   ├── tfidf.py          #     TFIDF 方案 → models/tfidf/
│   │   └── llm.py            #     LLM 扩展点（预留，未实现）
│   ├── train/                #   训练脚本
│   │   ├── bert.py           #     BERT 微调
│   │   └── tfidf.py          #     TFIDF 训练
│   └── api/                  #   FastAPI 推理服务（app.py）
├── assets/                   # 资源（不入库的大文件见 .gitignore）
│   ├── dataset/              #   原始数据：dataset.csv（TSV）+ 停用词
│   └── models/               #   本地 BERT 中文模型
├── models/                   # 训练好的模型（输出，git 忽略）
│   ├── final/                #   BERT 微调产物
│   └── tfidf/                #   TFIDF 产物：vectorizer/classifier/labels
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

## 使用流程

1. **数据**：`assets/dataset/dataset.csv`（TSV，`文本\t标签`，12 类意图）已就位，加载逻辑在 `data/loader.py`。
2. **配置**：`src/intent_recognition/config.py` 已指向本地模型和数据；意图类别为 12（`LABELS`）；推理默认方案 `bert`（`PredictorConfig`）。
3. **训练 BERT**（可选，`models/final/` 已有产物）：
   ```bash
   uv run python -m intent_recognition.train.bert
   ```
4. **训练 TFIDF**（首次必做，生成 `models/tfidf/` 产物）：
   ```bash
   uv run python -m intent_recognition.train.tfidf
   ```
5. **启动推理服务**：
   ```bash
   uv run uvicorn intent_recognition.api.app:app --reload --port 8000
   ```
6. **浏览器测试**：打开 `http://127.0.0.1:8000/docs`（Swagger UI）
   - `POST /predict`：`{"text": "今天天气怎么样"}`（默认 `bert`），或 `{"text": "播放一首歌", "model": "tfidf"}`
   - `GET /models`：列出可用方案；`GET /health`：存活与各方案加载状态
7. **运行测试**：
   ```bash
   uv run pytest            # 全量（含真实模型测试）
   uv run pytest -m "not slow"   # 只跑快路径，不加载真实模型
   ```

## 多方案设计

- 统一接口 `BasePredictor.predict(text, top_k) -> list[Prediction]`，各方案实现后通过 `@register_predictor` 注册。
- API 请求用 `model` 字段指定方案，未知方案返回 404 并附可用列表。
- LLM 接入步骤见 `predictors/llm.py` 顶部注释。
