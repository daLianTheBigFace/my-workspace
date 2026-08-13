"""LLM 推理方案（预留扩展点，未实现）。

刻意不注册，避免 GET /models 列出不可用方案、POST /predict 收到 500。

接入步骤（将来）：
1. 在本文件实现 LlmPredictor(BasePredictor)，完成 predict(text, top_k)；
2. 给类加 name = "llm"，并加上 @register_predictor 装饰器；
3. 在 predictors/__init__.py 里 import 本模块（触发注册）；
4. 重启服务，GET /models 与 POST /predict {"model":"llm"} 即可用。
"""
from __future__ import annotations
