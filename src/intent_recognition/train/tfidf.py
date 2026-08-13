"""TFIDF 训练脚本：中文 char n-gram + 逻辑回归，输出到 models/tfidf/。

用法：uv run python -m intent_recognition.train.tfidf
"""
from __future__ import annotations

import json

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from ..config import DataConfig, LABELS, PredictorConfig
from ..data.loader import load_raw_data


def train_tfidf() -> None:
    data_cfg, pred_cfg = DataConfig(), PredictorConfig()

    # 复用 BERT 同款数据加载
    ds = load_raw_data(data_cfg)
    texts, label_names = list(ds["text"]), list(ds["label"])
    label2id = {name: i for i, name in enumerate(LABELS)}
    y = [label2id[n] for n in label_names]

    # 中文处理：char n-gram(1,3) —— 免新增 jieba 依赖、对短中文指令句强、OOV 鲁棒
    # 将来换 jieba 分词：analyzer="char" 换成 analyzer="word" + tokenizer=jieba.lcut，单行改动
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(1, 3),
        sublinear_tf=True,
        min_df=2,
        max_features=50_000,
    )
    # 需 predict_proba 输出 top3 → 排除 LinearSVC；class_weight 缓解 6:1 类不均衡
    # sklearn>=1.9 的 LogisticRegression 已是 multinomial（softmax）行为，无需 multi_class 参数
    clf = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
    )

    X = vectorizer.fit_transform(texts)

    # 验证：8:2 分层划分（与 BERT 训练一致），打印 acc / f1
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=data_cfg.test_ratio, random_state=42, stratify=y
    )
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_te)
    print(
        f"TFIDF acc={accuracy_score(y_te, preds):.4f} "
        f"f1_macro={f1_score(y_te, preds, average='macro'):.4f}"
    )

    # 保存到 models/tfidf/
    pred_cfg.tfidf_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, pred_cfg.tfidf_dir / "vectorizer.joblib")
    joblib.dump(clf, pred_cfg.tfidf_dir / "classifier.joblib")
    (pred_cfg.tfidf_dir / "labels.json").write_text(
        json.dumps({"labels": list(LABELS)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"已保存到 {pred_cfg.tfidf_dir}")


if __name__ == "__main__":
    train_tfidf()
