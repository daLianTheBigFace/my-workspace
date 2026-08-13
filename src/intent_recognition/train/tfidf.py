"""TFIDF 训练脚本：中文 char n-gram + OneVsRest 逻辑回归（多标签），输出到 models/tfidf/。

主次意图 → multi-hot 2D 指示矩阵；每列一个独立二分类（predict_proba 每列独立概率）。

用法：uv run python -m intent_recognition.train.tfidf
"""
from __future__ import annotations

import json

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, hamming_loss
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier

from ..config import DataConfig, LABELS, PredictorConfig
from ..data.loader import load_multi_raw_data, to_multi_hot, to_primary_id


def train_tfidf() -> None:
    data_cfg, pred_cfg = DataConfig(), PredictorConfig()

    # 多标签数据：multi-hot 2D 指示矩阵 (n,12) + 主意图 id（用于分层划分）
    ds = load_multi_raw_data(data_cfg)
    texts, label_strs = list(ds["text"]), list(ds["label"])
    y = [to_multi_hot(LABELS, s) for s in label_strs]
    primary = [to_primary_id(LABELS, s) for s in label_strs]

    # 中文处理：char n-gram(1,3) —— 免新增 jieba 依赖、对短中文指令句强、OOV 鲁棒
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(1, 3),
        sublinear_tf=True,
        min_df=2,
        max_features=50_000,
    )
    X = vectorizer.fit_transform(texts)

    # 多标签：每列独立二分类；class_weight 缓解类不均衡
    # sklearn>=1.9 的 LogisticRegression 已是 multinomial（softmax）行为，无需 multi_class 参数
    clf = OneVsRestClassifier(
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, solver="lbfgs")
    )

    # 验证：8:2 分层划分（按主意图，与 BERT 一致），打印 subset acc / hamming / f1
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=data_cfg.test_ratio, random_state=42, stratify=primary
    )
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_te)
    subset_acc = (preds == y_te).all(axis=1).mean()
    print(
        f"TFIDF subset_acc={subset_acc:.4f} "
        f"hamming={hamming_loss(y_te, preds):.4f} "
        f"f1_macro={f1_score(y_te, preds, average='macro', zero_division=0):.4f}"
    )

    # 保存到 models/tfidf/（labels.json 供预测；meta.json 标记多标签模式供 predictor 检查）
    pred_cfg.tfidf_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, pred_cfg.tfidf_dir / "vectorizer.joblib")
    joblib.dump(clf, pred_cfg.tfidf_dir / "classifier.joblib")
    (pred_cfg.tfidf_dir / "labels.json").write_text(
        json.dumps({"labels": list(LABELS)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (pred_cfg.tfidf_dir / "meta.json").write_text(
        json.dumps({"mode": "multi_label", "num_labels": len(LABELS)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"已保存到 {pred_cfg.tfidf_dir}")


if __name__ == "__main__":
    train_tfidf()
