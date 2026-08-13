"""训练脚本：微调本地 BERT（bert-base-chinese）做多标签意图识别。

主次意图：text \\t main,sub → multi-hot 标签（(batch, num_labels) float32）。
模型 problem_type=multi_label_classification，自动用 BCEWithLogitsLoss + sigmoid。

用法：uv run python -m intent_recognition.train.bert
依赖：transformers、datasets、scikit-learn（见 pyproject.toml）
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, hamming_loss
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from ..config import DataConfig, ModelConfig, TrainConfig
from ..data.loader import to_multi_label_dataset


def compute_metrics(eval_pred) -> dict:
    """多标签指标：sigmoid + 0.5 阈值 → subset acc / hamming / f1_macro。"""
    logits, labels = eval_pred
    preds = (1 / (1 + np.exp(-logits)) > 0.5).astype(int)
    return {
        "subset_acc": float((preds == labels).all(axis=1).mean()),
        "hamming": float(hamming_loss(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
    }


class EpochSummaryCallback(TrainerCallback):
    """每个 epoch 结束时打印一行汇总：train_loss / eval_loss / acc / f1。"""

    def __init__(self) -> None:
        self._train_losses: list[float] = []
        self._printed_epochs: set[int] = set()

    def on_log(self, args, state, control, logs, **kwargs) -> None:
        if "loss" in logs:
            self._train_losses.append(float(logs["loss"]))

    def on_evaluate(self, args, state, control, metrics, **kwargs) -> None:
        epoch = int(metrics.get("epoch", 0))
        if epoch in self._printed_epochs:
            return  # 跳过 load_best_model_at_end 触发的重复评估
        self._printed_epochs.add(epoch)

        train_loss = (
            sum(self._train_losses) / len(self._train_losses)
            if self._train_losses
            else float("nan")
        )
        self._train_losses.clear()

        parts = [
            f"Epoch {epoch}/{args.num_train_epochs}",
            f"train_loss {train_loss:.4f}",
            f"eval_loss {float(metrics['eval_loss']):.4f}",
        ]
        if "eval_subset_acc" in metrics:
            parts.append(f"subset_acc {float(metrics['eval_subset_acc']):.4f}")
        if "eval_hamming" in metrics:
            parts.append(f"hamming {float(metrics['eval_hamming']):.4f}")
        if "eval_f1_macro" in metrics:
            parts.append(f"f1 {float(metrics['eval_f1_macro']):.4f}")
        print(" | ".join(parts), flush=True)


def train() -> None:
    data_cfg, model_cfg, train_cfg = DataConfig(), ModelConfig(), TrainConfig()
    train_cfg.ensure_dirs()

    # 1. 加载多标签数据并 8:2 划分（按主意图 primary 分层，保证小类在测试集里也有代表性）
    ds = to_multi_label_dataset(model_cfg, data_cfg)
    split = ds.train_test_split(
        test_size=data_cfg.test_ratio, seed=42, stratify_by_column="primary"
    )
    train_ds, eval_ds = split["train"], split["test"]
    print(f"训练集 {len(train_ds)} 条，测试集 {len(eval_ds)} 条")

    # 2. 分词
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.pretrained_model)
    id2label = {i: name for i, name in enumerate(model_cfg.labels)}
    label2id = {name: i for i, name in id2label.items()}

    def tokenize(examples: dict) -> dict:
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=model_cfg.max_length,
            padding="max_length",
        )

    train_ds = train_ds.map(tokenize, batched=True)
    eval_ds = eval_ds.map(tokenize, batched=True)
    keep = ("input_ids", "attention_mask", "token_type_ids", "label")
    train_ds = train_ds.remove_columns([c for c in train_ds.column_names if c not in keep])
    eval_ds = eval_ds.remove_columns([c for c in eval_ds.column_names if c not in keep])
    # label 为 multi-hot float32，DataCollatorWithPadding 自动推断 float32，无需自定义 collator

    # 3. 加载本地 BERT + 多标签分类头（BCEWithLogitsLoss + sigmoid）
    model = AutoModelForSequenceClassification.from_pretrained(
        model_cfg.pretrained_model,
        num_labels=model_cfg.num_labels,
        id2label=id2label,
        label2id=label2id,
        problem_type="multi_label_classification",
    )

    # 4. 训练参数
    args = TrainingArguments(
        output_dir=str(train_cfg.output_dir),
        num_train_epochs=train_cfg.epochs,
        per_device_train_batch_size=train_cfg.batch_size,
        per_device_eval_batch_size=train_cfg.batch_size,
        learning_rate=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,  # 只留 best + 最新 checkpoint，避免每次训练积累几个 G
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
        callbacks=[EpochSummaryCallback()],
    )

    # 5. 训练并保存
    trainer.train()
    final_dir = train_cfg.output_dir / "final"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)  # 5.x 的 Trainer 不再自动存 tokenizer
    print(f"训练完成，模型已保存到 {final_dir}")


if __name__ == "__main__":
    train()
