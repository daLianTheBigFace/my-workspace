"""全局配置：路径与超参数。

集中管理，避免散落在各脚本里。路径统一相对项目根目录解析，
无论从哪个目录运行都能正确找到 assets/、models/ 等。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# src/my_workspace/config.py -> 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 12 个意图类别（按字母序，作为 id2label 的固定顺序）
LABELS = [
    "Alarm-Update",
    "Audio-Play",
    "Calendar-Query",
    "FilmTele-Play",
    "HomeAppliance-Control",
    "Music-Play",
    "Other",
    "Radio-Listen",
    "Travel-Query",
    "TVProgram-Play",
    "Video-Play",
    "Weather-Query",
]


@dataclass
class DataConfig:
    """数据路径配置。资源放在 assets/ 下。"""

    raw_csv: Path = field(
        default_factory=lambda: PROJECT_ROOT / "assets" / "dataset" / "dataset.csv"
    )
    multi_csv: Path = field(
        default_factory=lambda: PROJECT_ROOT / "assets" / "dataset" / "dataset_multi.csv"
    )
    stopwords: Path = field(
        default_factory=lambda: PROJECT_ROOT / "assets" / "dataset" / "baidu_stopwords.txt"
    )
    test_ratio: float = 0.2  # 训练/测试划分比例


@dataclass
class ModelConfig:
    """模型配置：本地已下载的 BERT 中文模型。"""

    pretrained_model: Path = field(
        default_factory=lambda: PROJECT_ROOT / "assets" / "models" / "bert-base-chinese"
    )
    num_labels: int = len(LABELS)
    labels: list[str] = field(default_factory=lambda: LABELS)
    max_length: int = 128


@dataclass
class TrainConfig:
    """训练超参数。"""

    batch_size: int = 16
    epochs: int = 3
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    output_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "models")
    log_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "outputs")

    def ensure_dirs(self) -> None:
        for d in (self.output_dir, self.log_dir):
            d.mkdir(parents=True, exist_ok=True)


@dataclass
class PredictorConfig:
    """推理服务配置：默认方案 + 各方案模型产物路径。"""

    default_model: str = "bert"
    final_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "models" / "final"
    )
    tfidf_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "models" / "tfidf"
    )
    multi_intent_threshold: float = 0.3  # 平等多意图判定阈值：sigmoid 概率 ≥ 此值即返回（重训后可微调）
