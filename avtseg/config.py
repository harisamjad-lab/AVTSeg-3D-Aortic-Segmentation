from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
import json
import os
import torch


@dataclass
class DataConfig:
    data_dir: str = "data"
    workdir: str = "workdir"
    k_folds: int = 5
    val_fold: int = 0
    num_workers: int = 4


@dataclass
class PreprocessConfig:
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
    intensity_min: float = -200.0
    intensity_max: float = 800.0
    intensity_shift_threshold: float = 500.0


@dataclass
class Stage1Config:
    patch_size: tuple[int, int, int] = (128, 128, 128)
    batch_size: int = 1
    max_epochs: int = 5
    lr: float = 2e-4
    weight_decay: float = 1e-5
    sw_batch_size: int = 1
    overlap: float = 0.25
    threshold: float = 0.5
    train_fg_prob: float = 0.7


@dataclass
class Stage2Config:
    patch_size: tuple[int, int, int] = (96, 96, 96)
    batch_size: int = 2
    max_epochs: int = 5
    lr: float = 2e-4
    weight_decay: float = 1e-5
    threshold: float = 0.5
    center_spacing_vox: int = 12
    max_centers_per_case: int = 256
    min_component_size: int = 200


@dataclass
class RuntimeConfig:
    seed: int = 42
    amp: bool = True
    val_interval: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    stage1: Stage1Config = field(default_factory=Stage1Config)
    stage2: Stage2Config = field(default_factory=Stage2Config)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def to_dict(self):
        return {
            "data": asdict(self.data),
            "preprocess": asdict(self.preprocess),
            "stage1": asdict(self.stage1),
            "stage2": asdict(self.stage2),
            "runtime": asdict(self.runtime),
        }

    def save(self, path: str | os.PathLike) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def load_config() -> Config:
    return Config()


