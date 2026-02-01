"""
Configuration loader for RDALS.
Loads settings from config.yaml and provides a Config class for access.
"""

import os
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime

import yaml


def _load_yaml(config_path: str = None) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if config_path is None:
        # Default: look for config.yaml in project root
        project_root = Path(__file__).parent.parent
        config_path = project_root / "config.yaml"
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


@dataclass
class PathsConfig:
    data_root: str = "../code_rdals copy/data"
    embedding_dir: str = "../code_rdals copy/data/all_embedding"
    embedding_layer3_dir: str = "../code_rdals copy/data/all_embedding_layer3"
    results_prefix: str = "./results"


@dataclass
class DatasetConfig:
    name: str = "cifar10"
    num_classes: int = 10
    source_samples: int = 10000
    target_samples: int = 10000


@dataclass
class ModelConfig:
    name: str = "resnet18"
    weights: str = "DEFAULT"
    device: str = "cuda:0"
    batch_size: int = 256
    num_workers: int = 4


@dataclass
class ShiftConfig:
    name: str = "dirichlet"
    domain: str = "source"
    alpha: float = 1.0
    rho: float = 0.9
    target_label: int = 0


@dataclass
class EstimationConfig:
    regularizer_lambda: float = 0.01
    min_prob: float = 0.01
    q_init_type: int = 1


@dataclass
class RLLSConfig:
    rho: float = 0.5
    alpha: float = 0.01
    delta: float = 0.05


@dataclass
class MLLSConfig:
    em_max_iters: int = 100
    em_tol: float = 0.0001
    em_eps: float = 1e-12


@dataclass
class BBSEConfig:
    nonneg: bool = True
    renormalize: bool = True


@dataclass
class HeadConfig:
    """Linear head training parameters for baseline methods."""
    epochs: int = 5
    lr: float = 0.001
    batch_size: int = 128
    momentum: float = 0.9
    weight_decay: float = 0.0005


@dataclass
class BaselinesConfig:
    head: HeadConfig = field(default_factory=HeadConfig)
    rlls: RLLSConfig = field(default_factory=RLLSConfig)
    mlls: MLLSConfig = field(default_factory=MLLSConfig)
    bbse: BBSEConfig = field(default_factory=BBSEConfig)
    split_train_calibration: bool = False
    calibration_ratio: float = 0.5


@dataclass
class DownstreamConfig:
    train: bool = False
    model: Optional[str] = None
    finetune_mode: int = 1
    epochs: int = 20
    head_lr: float = 0.001
    head_batch_size: int = 256
    backbone_lr: float = 0.0001
    weight_decay: float = 0.0005
    momentum: float = 0.9
    num_workers: int = 4
    freeze_bn: bool = True
    eval_strategy: str = "reweight"
    logit_bias_gamma: float = 1.0
    expectation_normalize: bool = True
    eps: float = 1e-12
    seed_base: Optional[int] = None


@dataclass
class ExperimentConfig:
    num_trials: int = 100
    random_seed: Optional[int] = 42
    exclude_extreme_ratio: float = 0.05


@dataclass
class PlottingConfig:
    show_std: bool = True
    y_scale: str = "symlog"
    y_symlog_linthresh: float = 0.0001
    y_symlog_linscale: float = 1.0
    y_symlog_base: int = 10


@dataclass
class Config:
    """Main configuration class that holds all settings."""
    paths: PathsConfig = field(default_factory=PathsConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    shift: ShiftConfig = field(default_factory=ShiftConfig)
    estimation: EstimationConfig = field(default_factory=EstimationConfig)
    baselines: BaselinesConfig = field(default_factory=BaselinesConfig)
    downstream: DownstreamConfig = field(default_factory=DownstreamConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    plotting: PlottingConfig = field(default_factory=PlottingConfig)
    
    @classmethod
    def from_yaml(cls, config_path: str = None) -> "Config":
        """Load configuration from YAML file."""
        data = _load_yaml(config_path)
        return cls._from_dict(data)
    
    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create Config from dictionary."""
        paths = PathsConfig(**data.get("paths", {}))
        dataset = DatasetConfig(**data.get("dataset", {}))
        model = ModelConfig(**data.get("model", {}))
        shift = ShiftConfig(**data.get("shift", {}))
        estimation = EstimationConfig(**data.get("estimation", {}))
        
        # Handle nested baselines config
        baselines_data = data.get("baselines", {})
        head = HeadConfig(**baselines_data.get("head", {}))
        rlls = RLLSConfig(**baselines_data.get("rlls", {}))
        mlls = MLLSConfig(**baselines_data.get("mlls", {}))
        bbse = BBSEConfig(**baselines_data.get("bbse", {}))
        baselines = BaselinesConfig(
            head=head,
            rlls=rlls,
            mlls=mlls,
            bbse=bbse,
            split_train_calibration=baselines_data.get("split_train_calibration", False),
            calibration_ratio=baselines_data.get("calibration_ratio", 0.5),
        )
        
        downstream = DownstreamConfig(**data.get("downstream", {}))
        experiment = ExperimentConfig(**data.get("experiment", {}))
        plotting = PlottingConfig(**data.get("plotting", {}))
        
        return cls(
            paths=paths,
            dataset=dataset,
            model=model,
            shift=shift,
            estimation=estimation,
            baselines=baselines,
            downstream=downstream,
            experiment=experiment,
            plotting=plotting,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            "paths": {
                "data_root": self.paths.data_root,
                "embedding_dir": self.paths.embedding_dir,
                "embedding_layer3_dir": self.paths.embedding_layer3_dir,
                "results_prefix": self.paths.results_prefix,
            },
            "dataset": {
                "name": self.dataset.name,
                "num_classes": self.dataset.num_classes,
                "source_samples": self.dataset.source_samples,
                "target_samples": self.dataset.target_samples,
            },
            "model": {
                "name": self.model.name,
                "weights": self.model.weights,
                "device": self.model.device,
                "batch_size": self.model.batch_size,
                "num_workers": self.model.num_workers,
            },
            "shift": {
                "name": self.shift.name,
                "domain": self.shift.domain,
                "alpha": self.shift.alpha,
                "rho": self.shift.rho,
                "target_label": self.shift.target_label,
            },
            "estimation": {
                "regularizer_lambda": self.estimation.regularizer_lambda,
                "min_prob": self.estimation.min_prob,
                "q_init_type": self.estimation.q_init_type,
            },
            "baselines": {
                "rlls": {
                    "rho": self.baselines.rlls.rho,
                    "alpha": self.baselines.rlls.alpha,
                    "delta": self.baselines.rlls.delta,
                },
                "mlls": {
                    "em_max_iters": self.baselines.mlls.em_max_iters,
                    "em_tol": self.baselines.mlls.em_tol,
                    "em_eps": self.baselines.mlls.em_eps,
                },
                "bbse": {
                    "nonneg": self.baselines.bbse.nonneg,
                    "renormalize": self.baselines.bbse.renormalize,
                },
                "split_train_calibration": self.baselines.split_train_calibration,
                "calibration_ratio": self.baselines.calibration_ratio,
            },
            "downstream": {
                "train": self.downstream.train,
                "model": self.downstream.model,
                "finetune_mode": self.downstream.finetune_mode,
                "epochs": self.downstream.epochs,
                "head_lr": self.downstream.head_lr,
                "backbone_lr": self.downstream.backbone_lr,
                "weight_decay": self.downstream.weight_decay,
                "momentum": self.downstream.momentum,
                "eval_strategy": self.downstream.eval_strategy,
            },
            "experiment": {
                "num_trials": self.experiment.num_trials,
                "random_seed": self.experiment.random_seed,
                "exclude_extreme_ratio": self.experiment.exclude_extreme_ratio,
            },
            "plotting": {
                "show_std": self.plotting.show_std,
                "y_scale": self.plotting.y_scale,
                "y_symlog_linthresh": self.plotting.y_symlog_linthresh,
            },
        }
    
    def create_results_dir(self, extra_meta: Dict[str, Any] = None) -> Path:
        """Create a timestamped results directory and save config metadata."""
        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        results_dir = Path(self.paths.results_prefix) / f"results_{timestamp}"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        meta = self.to_dict()
        if extra_meta:
            meta["extra"] = extra_meta
        
        meta_path = results_dir / "meta.json"
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False, default=str)
        
        return results_dir


# Global config instance (can be overridden)
_global_config: Optional[Config] = None


def get_config(config_path: str = None) -> Config:
    """Get the global config instance, loading from YAML if needed."""
    global _global_config
    if _global_config is None:
        _global_config = Config.from_yaml(config_path)
    return _global_config


def set_config(config: Config) -> None:
    """Set the global config instance."""
    global _global_config
    _global_config = config


def reload_config(config_path: str = None) -> Config:
    """Reload configuration from YAML file."""
    global _global_config
    _global_config = Config.from_yaml(config_path)
    return _global_config
