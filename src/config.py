"""Experiment settings: no Drive mount, installation, or file writes on import."""

from dataclasses import dataclass
from pathlib import Path

CLASS_NAMES = ("W", "N1", "N2", "N3", "REM")
LABEL_TO_NAME = dict(enumerate(CLASS_NAMES))
N_CLASSES = len(CLASS_NAMES)
TARGET_CHANNELS = ("EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal", "EMG submental")
CLASS_WEIGHT = {0: 0.95, 1: 1.18, 2: 1.00, 3: 1.13, 4: 1.20}
MODEL_KEYS = ("rf", "svm", "xgb", "mlp", "cnn1d", "bilstm", "cnn_bilstm", "adaptive")


@dataclass(frozen=True)
class Config:
    cache_path: Path = Path("data/processed_dataset_final.npz")
    result_dir: Path = Path("results/runs")
    models: tuple[str, ...] = MODEL_KEYS
    random_state: int = 42
    batch_size: int = 128
    epochs: int = 40
    learning_rate: float = 2e-4
    patience: int = 8
    n_jobs: int = -1
    show_plots: bool = False
    trust_npz: bool = False
    smoke_test: bool = False
    verbose: int = 1

    def validate(self):
        if not self.models or any(key not in MODEL_KEYS for key in self.models):
            raise ValueError(f"Choose at least one model from {MODEL_KEYS}.")
        if len(set(self.models)) != len(self.models):
            raise ValueError("Model names must not be repeated.")
        if min(self.batch_size, self.epochs, self.patience) < 1:
            raise ValueError("Batch size, epochs and patience must be positive.")
        if self.learning_rate <= 0 or self.n_jobs == 0:
            raise ValueError("Learning rate must be positive; n_jobs cannot be zero.")
