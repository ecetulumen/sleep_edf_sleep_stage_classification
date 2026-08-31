

from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import CLASS_NAMES, CLASS_WEIGHT, N_CLASSES, TARGET_CHANNELS


@dataclass
class Dataset:
    feat_seq: np.ndarray
    feat_center: np.ndarray
    raw_center: np.ndarray
    y: np.ndarray
    subjects: np.ndarray
    feature_names: np.ndarray


@dataclass
class Split:
    raw: np.ndarray
    seq: np.ndarray
    center: np.ndarray
    y: np.ndarray
    subjects: np.ndarray
    indices: np.ndarray


@dataclass
class PreparedData:
    train: Split
    val: Split
    test: Split
    scaler_center: StandardScaler
    scaler_seq: StandardScaler
    feature_names: np.ndarray

    @property
    def ml_trainval(self):
        return np.vstack([self.train.center, self.val.center])

    @property
    def y_trainval(self):
        return np.concatenate([self.train.y, self.val.y])

    @property
    def sample_weight_ml(self):
        return np.array([CLASS_WEIGHT[int(label)] for label in self.y_trainval])


def validate_dataset(data):
    arrays = {"X_feat_seq": data.feat_seq, "X_feat_center": data.feat_center,
              "X_raw_center": data.raw_center, "y_seq": data.y,
              "subject_seq": data.subjects}
    expected_dims = {"X_feat_seq": 3, "X_feat_center": 2, "X_raw_center": 3,
                     "y_seq": 1, "subject_seq": 1}
    for key, array in arrays.items():
        if array.ndim != expected_dims[key]:
            raise ValueError(f"{key}: expected {expected_dims[key]} dimensions; got {array.shape}.")
        if any(size == 0 for size in array.shape):
            raise ValueError(f"{key} must not be empty.")
    n_samples = len(data.y)
    if any(len(array) != n_samples for array in arrays.values()):
        raise ValueError("All NPZ arrays must have the same number of samples.")
    if data.feat_seq.shape[-1] != data.feat_center.shape[-1]:
        raise ValueError("Sequence and center feature dimensions must agree.")
    if data.raw_center.shape[-1] != len(TARGET_CHANNELS):
        raise ValueError("X_raw_center must be (samples, timepoints, 4 channels), in the documented order.")
    if data.raw_center.shape[1] < 4:
        raise ValueError("Raw epochs need at least four timepoints for the two pooling layers.")
    for key, array in arrays.items():
        if key != "subject_seq":
            if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
                raise ValueError(f"{key} must contain finite numeric values.")
    if not np.all(np.isin(data.y, np.arange(N_CLASSES))):
        raise ValueError("Labels must be integers 0=W, 1=N1, 2=N2, 3=N3, 4=REM.")
    if pd.isna(data.subjects).any() or any(not str(s).strip() for s in data.subjects):
        raise ValueError("subject_seq contains a missing participant identifier.")
    if data.feature_names.ndim != 1 or len(data.feature_names) != data.feat_center.shape[1]:
        raise ValueError("feature_names must contain one name per feature.")


def load_dataset(path, *, trust_npz=False):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"NPZ not found: {path}. Set --cache-path to your existing processed dataset.")
    keys = ("X_feat_seq", "X_feat_center", "X_raw_center", "y_seq", "subject_seq")
    try:
        with np.load(path, allow_pickle=trust_npz) as archive:
            missing = sorted(set(keys) - set(archive.files))
            if missing:
                raise ValueError(f"Missing NPZ keys: {missing}")
            values = [archive[key] for key in keys]
            center = values[1]
            names = (archive["feature_names"] if "feature_names" in archive.files else
                     np.array([f"feature_{i}" for i in range(center.shape[-1])]))
    except ValueError as exc:
        if "Object arrays cannot be loaded" in str(exc):
            raise ValueError("This NPZ contains object arrays. Only for a file you created/trust, add --trust-npz.") from exc
        raise
    data = Dataset(*values, names)
    validate_dataset(data)
    # Validate labels BEFORE casting so fractional labels cannot be silently truncated.
    data.feat_seq = data.feat_seq.astype(np.float32)
    data.feat_center = data.feat_center.astype(np.float32)
    data.raw_center = data.raw_center.astype(np.float32)
    data.y = data.y.astype(np.int64)
    validate_dataset(data)
    return data


def prepare_data(data, *, random_state=42):
    validate_dataset(data)
    unique_subjects = np.unique(data.subjects)
    try:
        train_ids, temp_ids = train_test_split(unique_subjects, test_size=0.30, random_state=random_state)
        val_ids, test_ids = train_test_split(temp_ids, test_size=2/3, random_state=random_state)
    except ValueError as exc:
        raise ValueError("Not enough participants for the original 70/10/20 split; at least 7 are needed.") from exc
    split_ids = (train_ids, val_ids, test_ids)
    masks = [np.isin(data.subjects, ids) for ids in split_ids]
    if any(not mask.any() for mask in masks):
        raise ValueError("An empty participant split was produced.")
    for i, ids in enumerate(split_ids):
        for other in split_ids[i+1:]:
            if np.intersect1d(ids, other).size:
                raise ValueError("Participant overlap between splits.")
    n_features = data.feat_center.shape[1]
    center_scaler = StandardScaler().fit(data.feat_center[masks[0]])
    seq_scaler = StandardScaler().fit(data.feat_seq[masks[0]].reshape(-1, n_features))
    splits = []
    for name, mask in zip(("train", "validation", "test"), masks):
        seq = data.feat_seq[mask]
        split = Split(
            raw=data.raw_center[mask],
            seq=seq_scaler.transform(seq.reshape(-1, n_features)).reshape(seq.shape),
            center=center_scaler.transform(data.feat_center[mask]),
            y=data.y[mask], subjects=data.subjects[mask], indices=np.flatnonzero(mask),
        )
        missing = sorted(set(range(N_CLASSES)) - set(split.y.tolist()))
        if missing:
            warnings.warn(f"{name} split lacks classes: {missing}; inspect class-wise metrics.", stacklevel=2)
        splits.append(split)
    return PreparedData(*splits, center_scaler, seq_scaler, data.feature_names)


def save_data_audit(data, output_dir):
    output_dir = Path(output_dir)
    distributions = {}
    assignments = []
    for name, split in (("Train", data.train), ("Validation", data.val), ("Test", data.test)):
        distributions[name] = np.bincount(split.y, minlength=N_CLASSES)
        assignments.append(pd.DataFrame({"sample_index": split.indices,
                                         "participant_id": split.subjects,
                                         "split": name, "label": split.y}))
    frame = pd.DataFrame(distributions, index=CLASS_NAMES)
    frame.to_csv(output_dir / "train_validation_test_sinif_dagilimi.csv", encoding="utf-8-sig")
    pd.concat(assignments).sort_values("sample_index").to_csv(output_dir / "split_assignments.csv", index=False)
    np.savez_compressed(output_dir / "scaling_parameters.npz",
                        center_mean=data.scaler_center.mean_, center_scale=data.scaler_center.scale_,
                        sequence_mean=data.scaler_seq.mean_, sequence_scale=data.scaler_seq.scale_,
                        feature_names=data.feature_names.astype(str))
    print(frame)


def make_demo_dataset(seed=42):
    """Small synthetic fixture for software tests, NEVER scientific evidence."""
    rng = np.random.default_rng(seed)
    n_subjects, per_subject, n_features = 20, 10, 8
    n_samples = n_subjects * per_subject
    return Dataset(
        rng.normal(size=(n_samples, 3, n_features)).astype(np.float32),
        rng.normal(size=(n_samples, n_features)).astype(np.float32),
        rng.normal(size=(n_samples, 32, 4)).astype(np.float32),
        np.tile(np.arange(5, dtype=np.int64), n_samples // 5),
        np.repeat(np.arange(n_subjects), per_subject),
        np.array([f"synthetic_feature_{i}" for i in range(n_features)]),
    )
