"""Read Phase 1/2 exports and compute metrics for new DL evaluations."""

from dataclasses import dataclass
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd


METRICS = ("MAE", "RMSE", "R2")
PREDICTION_NAME = re.compile(r"(aml|dl)_[a-z0-9_]+_seed\d+_(val|test)")


@dataclass(frozen=True)
class PredictionKey:
    model: str
    seed: int
    split: str

    @property
    def filename(self) -> str:
        stem = f"{self.model}_seed{self.seed}_{self.split}"
        if not PREDICTION_NAME.fullmatch(stem):
            raise ValueError(f"Invalid prediction key: {stem}")
        return f"{stem}.csv"


def load_predictions(directory: Path, key: PredictionKey) -> pd.DataFrame:
    path = directory / key.filename
    frame = pd.read_csv(path, dtype={"row_id": "string"}, keep_default_na=False)
    required = {"row_id", "y_true", "y_pred"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
    if frame.empty or frame.row_id.str.strip().eq("").any() or not frame.row_id.is_unique:
        raise ValueError(f"{path.name}: expected nonempty and unique row IDs")
    for column in ("y_true", "y_pred"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(frame[column]).all():
            raise ValueError(f"{path.name}: non-finite {column}")
    return frame.set_index("row_id")


def align_predictions(reference: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    """Join by ID and fail on dropped rows, added rows, or a different target."""
    if (not reference.index.is_unique or not other.index.is_unique
            or len(reference) != len(other) or not other.index.isin(reference.index).all()):
        raise ValueError("Predictions must contain the same unique row IDs")
    aligned = other.reindex(reference.index)
    if not np.array_equal(reference.y_true.to_numpy(), aligned.y_true.to_numpy()):
        raise ValueError("Predictions disagree on y_true after aligning row IDs")
    if "class_true" in reference and "class_true" in aligned:
        if not np.array_equal(reference.class_true.to_numpy(), aligned.class_true.to_numpy()):
            raise ValueError("Predictions disagree on class_true after aligning row IDs")
    return aligned


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text())
    seen = set()
    for split in ("train", "val", "test"):
        ids = manifest[f"{split}_sample_idx"]
        unique = set(ids)
        if not ids or any(type(row) is not int or row < 0 for row in ids):
            raise ValueError(f"Invalid row positions in {split} manifest")
        if len(ids) != len(unique) or seen.intersection(unique):
            raise ValueError(f"Duplicate or overlapping row positions in {split} manifest")
        seen.update(unique)
    return manifest


def check_manifest_rows(frame: pd.DataFrame, manifest: dict, split: str) -> None:
    expected = pd.Index([str(row) for row in manifest[f"{split}_sample_idx"]], dtype=frame.index.dtype)
    if not frame.index.equals(expected):
        raise ValueError(f"Reference predictions differ from ordered {split} manifest")


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    actual = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if (actual.ndim != 1 or actual.shape != pred.shape or not actual.size
            or not np.isfinite(actual).all() or not np.isfinite(pred).all()):
        raise ValueError("Expected equal, nonempty, finite regression vectors")
    error = pred - actual
    sse = np.sum(error ** 2)
    sst = np.sum((actual - actual.mean()) ** 2)
    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(sse / actual.size)),
        "R2": float(1 - sse / sst) if sst > 0 else float("nan"),
    }


def classification_metrics(y_true, y_pred, labels) -> dict[str, float]:
    """Evaluate an explicit class scheme. Absent classes have undefined recall."""
    actual = np.asarray(y_true)
    pred = np.asarray(y_pred)
    labels = tuple(labels)
    if (actual.ndim != 1 or actual.shape != pred.shape or not actual.size
            or not labels or len(set(labels)) != len(labels)
            or not np.isin(actual, labels).all() or not np.isin(pred, labels).all()):
        raise ValueError("Expected equal class vectors using the configured labels")
    scores = {}
    f1 = []
    for label in labels:
        true = actual == label
        predicted = pred == label
        tp = np.count_nonzero(true & predicted)
        support = np.count_nonzero(true)
        retrieved = np.count_nonzero(predicted)
        scores[f"Recall_{label}"] = tp / support if support else float("nan")
        if support + retrieved:
            f1.append(2 * tp / (support + retrieved))
    return {"Macro_F1": float(np.mean(f1)), **scores}


def reliability_table(y_true, probabilities, labels, bins=10) -> pd.DataFrame:
    """Top-label multiclass reliability including the exact 0 and 1 endpoints."""
    actual, probs, labels = _class_probabilities(y_true, probabilities, labels)
    if bins < 1:
        raise ValueError("Expected at least one reliability bin")
    confidence = probs.max(axis=1)
    correct = labels[probs.argmax(axis=1)] == actual
    bin_id = np.minimum((confidence * bins).astype(int), bins - 1)
    data = pd.DataFrame({"bin": bin_id, "confidence": confidence, "accuracy": correct})
    return data.groupby("bin", as_index=False).agg(
        n=("accuracy", "size"), confidence=("confidence", "mean"), accuracy=("accuracy", "mean")
    )


def _class_probabilities(y_true, probabilities, labels):
    actual = np.asarray(y_true)
    labels = np.asarray(labels)
    probs = np.asarray(probabilities, dtype=float)
    if (actual.ndim != 1 or labels.ndim != 1 or len(labels) < 2
            or len(set(labels)) != len(labels) or not actual.size
            or probs.shape != (actual.size, labels.size)
            or not np.isin(actual, labels).all() or not np.isfinite(probs).all()
            or (probs < 0).any() or (probs > 1).any()
            or not np.allclose(probs.sum(axis=1), 1, atol=1e-6, rtol=0)):
        raise ValueError("Expected probabilities summing to one in configured class order")
    return actual, probs, labels


def probability_metrics(y_true, probabilities, labels, bins=10) -> dict[str, float]:
    """Brier uses the sum over classes per observation. ECE uses top confidence."""
    actual, probs, labels = _class_probabilities(y_true, probabilities, labels)
    table = reliability_table(actual, probs, labels, bins)
    return {
        "Brier": float(np.mean(np.sum((probs - (actual[:, None] == labels)) ** 2, axis=1))),
        "ECE": float(np.average(abs(table.confidence - table.accuracy), weights=table.n)),
    }
