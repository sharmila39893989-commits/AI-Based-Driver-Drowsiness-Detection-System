"""Compatibility exports for the shared model-training utilities."""

from backend.src.training import (
    BACKEND_ROOT,
    DATASET_ROOT,
    MODEL_ROOT,
    build_binary_cnn,
    compute_binary_metrics,
    make_generators,
    save_model,
    train_binary_classifier,
    validate_binary_dataset,
)

ROOT = BACKEND_ROOT

__all__ = [
    "BACKEND_ROOT",
    "DATASET_ROOT",
    "MODEL_ROOT",
    "ROOT",
    "build_binary_cnn",
    "compute_binary_metrics",
    "make_generators",
    "save_model",
    "train_binary_classifier",
    "validate_binary_dataset",
]
