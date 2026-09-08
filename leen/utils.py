"""
Shared utilities for LEEN: metrics, config, seeding, logging.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.stats import pearsonr, spearmanr


# ──────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────

def compute_metrics(
    pred: np.ndarray, target: np.ndarray
) -> dict[str, float]:
    """Compute regression metrics between predictions and targets.

    Returns:
        Dict with spearman, pearson, rmse, mae.
    """
    mask = np.isfinite(pred) & np.isfinite(target)
    pred, target = pred[mask], target[mask]

    if len(pred) < 3:
        return {"spearman": 0.0, "pearson": 0.0, "rmse": float("inf"), "mae": float("inf")}

    sp = spearmanr(pred, target).correlation
    pc = pearsonr(pred, target)[0]
    rmse = float(np.sqrt(np.mean((pred - target) ** 2)))
    mae = float(np.mean(np.abs(pred - target)))

    return {
        "spearman": float(sp) if np.isfinite(sp) else 0.0,
        "pearson": float(pc) if np.isfinite(pc) else 0.0,
        "rmse": rmse,
        "mae": mae,
    }


def compute_antisymmetry_diagnostic(
    pred_forward: np.ndarray, pred_reverse: np.ndarray
) -> dict[str, float]:
    """Compute forward-reverse antisymmetry diagnostics.

    For LEEN, this should be exact (bias ≈ 0) by construction.
    Useful as a sanity check.
    """
    eps = pred_forward + pred_reverse  # should be ≈ 0
    return {
        "mean_bias": float(np.mean(eps)),
        "std_bias": float(np.std(eps)),
        "mean_abs_violation": float(np.mean(np.abs(eps))),
        "pcc_forward_neg_reverse": float(
            pearsonr(pred_forward, -pred_reverse)[0]
            if len(pred_forward) > 2 else 0.0
        ),
    }


# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

class Config:
    """Simple nested config from a YAML file."""

    def __init__(self, d: dict):
        for key, value in d.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def __repr__(self):
        return f"Config({self.__dict__})"

    def to_dict(self) -> dict:
        d = {}
        for k, v in self.__dict__.items():
            d[k] = v.to_dict() if isinstance(v, Config) else v
        return d


def load_config(path: str | Path) -> Config:
    """Load a YAML config file."""
    with open(path) as f:
        return Config(yaml.safe_load(f))


# ──────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────

def seed_everything(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ──────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────

def setup_logging(level: int = logging.INFO):
    """Configure basic logging."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
