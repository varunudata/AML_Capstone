"""
Utility functions for DL Phase:
- Seed management across Python, NumPy, PyTorch
- Configuration loader with fallback search paths
- Standardized logging
"""

import os
import random
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import yaml

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def get_logger(name: str = "DL_Phase") -> logging.Logger:
    """Configure and return a standard logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def seed_everything(seed: int = 42) -> None:
    """
    Ensure 100% deterministic reproducibility across random, numpy, and torch.
    WHY: Fair comparison between AML baselines and DL MLP models requires strict determinism.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    Resolves configuration file paths relative to current script, workspace, or explicit argument.
    """
    if config_path is not None and Path(config_path).exists():
        target_path = Path(config_path)
    else:
        # Search sensible locations
        candidates = [
            Path("configs/config.yaml"),
            Path("dl/configs/config.yaml"),
            Path(__file__).resolve().parent.parent / "configs" / "config.yaml",
            Path(__file__).resolve().parent / "config.yaml"
        ]
        target_path = None
        for cand in candidates:
            if cand.exists():
                target_path = cand
                break
        if target_path is None:
            raise FileNotFoundError(f"Could not locate config.yaml in search paths: {candidates}")

    with open(target_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg
