"""
Utility functions for DL Phase:
- Deterministic seed management across Python, NumPy, PyTorch
- Configuration loader with fallback search paths
- Robust workspace path resolution
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

DL_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = DL_ROOT.parent


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


def resolve_path(path_str: str) -> Path:
    """
    Resolve a file path relative to DL_ROOT, REPO_ROOT, or REPO_ROOT/AML without hardcoded absolute paths.
    WHY: Enables seamless execution whether commands run from AML_Capstone/, AML/, DL/, or notebooks/.
    """
    p = Path(path_str)
    if p.is_absolute() and p.exists():
        return p
    if (DL_ROOT / p).exists():
        return (DL_ROOT / p).resolve()
    if (REPO_ROOT / p).exists():
        return (REPO_ROOT / p).resolve()
    if (REPO_ROOT / "AML" / p).exists():
        return (REPO_ROOT / "AML" / p).resolve()
    # Handle '../data/...' when inside DL/ and data is at REPO_ROOT/data or REPO_ROOT/AML/data
    stripped = str(path_str).lstrip("./")
    if (REPO_ROOT / stripped).exists():
        return (REPO_ROOT / stripped).resolve()
    if (REPO_ROOT / "AML" / stripped).exists():
        return (REPO_ROOT / "AML" / stripped).resolve()
    if p.exists():
        return p.resolve()
    return (DL_ROOT / p).resolve()


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    Resolves configuration file paths relative to current script, workspace, or explicit argument.
    """
    if config_path is not None:
        target_path = resolve_path(config_path)
    else:
        candidates = [
            DL_ROOT / "configs" / "config.yaml",
            REPO_ROOT / "configs" / "config.yaml",
            Path("configs/config.yaml"),
            Path("dl/configs/config.yaml")
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
