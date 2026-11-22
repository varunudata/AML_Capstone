"""
Configurable Multi-Layer Perceptron (MLP) Architecture for Water Quality Prediction (Member 2).
Implements Task A:
- Configurable input_dim, hidden_dims list, activation (ReLU, GELU, SiLU), dropout p, optional BatchNorm.
- Kaiming / He weight initialization for deep non-linear stability.
- Regression head (1 output for continuous CCME_Values, or 3 outputs for quantile regression q05/q50/q95).
- Optional classification head ONLY when a class scheme is explicitly enabled in config,
  paired with a weighted multi-task loss (MSE/Huber + CrossEntropy).
- Small, heavily regularized tabular design to prevent overfitting on 21 physicochemical features.
"""

from typing import List, Optional, Tuple, Union, Dict, Any
import torch
import torch.nn as nn


def get_activation(name: str) -> nn.Module:
    """
    Factory for non-linear activation functions.
    WHY: ReLU provides sharp piecewise linear boundaries similar to tree splits,
    while GELU and SiLU provide smooth, differentiable curvature for interacting chemical sensors.
    """
    act = name.strip().lower()
    if act == "relu":
        return nn.ReLU()
    if act == "gelu":
        return nn.GELU()
    if act in ("silu", "swish"):
        return nn.SiLU()
    raise ValueError(f"Unsupported activation '{name}'. Choose from ['relu', 'gelu', 'silu'].")


class WaterQualityMLP(nn.Module):
    """
    Small, regularized Multi-Layer Perceptron for tabular surface water quality prediction.
    Supports:
    - Standard point regression (output_dim=1)
    - Quantile regression (output_dim=3 for q05, q50, q95)
    - Optional auxiliary classification head (num_classes=5 for CCME_WQI tiers)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        activation: str = "relu",
        dropout: float = 0.1,
        batch_norm: bool = False,
        output_dim: int = 1,
        num_classes: Optional[int] = None,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one hidden layer dimension.")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0.0, 1.0), got {dropout}")

        self.input_dim = input_dim
        self.hidden_dims = list(hidden_dims)
        self.activation_name = activation.lower()
        self.dropout_p = float(dropout)
        self.batch_norm = bool(batch_norm)
        self.output_dim = int(output_dim)
        self.num_classes = int(num_classes) if (num_classes is not None and num_classes > 0) else None

        layers: List[nn.Module] = []
        in_features = self.input_dim

        for h_dim in self.hidden_dims:
            layers.append(nn.Linear(in_features, h_dim))
            if self.batch_norm:
                # WHY: BatchNorm stabilizes internal covariate shift across heterogeneous chemical scales
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(get_activation(self.activation_name))
            if self.dropout_p > 0.0:
                # WHY: Dropout acts as stochastic ensemble regularization and enables MC Dropout at test time
                layers.append(nn.Dropout(p=self.dropout_p))
            in_features = h_dim

        self.backbone = nn.Sequential(*layers)
        self.reg_head = nn.Linear(in_features, self.output_dim)

        # Optional classification head ONLY if explicitly requested
        if self.num_classes is not None:
            self.cls_head: Optional[nn.Linear] = nn.Linear(in_features, self.num_classes)
        else:
            self.cls_head = None

        self._init_weights()

    def _init_weights(self) -> None:
        """
        Apply Kaiming (He) normal initialization to all linear layers.
        WHY: He initialization preserves forward activation variance and backward gradient variance
        through ReLU/GELU/SiLU layers, preventing early gradient explosion or dead neurons.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract latent representation from the MLP backbone."""
        return self.backbone(x)

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass:
        - When classification head is disabled (default):
            returns Tensor of shape (batch_size,) if output_dim == 1,
            or (batch_size, output_dim) if output_dim > 1.
        - When classification head is enabled:
            returns Tuple (reg_output, cls_logits).
        """
        feats = self.backbone(x)
        reg_out = self.reg_head(feats)
        if self.output_dim == 1:
            reg_out = reg_out.squeeze(-1)

        if self.cls_head is not None:
            cls_logits = self.cls_head(feats)
            return reg_out, cls_logits

        return reg_out


class MultiTaskWaterQualityLoss(nn.Module):
    """
    Weighted Multi-Task Loss combining continuous WQI regression (MSE or Huber)
    and optional discrete WQI tier classification (CrossEntropyLoss).
    WHY: Auxiliary classification supervision encourages latent features to respect
    regulatory tier boundaries while optimizing continuous regression precision.
    """

    def __init__(
        self,
        loss_type: str = "mse",
        huber_delta: float = 1.0,
        reg_weight: float = 1.0,
        cls_weight: float = 0.2,
    ) -> None:
        super().__init__()
        lt = loss_type.strip().lower()
        if lt == "mse":
            self.reg_loss_fn: nn.Module = nn.MSELoss()
        elif lt == "huber":
            self.reg_loss_fn = nn.HuberLoss(delta=huber_delta)
        else:
            raise ValueError(f"Unsupported regression loss_type '{loss_type}'. Choose 'mse' or 'huber'.")

        self.cls_loss_fn = nn.CrossEntropyLoss()
        self.reg_weight = float(reg_weight)
        self.cls_weight = float(cls_weight)

    def forward(
        self,
        outputs: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        y_reg: torch.Tensor,
        y_cls: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute single-task regression loss or weighted multi-task regression + classification loss."""
        if isinstance(outputs, tuple):
            reg_pred, cls_logits = outputs
            reg_loss = self.reg_loss_fn(reg_pred.view(-1), y_reg.view(-1))
            if y_cls is not None:
                cls_loss = self.cls_loss_fn(cls_logits, y_cls.long())
                return self.reg_weight * reg_loss + self.cls_weight * cls_loss
            return self.reg_weight * reg_loss

        return self.reg_loss_fn(outputs.view(-1), y_reg.view(-1))


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in the PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_mlp_from_config(
    cfg: Dict[str, Any],
    input_dim: int,
    output_dim: int = 1,
) -> WaterQualityMLP:
    """
    Instantiate a WaterQualityMLP directly from the project's config dictionary.
    Checks whether an explicit classification head is requested in `cfg["mlp"]`.
    """
    mlp_cfg = cfg.get("mlp", {})
    hidden_dims = mlp_cfg.get("hidden_dims", [128, 64, 32])
    activation = mlp_cfg.get("activation", "relu")
    dropout = mlp_cfg.get("dropout", 0.1)
    batch_norm = mlp_cfg.get("batch_norm", False)

    enable_cls = mlp_cfg.get("enable_classification_head", False)
    num_classes = mlp_cfg.get("num_classes", 5) if enable_cls else None

    return WaterQualityMLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        activation=activation,
        dropout=dropout,
        batch_norm=batch_norm,
        output_dim=output_dim,
        num_classes=num_classes,
    )
