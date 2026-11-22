"""
Unit tests for Task A (dl/src/models.py):
- Verifies configurable MLP forward shapes on mock data from load_splits(cfg, mock=True).
- Tests all activations (ReLU, GELU, SiLU), BatchNorm on/off, and Dropout behavior.
- Verifies Kaiming/He weight initialization (zero biases, finite non-zero weights).
- Tests optional classification head and weighted MultiTaskWaterQualityLoss backward pass.
- Verifies parameter count remains small and regularized (<100k parameters).
"""

import pytest
import torch
import torch.nn as nn

from src.utils import load_config, seed_everything
from src.data import load_splits, get_dataloaders
from src.models import (
    WaterQualityMLP,
    MultiTaskWaterQualityLoss,
    count_parameters,
    build_mlp_from_config,
)


def test_mlp_forward_shape_on_mock_data():
    """Verify forward pass output shapes match mock dataset batch dimensions."""
    seed_everything(42)
    cfg = load_config()
    splits = load_splits(cfg, scaled=True, mock=True)
    x_val = torch.from_numpy(splits["X_val"])
    input_dim = x_val.shape[1]

    model = build_mlp_from_config(cfg, input_dim=input_dim)
    model.eval()
    with torch.no_grad():
        preds = model(x_val)

    assert isinstance(preds, torch.Tensor)
    assert preds.shape == (len(x_val),)
    assert preds.dtype == torch.float32
    assert not torch.isnan(preds).any()


@pytest.mark.parametrize("activation", ["relu", "gelu", "silu"])
@pytest.mark.parametrize("batch_norm", [False, True])
def test_mlp_activations_and_batchnorm(activation: str, batch_norm: bool):
    """Verify all supported activations and BatchNorm options forward and backprop cleanly."""
    seed_everything(42)
    cfg = load_config()
    train_loader, _, _ = get_dataloaders(cfg, batch_size=16, mock=True)
    bx, by = next(iter(train_loader))

    model = WaterQualityMLP(
        input_dim=bx.shape[1],
        hidden_dims=[64, 32],
        activation=activation,
        dropout=0.2,
        batch_norm=batch_norm,
    )
    model.train()
    out = model(bx)
    assert out.shape == (16,)

    loss = nn.MSELoss()(out, by)
    loss.backward()

    # Check gradients exist and are finite
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Missing gradient for {name}"
        assert torch.isfinite(param.grad).all(), f"Non-finite gradient in {name}"


def test_he_initialization_properties():
    """Verify Kaiming/He initialization sets biases to zero and weights to finite non-zero values."""
    seed_everything(42)
    model = WaterQualityMLP(input_dim=21, hidden_dims=[128, 64, 32], batch_norm=True)

    for module in model.modules():
        if isinstance(module, nn.Linear):
            assert torch.allclose(module.bias, torch.zeros_like(module.bias))
            assert module.weight.abs().mean().item() > 0.01
            assert torch.isfinite(module.weight).all()
        elif isinstance(module, nn.BatchNorm1d):
            assert torch.allclose(module.weight, torch.ones_like(module.weight))
            assert torch.allclose(module.bias, torch.zeros_like(module.bias))


def test_optional_classification_head_and_multitask_loss():
    """Verify dual-head forward pass and weighted multi-task loss computation."""
    seed_everything(42)
    bx = torch.randn(24, 21)
    by_reg = torch.rand(24) * 60.0 + 40.0
    by_cls = torch.randint(0, 5, (24,))

    model = WaterQualityMLP(
        input_dim=21,
        hidden_dims=[64, 32],
        activation="gelu",
        dropout=0.1,
        batch_norm=True,
        num_classes=5,
    )
    outputs = model(bx)
    assert isinstance(outputs, tuple)
    reg_out, cls_logits = outputs
    assert reg_out.shape == (24,)
    assert cls_logits.shape == (24, 5)

    criterion = MultiTaskWaterQualityLoss(loss_type="huber", reg_weight=1.0, cls_weight=0.5)
    loss = criterion(outputs, by_reg, by_cls)
    assert loss.ndim == 0
    assert loss.item() > 0.0
    loss.backward()


def test_quantile_head_output_shape():
    """Verify 3-quantile output head produces (batch_size, 3) predictions."""
    seed_everything(42)
    bx = torch.randn(16, 21)
    model = WaterQualityMLP(input_dim=21, hidden_dims=[64, 32], output_dim=3)
    out = model(bx)
    assert out.shape == (16, 3)


def test_model_is_small_and_regularized():
    """Verify default MLP remains compact (<50,000 parameters) per project brief."""
    cfg = load_config()
    model = build_mlp_from_config(cfg, input_dim=21)
    n_params = count_parameters(model)
    assert 1000 < n_params < 50000, f"Expected small tabular MLP, got {n_params} params"
