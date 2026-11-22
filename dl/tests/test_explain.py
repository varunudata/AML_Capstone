import numpy as np
import pytest

from dl.src.compare import DL_ROOT
from dl.src.explain import dl_permutation_importance, gradient_importance, load_preprocessor


def test_frozen_preprocessor_loads_from_root():
    obj = load_preprocessor(DL_ROOT / "data/processed/preprocessor.joblib")
    assert len(obj.feature_names_) == obj.scaler_.n_features_in_


def test_dl_importance_recovers_the_used_feature():
    torch = pytest.importorskip("torch")
    pytest.importorskip("shap")
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[2., 0.]]))
    data = np.random.default_rng(42).normal(size=(50, 2)).astype("float32")
    names = ["used", "unused"]
    gradient = gradient_importance(model, data[:10], data[10:], names, 42)
    permutation = dl_permutation_importance(model, data, 2 * data[:, 0], names, 42)
    assert gradient.mean_abs_SHAP.iloc[0] > 0
    assert gradient.mean_abs_SHAP.iloc[1] == 0
    assert permutation.DL_R2_drop.iloc[0] > 0
    assert permutation.DL_R2_drop.iloc[1] == 0
