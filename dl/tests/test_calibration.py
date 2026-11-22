import numpy as np
import pytest

from dl.src.calibration import conformal_radius, interval_metrics, quantile_check


def test_order_statistic_uses_only_validation_and_includes_infinite_case():
    assert conformal_radius([0, 0, 0, 0], [4, 1, 3, 2], .6) == 3
    assert np.isinf(conformal_radius([0], [1], .9))


def test_interval_metrics_include_endpoints_and_reject_crossing():
    expected = {"PICP": .5, "MPIW": 1.5, "below": 0, "above": .5}
    assert interval_metrics([1, 4], [1, 1], [2, 3]) == pytest.approx(expected)
    with pytest.raises(ValueError, match="ordered"):
        interval_metrics([1], [2], [1])


def test_quantile_check_preserves_point_mass_ties():
    assert quantile_check([1, 100], {.95: np.array([2, 100])}) == [
        {"quantile": .95, "strict": .5, "inclusive": 1}
    ]
