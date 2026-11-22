import numpy as np
import pandas as pd
import pytest

try:
    from src.evaluate import (
        PredictionKey, align_predictions, check_manifest_rows, classification_metrics,
        load_predictions, probability_metrics, regression_metrics,
    )
except ImportError:
    from DL.src.evaluate import (
        PredictionKey, align_predictions, check_manifest_rows, classification_metrics,
        load_predictions, probability_metrics, regression_metrics,
    )


def test_regression_matches_hand_calculations():
    result = regression_metrics([1, 2, 3], [2, 2, 2])
    assert result == pytest.approx({"MAE": 2 / 3, "RMSE": np.sqrt(2 / 3), "R2": 0})
    assert np.isnan(regression_metrics([1], [1])["R2"])


def test_class_metrics_and_top_label_ece():
    result = classification_metrics(["a", "b", "b"], ["a", "a", "b"], ["a", "b"])
    assert result == pytest.approx({"Macro_F1": 2 / 3, "Recall_a": 1, "Recall_b": 0.5})
    assert probability_metrics(["a", "b"], [[1, 0], [.6, .4]], ["a", "b"], bins=5) == (
        pytest.approx({"Brier": .36, "ECE": .3})
    )


def test_alignment_requires_exact_rows_and_truth(tmp_path):
    key = PredictionKey("dl_mock", 42, "test")
    path = tmp_path / key.filename
    path.write_text("row_id,y_true,y_pred\n0002,2,3\n0001,1,0\n")
    mock = load_predictions(tmp_path, key)
    reference = mock.iloc[::-1]
    pd.testing.assert_frame_equal(align_predictions(reference, mock), reference)
    with pytest.raises(ValueError, match="y_true"):
        align_predictions(reference, mock.assign(y_true=0))
    with pytest.raises(ValueError, match="row IDs"):
        align_predictions(reference, mock.iloc[:1])
    with pytest.raises(ValueError, match="manifest"):
        check_manifest_rows(reference, {"test_sample_idx": [1, 2]}, "test")
    path.write_text("row_id,y_true,y_pred\n7,2,3\n7,1,0\n")
    with pytest.raises(ValueError, match="unique"):
        load_predictions(tmp_path, key)
