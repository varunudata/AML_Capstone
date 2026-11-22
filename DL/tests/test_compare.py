import numpy as np
import pytest

try:
    from src.compare import archived_split_matches, paired_bootstrap
except ImportError:
    from DL.src.compare import archived_split_matches, paired_bootstrap


@pytest.mark.parametrize("groups", [None, ["a", "a", "b"]])
def test_paired_bootstrap_uses_common_resamples(groups):
    result = paired_bootstrap([0, 0, 0], [0, 0, 0], [[1, 1, 1], [3, 3, 3]],
                              groups, draws=20, seed=42)
    assert result == {"delta_rmse": 2, "ci_low": 2, "ci_high": 2}


def test_frozen_archive_matches_in_order(tmp_path):
    path = tmp_path / "split.npz"
    arrays = {f"{split}_idx": np.arange(start, start + 5)
              for split, start in zip(("train", "val", "test"), (0, 5, 10))}
    np.savez(path, **arrays)
    rng = np.random.RandomState(42)
    manifest = {key.replace("_idx", "_sample_idx"): rng.choice(v, 3, replace=False).tolist()
                for key, v in arrays.items()}
    assert archived_split_matches(path, manifest, 42)
    manifest["test_sample_idx"].reverse()
    with pytest.raises(ValueError, match="test row manifest"):
        archived_split_matches(path, manifest, 42)
