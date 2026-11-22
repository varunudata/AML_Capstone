"""Import frozen AML attributions and explain a matching saved DL checkpoint."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .compare import DL_ROOT, plt, read_json, save_figure, settings, write_table
from .evaluate import PredictionKey, check_manifest_rows, load_manifest, load_predictions


def frozen_aml_importance():
    aml = read_json(DL_ROOT.parent / "outputs/06_models_results.json")
    return pd.DataFrame(aml["top_permutation_features"]).rename(
        columns={"importance": "AML_R2_drop"}
    )


def load_preprocessor(path):
    from . import data as dl_data

    # Phase 1 saved the class as src.data from inside dl/. Keep its pickle unchanged.
    alias = "src.data"
    registered = alias not in sys.modules
    if registered:
        sys.modules[alias] = dl_data
    try:
        return dl_data.WaterQualityPreprocessor.load(path)
    finally:
        if registered:
            del sys.modules[alias]


def predict(model, features):
    import torch

    with torch.no_grad():
        return model(torch.as_tensor(features, dtype=torch.float32)).numpy().reshape(-1)


def clipped_model(checkpoint, n_features, cfg):
    import torch
    from .models import build_mlp_from_config

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    weights = state["hyperparameters"]
    model = build_mlp_from_config({"mlp": {**cfg["mlp"], **weights}}, n_features)
    model.load_state_dict(state["state_dict"])
    model.eval()
    bounds = cfg["evaluation"]["wqi_bounds"]

    class ClippedModel(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            prediction = self.inner(x).reshape(-1, 1)
            return prediction.clamp(*bounds) if cfg["mlp"]["clip_predictions"] else prediction

    return ClippedModel(model).eval()


def gradient_importance(model, test, background, names, seed):
    import shap
    import torch

    explainer = shap.GradientExplainer(model, torch.as_tensor(background, dtype=torch.float32))
    explained = explainer.shap_values(torch.as_tensor(test, dtype=torch.float32), rseed=seed)
    values = np.asarray(explained).reshape(len(test), len(names))
    return pd.DataFrame({"feature": names, "mean_abs_SHAP": np.abs(values).mean(axis=0)})


def dl_permutation_importance(model, test, target, names, seed):
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import r2_score

    class Predictor:
        def predict(self, features):
            return predict(model, features)

        def fit(self, features, target):
            return self

    ranking = permutation_importance(
        Predictor(), test, target, scoring=lambda est, x, y: r2_score(y, est.predict(x)),
        n_repeats=5, random_state=seed,
    )
    return pd.DataFrame({"feature": names, "DL_R2_drop": ranking.importances_mean,
                         "DL_R2_drop_SD": ranking.importances_std})


def dependence(model, test, names, top, rng, centers, scales, output):
    fig, axes = plt.subplots(1, len(top), figsize=(10.5, 3.7), layout="constrained")
    curves = []
    ice = rng.choice(len(test), min(32, len(test)), replace=False)
    for ax, feature in zip(np.atleast_1d(axes), top):
        i = names.index(feature)
        grid = np.unique(np.quantile(test[:, i], np.linspace(.05, .95, 21)))
        displayed = grid * scales[i] + centers[i]
        modified = test.copy()
        traces = []
        for value in grid:
            modified[:, i] = value
            traces.append(predict(model, modified))
        traces = np.column_stack(traces)
        means = traces.mean(axis=0)
        ax.plot(displayed, traces[ice].T, color="#65a3b9", alpha=.24, linewidth=.6)
        ax.plot(displayed, means, color="#084e65", linewidth=2)
        ax.set(xlabel=feature, ylabel="Predicted WQI")
        curves.extend({"feature": feature, "value": value, "PDP": mean}
                      for value, mean in zip(displayed, means))
    save_figure(fig, output, "phase3_mlp_pdp_ice")
    write_table(pd.DataFrame(curves), output, "phase3_mlp_partial_dependence")


def plot_importance(table, output, column, title, filename):
    rows = table.nlargest(10, column).sort_values(column)
    fig, ax = plt.subplots(figsize=(7.5, 4.4), layout="constrained")
    ax.barh(rows.feature, rows[column], color="#126782")
    ax.set_xlabel(title)
    save_figure(fig, output, filename)


def run(output, *, clean_data=None, checkpoint=None, cfg=None, manifest=None):
    cfg = cfg if cfg is not None else settings()
    aml = frozen_aml_importance()
    write_table(aml, output, "phase3_aml_frozen_importance")
    plot_importance(aml, output, "AML_R2_drop", "Frozen XGBoost permutation importance (R² decrease)",
                    "phase3_aml_frozen_importance")
    status = {"mlp_checkpoint_evaluated": False}
    if checkpoint is not None:
        if clean_data is None:
            raise ValueError("A saved checkpoint requires its matching frozen cleaned data")
        manifest = manifest if manifest is not None else load_manifest(
            DL_ROOT / cfg["data"]["split_indices_path"]
        )
        status = run_checkpoint(output, clean_data, checkpoint, cfg, manifest, aml)
    (output / "tables/phase3_explain_status.json").write_text(json.dumps(status, indent=2) + "\n")
    return status


def run_checkpoint(output, clean_data, checkpoint, cfg, manifest, aml):
    data, evaluation = cfg["data"], cfg["evaluation"]
    rng = np.random.default_rng(evaluation["seed"])
    target_column = data["target_name"]
    raw = pd.read_parquet(clean_data, columns=[*data["raw_features"], data["time_column"], target_column])
    preprocessor = load_preprocessor(DL_ROOT / data["preprocessor_path"])
    names = list(preprocessor.feature_names_)
    seed = cfg["mlp"]["seeds"][0]
    directory = DL_ROOT / "results/predictions"
    reference = load_predictions(directory, PredictionKey("dl_mlp", seed, "test"))
    check_manifest_rows(reference, manifest, "test")
    test_size = min(len(reference), evaluation["explain_test_size"])
    # The original AML permutation diagnostic used the first 4,000 test rows.
    test_rows = raw.iloc[manifest["test_sample_idx"][:test_size]]
    actual = test_rows[target_column].to_numpy()
    if not np.allclose(actual, reference.y_true.iloc[:test_size], atol=.00015, rtol=0):
        raise ValueError("Frozen feature rows disagree with the saved test target")
    test = preprocessor.transform(test_rows, scaled=True)
    model = clipped_model(checkpoint, len(names), cfg)
    if not np.allclose(predict(model, test), reference.y_pred.iloc[:test_size], atol=.002, rtol=0):
        raise ValueError(f"Checkpoint does not match saved seed {seed} predictions")
    shap_size = min(evaluation["shap_sample_size"], 500, len(test))
    train_ids = manifest["train_sample_idx"]
    background = preprocessor.transform(
        raw.iloc[rng.choice(train_ids, min(shap_size, len(train_ids)), replace=False)], scaled=True
    )
    explained = test[rng.choice(len(test), shap_size, replace=False)]
    attribution = gradient_importance(model, explained, background, names, evaluation["seed"])
    permutation = dl_permutation_importance(model, test, actual, names, evaluation["seed"])
    importance = attribution.merge(permutation, on="feature", validate="one_to_one")
    write_table(importance.sort_values("mean_abs_SHAP", ascending=False), output, "phase3_mlp_importance")
    plot_importance(importance, output, "mean_abs_SHAP", "MLP mean absolute GradientSHAP (WQI)",
                    "phase3_mlp_shap")
    plot_importance(importance, output, "DL_R2_drop", "MLP permutation importance (R² decrease)",
                    "phase3_mlp_permutation")
    matches = aml.merge(permutation, on="feature", validate="one_to_one")
    agreement = spearmanr(matches.AML_R2_drop, matches.DL_R2_drop).statistic
    write_table(matches, output, "phase3_permutation_overlap")
    top = attribution.nlargest(3, "mean_abs_SHAP").feature.tolist()
    # Add DO as a separate check when the learnt top three do not include it.
    dependence(model, explained, names, top, rng, preprocessor.scaler_.mean_,
               preprocessor.scaler_.scale_, output)
    do_name = "Dissolved Oxygen (mg/l)"
    oxygen = names.index(do_name)
    low, high = np.quantile(explained[:, oxygen], [.1, .9])
    lower, higher = explained.copy(), explained.copy()
    lower[:, oxygen], higher[:, oxygen] = low, high
    return {"mlp_checkpoint_evaluated": True, "seed": seed, "shap_rows": shap_size,
            "permutation_rows": len(test), "common_importance_features": len(matches),
            "permutation_spearman": float(agreement) if np.isfinite(agreement) else None,
            "do_low_mg_l": float(low * preprocessor.scaler_.scale_[oxygen]
                                 + preprocessor.scaler_.mean_[oxygen]),
            "do_high_mg_l": float(high * preprocessor.scaler_.scale_[oxygen]
                                  + preprocessor.scaler_.mean_[oxygen]),
            "do_shift_wqi": float(np.mean(predict(model, higher) - predict(model, lower)))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DL_ROOT / "results")
    parser.add_argument("--clean-data", type=Path)
    parser.add_argument("--mlp-checkpoint", type=Path)
    args = parser.parse_args()
    run(args.output, clean_data=args.clean_data, checkpoint=args.mlp_checkpoint)
