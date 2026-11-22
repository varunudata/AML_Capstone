"""Evaluate exported intervals and calibrate a wrapper around saved AML predictions."""

import argparse
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd

from .compare import (
    BOOST, DL_MODELS, DL_ROOT, load_comparison, plt, save_figure, settings, write_table,
)
from .evaluate import load_manifest, probability_metrics, reliability_table


def interval_metrics(actual, lower, upper):
    y, lo, hi = (np.asarray(v, dtype=float) for v in (actual, lower, upper))
    if (y.ndim != 1 or y.shape != lo.shape or y.shape != hi.shape or not y.size
            or not np.isfinite([y, lo, hi]).all() or (lo > hi).any()):
        raise ValueError("Expected finite aligned observations and ordered interval bounds")
    return {"PICP": np.mean((lo <= y) & (y <= hi)), "MPIW": np.mean(hi - lo),
            "below": np.mean(y < lo), "above": np.mean(y > hi)}


def conformal_radius(actual, pred, coverage):
    """Finite-sample split-conformal order statistic, with an infinite sentinel."""
    y, point = (np.asarray(v, dtype=float) for v in (actual, pred))
    if (y.ndim != 1 or y.shape != point.shape or not y.size
            or not np.isfinite([y, point]).all() or not 0 < coverage < 1):
        raise ValueError("Expected finite validation vectors and coverage in (0, 1)")
    errors = np.abs(y - point)
    rank = ceil((len(errors) + 1) * coverage)
    return float(np.partition(errors, rank - 1)[rank - 1]) if rank <= len(errors) else np.inf


def quantile_check(actual, quantiles):
    """Use strict and inclusive frequencies to show ties at the WQI ceiling."""
    y = np.asarray(actual, dtype=float)
    return [
        {"quantile": level, "strict": np.mean(y < value), "inclusive": np.mean(y <= value)}
        for level, value in sorted(quantiles.items())
    ]


def conformal_bounds(point, radius, bounds):
    return np.clip(point - radius, *bounds), np.clip(point + radius, *bounds)


def plot_calibration(curve, quantiles, output):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.1), layout="constrained")
    for ax in axes:
        ax.plot([0, 1], [0, 1], color="#616a70", linestyle="--", linewidth=1)
        ax.set(xlim=(0, 1.02), ylim=(0, 1.02))
    ax = axes[0]
    xgb = curve[curve.model == "aml_xgboost_conformal"]
    ax.plot(xgb.nominal, xgb.PICP, marker="o", color="#126782", label="XGBoost conformal")
    dl = curve[curve.model != "aml_xgboost_conformal"]
    for (model, rows), marker in zip(dl.groupby("model", sort=False), ("s", "D", "^")):
        ax.scatter(rows.nominal.iloc[0], rows.PICP.mean(), marker=marker,
                   label=model.removeprefix("dl_"))
    ax.set(xlabel="Nominal interval coverage", ylabel="Observed interval coverage")
    ax.legend(fontsize=7, loc="upper left")
    ax = axes[1]
    ax.fill_between(quantiles["quantile"], quantiles.strict, quantiles.inclusive,
                    color="#126782", alpha=.20, label="Range from ties")
    ax.plot(quantiles["quantile"], quantiles.inclusive, marker="o", color="#126782", label="P(Y ≤ q)")
    ax.plot(quantiles["quantile"], quantiles.strict, marker="o", color="#b45b23", label="P(Y < q)")
    ax.set(xlabel="Nominal quantile", ylabel="Observed fraction")
    ax.legend(fontsize=8, loc="upper left")
    save_figure(fig, output, "phase3_calibration")


def export_class_calibration(tests, labels, output):
    """Class probabilities are required, including one column for every tier."""
    probability_columns = [f"p_class_{i}" for i in range(len(labels))]
    rows = []
    for model, runs in tests.items():
        if model.startswith("aml_"):
            continue
        for seed, frame in runs.items():
            supplied = set(probability_columns).intersection(frame.columns)
            if not supplied:
                continue
            if len(supplied) != len(labels) or "class_true" not in frame:
                raise ValueError(f"{model} seed {seed} exports incomplete class probabilities")
            probs = frame[probability_columns]
            table = reliability_table(frame.class_true, probs, labels)
            rows.append({"model": model, "seed": seed,
                         **probability_metrics(frame.class_true, probs, labels)})
            fig, ax = plt.subplots(figsize=(5, 4), layout="constrained")
            ax.plot([0, 1], [0, 1], "--", color="#616a70")
            ax.plot(table.confidence, table.accuracy, "o-", color="#126782")
            ax.set(xlim=(0, 1.02), ylim=(0, 1.02), xlabel="Top class confidence",
                   ylabel="Observed accuracy")
            save_figure(fig, output, f"phase3_reliability_{model}_seed{seed}")
    if rows:
        write_table(pd.DataFrame(rows), output, "phase3_class_calibration")


def run(output, frames=None, cfg=None):
    cfg = cfg if cfg is not None else settings()
    if frames is None:
        manifest = load_manifest(DL_ROOT / cfg["data"]["split_indices_path"])
        frames = {split: load_comparison(cfg, split, manifest) for split in ("val", "test")}
    val = next(iter(frames["val"][BOOST].values()))
    test = next(iter(frames["test"][BOOST].values()))
    bounds = cfg["evaluation"]["wqi_bounds"]
    levels = cfg["evaluation"]["interval_levels"]
    nominal = cfg["evaluation"]["interval_coverage"]
    dl_rows, quantile_rows = [], []
    for model in DL_MODELS[1:]:
        for seed, frame in frames["test"][model].items():
            lower, upper = (pd.to_numeric(frame[name], errors="raise") for name in ("q05", "q95"))
            stats = interval_metrics(frame.y_true, lower, upper)
            dl_rows.append({"model": model, "seed": seed, "nominal": nominal, **stats})
            if model == "dl_mlp_quantile":
                quantile_rows.extend({"seed": seed, **row} for row in quantile_check(
                    frame.y_true, {.05: lower, .50: frame.y_pred, .95: upper}
                ))
    seed = next(iter(frames["val"][BOOST]))
    xgb_rows = []
    for level in sorted(set([nominal, *levels])):
        radius = conformal_radius(val.y_true, val.y_pred, level)
        lower, upper = conformal_bounds(test.y_pred, radius, bounds)
        xgb_rows.append({"model": "aml_xgboost_conformal", "seed": seed, "nominal": level,
                         **interval_metrics(test.y_true, lower, upper), "radius": radius})
    curve = pd.DataFrame([*xgb_rows, *dl_rows])
    summary = []
    for model, rows in curve[curve.nominal == nominal].groupby("model", sort=False):
        summary.append({"model": model, "runs": len(rows), "nominal": nominal,
                        "PICP": rows.PICP.mean(), "MPIW": rows.MPIW.mean(),
                        "PICP_SD": rows.PICP.std(), "MPIW_SD": rows.MPIW.std(),
                        "below": rows.below.mean(), "above": rows.above.mean()})
    quantiles = pd.DataFrame(quantile_rows)
    write_table(pd.DataFrame(summary), output, "phase3_intervals")
    write_table(curve, output, "phase3_interval_curve")
    write_table(quantiles, output, "phase3_quantile_calibration")
    plot_calibration(curve, quantiles.groupby("quantile", as_index=False)[["strict", "inclusive"]].mean(),
                     output)
    export_class_calibration(frames["test"], cfg["evaluation"]["class_labels"], output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DL_ROOT / "results")
    run(parser.parse_args().output)
