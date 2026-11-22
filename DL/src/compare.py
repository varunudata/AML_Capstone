"""Compare DL runs against saved AML results and row-matched residuals."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .evaluate import (
    METRICS, PredictionKey, align_predictions, check_manifest_rows,
    classification_metrics, load_manifest, load_predictions, regression_metrics,
)

DL_ROOT = Path(__file__).resolve().parents[1]
AML_MODELS = ("aml_xgboost", "aml_random_forest")
DL_MODELS = ("dl_mlp", "dl_mlp_mcdropout", "dl_mlp_ensemble", "dl_mlp_quantile")
BOOST = AML_MODELS[0]
ENSEMBLE = DL_MODELS[2]
METRIC_COLUMNS = (*METRICS, "Macro_F1")


def settings():
    import yaml
    return yaml.safe_load((DL_ROOT / "configs/config.yaml").read_text())


def read_json(path):
    return json.loads(path.read_text())


def run_keys(cfg):
    aml_seed = cfg["aml_baselines"]["seed"]
    seeds = tuple(cfg["mlp"]["seeds"])
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("DL seeds must be nonempty and unique")
    return {
        **{model: (aml_seed,) for model in AML_MODELS},
        **{model: (seeds[0],) if model == ENSEMBLE else seeds for model in DL_MODELS},
    }


def load_comparison(cfg, split, manifest):
    directory = DL_ROOT / "results/predictions"
    keys = run_keys(cfg)
    reference = load_predictions(directory, PredictionKey(BOOST, keys[BOOST][0], split))
    check_manifest_rows(reference, manifest, split)
    frames = {}
    for model, seeds in keys.items():
        frames[model] = {
            seed: align_predictions(
                reference, load_predictions(directory, PredictionKey(model, seed, split))
            )
            for seed in seeds
        }
    for alias in cfg["mlp"]["seeds"][1:]:
        path = directory / PredictionKey(ENSEMBLE, alias, split).filename
        if path.exists():
            duplicate = align_predictions(
                reference, load_predictions(directory, PredictionKey(ENSEMBLE, alias, split))
            )
            if not duplicate.equals(next(iter(frames[ENSEMBLE].values()))):
                raise ValueError(f"{path.name} differs from the canonical ensemble")
    return frames


def archived_split_matches(path, manifest, seed):
    """Reapply only the original AML sampling rule to the supplied frozen indices."""
    rng = np.random.RandomState(seed)
    with np.load(path, allow_pickle=False) as archive:
        for split in ("train", "val", "test"):
            positions = archive[f"{split}_idx"]
            expected = manifest[f"{split}_sample_idx"]
            sampled = rng.choice(positions, size=len(expected), replace=False)
            if not np.array_equal(sampled, expected):
                raise ValueError(f"Original AML archive disagrees with the {split} row manifest")
    return True


def load_metadata(path, cfg, manifest, frames):
    """Select frozen data by positional ID, the convention used by the AML pipeline."""
    data = cfg["data"]
    schema = pd.read_parquet(path, columns=[data["target_name"], *data["raw_features"],
                                            data["group_column"], data["time_column"]])
    group = data["group_column"]
    assigned_groups = {}
    for split in ("train", "val", "test"):
        positions = manifest[f"{split}_sample_idx"]
        frame = schema.iloc[positions].copy()
        frame.index = pd.Index([str(pos) for pos in positions], name="row_id", dtype="string")
        if frame[group].isna().any():
            raise ValueError(f"Missing site IDs in {split} metadata")
        assigned_groups[split] = set(frame[group])
        if split in frames:
            target = next(iter(frames[split][BOOST].values())).y_true
            # The frozen CSVs round float32 predictions and labels to four decimal places.
            if not np.allclose(frame[data["target_name"]], target, atol=.00015, rtol=0):
                raise ValueError(f"Frozen cleaned data disagrees with {split} prediction targets")
    if any(assigned_groups[first] & assigned_groups[second]
           for first, second in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("Frozen cleaned data assigns a site to multiple partitions")
    return frame


def dl_metrics(frame, labels):
    scores = regression_metrics(frame.y_true, frame.y_pred)
    if "class_true" in frame and "class_pred" in frame:
        scores.update(classification_metrics(frame.class_true, frame.class_pred, labels))
    return scores


def paired_bootstrap(actual, baseline, alternatives, groups, *, draws, seed):
    """Use common resamples and compare mean seed RMSE against one fixed AML model."""
    actual = np.asarray(actual, dtype=float)
    errors = (np.column_stack([baseline, *alternatives]) - actual[:, None]) ** 2
    counts = np.ones(len(actual), dtype=int)
    if groups is not None:
        codes, names = pd.factorize(np.asarray(groups))
        if len(codes) != len(actual) or (codes < 0).any() or len(names) < 2:
            raise ValueError("Bootstrap needs complete group labels and at least two groups")
        sums = np.zeros((len(names), errors.shape[1]))
        np.add.at(sums, codes, errors)
        errors = sums
        counts = np.bincount(codes)
    rng = np.random.default_rng(seed)
    samples = np.empty(draws)
    for i in range(draws):
        positions = rng.integers(len(counts), size=len(counts))
        rmse = np.sqrt(errors[positions].sum(axis=0) / counts[positions].sum())
        samples[i] = rmse[1:].mean() - rmse[0]
    point = np.sqrt(errors.sum(axis=0) / counts.sum())
    lower, upper = np.quantile(samples, [.025, .975])
    return {"delta_rmse": point[1:].mean() - point[0], "ci_low": lower, "ci_high": upper}


def write_table(table, output, name):
    directory = output / "tables"
    directory.mkdir(parents=True, exist_ok=True)
    table.to_csv(directory / f"{name}.csv", index=False, float_format="%.8g")
    text = table.astype(object).fillna("n/a").to_markdown(index=False, floatfmt=".4f")
    (directory / f"{name}.md").write_text(text + "\n")


def save_figure(fig, output, name):
    directory = output / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{name}.png", dpi=300, bbox_inches="tight",
                metadata={"Software": "AML Capstone Phase 3"})
    plt.close(fig)


def subgroup_rows(frame, model, seed, grouping, values):
    actual, pred = frame.y_true, frame.y_pred
    error = pred - actual
    group_mean = actual.groupby(values, observed=True).transform("mean")
    scored = pd.DataFrame({"group": values, "ae": abs(error), "se": error**2,
                           "st": (actual - group_mean)**2})
    stats = scored.groupby("group", observed=True).agg(
        n=("ae", "size"), MAE=("ae", "mean"), sse=("se", "sum"), sst=("st", "sum")
    ).reset_index()
    stats["RMSE"] = np.sqrt(stats.sse / stats.n)
    stats["R2"] = 1 - stats.sse / stats.sst.where(stats.sst > 0)
    return stats.drop(columns=["sse", "sst"]).assign(model=model, seed=seed, grouping=grouping)


def subgroup_table(frames, metadata, cfg):
    reference = next(iter(frames[BOOST].values()))
    groups = {"CCME band": reference.class_true}
    if metadata is not None:
        data = cfg["data"]
        groups["Site"] = metadata[data["group_column"]]
        groups["Year"] = pd.to_datetime(metadata[data["time_column"]], errors="coerce").dt.year
        for feature in data["raw_features"]:
            values = pd.to_numeric(metadata[feature], errors="coerce")
            groups[feature] = pd.qcut(values, 4, duplicates="drop").astype("string")
        groups = {name: val.astype("string").fillna("Unavailable") for name, val in groups.items()}
    stats = pd.concat([
        subgroup_rows(frame, model, seed, name, values)
        for model, models in frames.items()
        for seed, frame in models.items()
        for name, values in groups.items()
    ], ignore_index=True)
    return stats[["model", "seed", "grouping", "group", "n", *METRICS]]


def plot_residuals(frames, output, seed):
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.5), sharex=True, sharey=True,
                             layout="constrained")
    rng = np.random.default_rng(seed)
    for ax, (model, models) in zip(axes.flat, frames.items()):
        frame = next(iter(models.values()))
        sample = frame.iloc[rng.choice(len(frame), min(len(frame), 4_000), replace=False)]
        ax.scatter(sample.y_true, sample.y_pred - sample.y_true, s=3, alpha=.22, color="#126782")
        ax.axhline(0, color="#424a52", linewidth=.8)
        ax.set_title(f"{model} (seed {next(iter(models))})" if model != ENSEMBLE else model)
    fig.supxlabel("Observed CCME WQI")
    fig.supylabel("Predicted minus observed WQI")
    save_figure(fig, output, "phase3_residuals")


def resources(aml_model=None, mlp_checkpoint=None):
    frozen = read_json(DL_ROOT / "results/tables/phase2_final_results.json")
    rows = [{"model": model} for model in AML_MODELS]
    for model, item in zip((*DL_MODELS, "dl_mlp_leakaware"), frozen["summary_table"], strict=True):
        if not item["Variant"].startswith(model + " "):
            raise ValueError("Unexpected Phase 2 resource summary order")
        rows.append({"model": model, "parameters": item["Params"],
                     "train_seconds": item["Train_Time_s"],
                     "test_inference_seconds": item["Infer_Time_s"]})
    if aml_model is not None:
        rows[0]["checkpoint_bytes"] = aml_model.stat().st_size
    if mlp_checkpoint is not None:
        rows[len(AML_MODELS)]["checkpoint_bytes"] = mlp_checkpoint.stat().st_size
    return pd.DataFrame(rows)


def run(output, *, metadata_path=None, aml_split=None, aml_model=None,
        mlp_checkpoint=None, cfg=None):
    cfg = cfg if cfg is not None else settings()
    eval_cfg = cfg["evaluation"]
    manifest = load_manifest(DL_ROOT / cfg["data"]["split_indices_path"])
    frames = {split: load_comparison(cfg, split, manifest) for split in ("val", "test")}
    archived_split = aml_split is not None and archived_split_matches(
        aml_split, manifest, cfg["data"]["split_seed"]
    )
    metadata = load_metadata(metadata_path, cfg, manifest, frames) if metadata_path else None
    fixed = read_json(DL_ROOT / "results/aml_baseline_metrics.json")
    from .utils import resolve_path
    original = pd.read_csv(resolve_path("outputs/06_models_metrics.csv"))
    frozen_names = ["4. Random Forest Regressor", "6. XGBoost (Boundary-Clipped [0, 100])"]
    recap = original.set_index("Model").loc[frozen_names].reset_index()
    write_table(recap.drop(columns=["Test RMSE 95% CI"]), output, "phase3_aml_stage6_recap")
    summary, per_seed = [], []
    for split, split_frames in frames.items():
        for model, models in split_frames.items():
            if model in AML_MODELS:
                scores = fixed[model.removeprefix("aml_")][split]
                summary.append({"split": split, "model": model, "runs": 1,
                                "source": "Frozen AML bridge", **scores})
                continue
            stats = pd.DataFrame([
                {"seed": seed, **dl_metrics(frame, eval_cfg["class_labels"])}
                for seed, frame in models.items()
            ])
            per_seed.extend(stats.assign(split=split, model=model).to_dict("records"))
            record = {"split": split, "model": model, "runs": len(stats), "source": "DL CSV"}
            for metric in METRIC_COLUMNS:
                if metric in stats:
                    record[metric] = stats[metric].mean()
                    if len(stats) > 1:
                        record[f"{metric}_SD"] = stats[metric].std()
            summary.append(record)
    write_table(pd.DataFrame(summary), output, "phase3_comparison")
    write_table(pd.DataFrame(per_seed), output, "phase3_dl_per_seed")
    write_table(resources(aml_model, mlp_checkpoint), output, "phase3_resources")

    tests = frames["test"]
    baseline = next(iter(tests[BOOST].values()))
    groups = None if metadata is None else metadata[cfg["data"]["group_column"]]
    units = "row (exploratory)" if groups is None else cfg["data"]["group_column"]
    ci = [
        {"model": model, "reference": BOOST, "unit": units,
         **paired_bootstrap(baseline.y_true, baseline.y_pred, [f.y_pred for f in tests[model].values()],
                            groups, draws=eval_cfg["bootstrap_draws"], seed=eval_cfg["seed"])}
        for model in DL_MODELS
    ]
    write_table(pd.DataFrame(ci), output, "phase3_paired_bootstrap")
    subgroups = subgroup_table(tests, metadata, cfg)
    write_table(subgroups[subgroups.grouping != "Site"], output, "phase3_subgroups")
    if metadata is not None:
        sites = subgroups[subgroups.grouping == "Site"].groupby(["model", "group"], as_index=False)
        sites = sites.agg(n=("n", "first"), runs=("seed", "size"),
                          **{name: (name, "mean") for name in METRICS})
        write_table(sites, output, "phase3_sites")
    plot_residuals(tests, output, eval_cfg["seed"])
    status = {"original_aml_indices_verified": archived_split, "metadata_available": metadata is not None,
              "bootstrap_unit": units, "test_sites": None if groups is None else groups.nunique(),
              "manifest_counts": {s: len(manifest[f"{s}_sample_idx"])
                                                            for s in ("train", "val", "test")}}
    (output / "tables/phase3_provenance.json").write_text(json.dumps(status, indent=2) + "\n")
    return frames, manifest, status


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DL_ROOT / "results")
    parser.add_argument("--clean-data", type=Path)
    parser.add_argument("--aml-split", type=Path)
    args = parser.parse_args()
    run(args.output, metadata_path=args.clean_data, aml_split=args.aml_split)
