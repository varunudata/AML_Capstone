"""Build the Phase 3 tables and figures from saved predictions and diagnostics."""

import argparse
from pathlib import Path

import pandas as pd

from . import calibration, compare, explain
from .compare import BOOST, DL_ROOT, read_json, settings, write_table
from .evaluate import PredictionKey, align_predictions, load_predictions


def leakage_comparison(output, frames, cfg):
    fixed = read_json(DL_ROOT / "results/aml_baseline_metrics.json")
    phase2 = read_json(DL_ROOT / "results/tables/phase2_final_results.json")
    dropped = fixed["xgboost_leakage_aware"]["dropped_features"]
    if set(dropped) != set(phase2["dl_mlp_leakaware"]["dropped_features"]):
        raise ValueError("AML and DL leakage ablations dropped different features")
    directory = DL_ROOT / "results/predictions"
    labels = cfg["evaluation"]["class_labels"]
    rows = []
    for split in ("val", "test"):
        aml_seed, aml_base = next(iter(frames[split][BOOST].items()))
        dl_seed, dl_base = next(iter(frames[split]["dl_mlp"].items()))
        aml_key = PredictionKey("aml_xgboost_leakage_aware", aml_seed, split)
        dl_key = PredictionKey("dl_mlp_leakaware", dl_seed, split)
        # Both ablations must predict the same rows as their full-feature comparator.
        align_predictions(aml_base, load_predictions(directory, aml_key))
        dl_ablated = align_predictions(dl_base, load_predictions(directory, dl_key))
        pairs = [
            (BOOST, aml_seed, fixed["xgboost"][split], fixed["xgboost_leakage_aware"][split]),
            ("dl_mlp", dl_seed, compare.dl_metrics(dl_base, labels),
             compare.dl_metrics(dl_ablated, labels)),
        ]
        for model, seed, base, ablated in pairs:
            for metric in compare.METRIC_COLUMNS:
                rows.append({"split": split, "model": model, "seed": seed, "metric": metric,
                             "full": base[metric], "ablated": ablated[metric],
                             "delta_ablated_minus_full": ablated[metric] - base[metric],
                             "dropped_features": ", ".join(dropped)})
    write_table(pd.DataFrame(rows), output, "phase3_leakage_ablation")


def run(output, *, clean_data=None, aml_split=None, aml_model=None, mlp_checkpoint=None):
    cfg = settings()
    frames, manifest, status = compare.run(
        output, metadata_path=clean_data, aml_split=aml_split, aml_model=aml_model,
        mlp_checkpoint=mlp_checkpoint, cfg=cfg
    )
    calibration.run(output, frames, cfg)
    explain.run(output, clean_data=clean_data, checkpoint=mlp_checkpoint, cfg=cfg,
                manifest=manifest)
    leakage_comparison(output, frames, cfg)
    return status


def add_arguments(parser):
    parser.add_argument("--output", type=Path, default=DL_ROOT / "results")
    parser.add_argument("--artifact-root", type=Path, help="Root of the extracted project ZIP")
    parser.add_argument("--aml-model", type=Path, help="Frozen XGBoost artifact for file size")
    parser.add_argument("--clean-data", type=Path, help="Frozen AML clean.parquet for sites/features")
    parser.add_argument("--aml-split", type=Path, help="Original site_held_out_split.npz to verify")
    parser.add_argument("--mlp-checkpoint", type=Path, help="Frozen seed-42 MLP .pt for explanation")
    return parser


def artifact_paths(root=None, *, clean_data=None, aml_split=None, aml_model=None, mlp_checkpoint=None):
    paths = {"clean_data": clean_data, "aml_split": aml_split, "aml_model": aml_model,
             "mlp_checkpoint": mlp_checkpoint}
    if root is not None:
        relative = {"clean_data": "data/processed/clean.parquet",
                    "aml_split": "data/processed/splits/site_held_out_split.npz",
                    "aml_model": "models/xgboost_production_model.joblib",
                    "mlp_checkpoint": "dl/results/models/dl_mlp_seed42.pt"}
        for name, filename in relative.items():
            paths[name] = paths[name] if paths[name] is not None else root / filename
    missing = [str(path) for path in paths.values() if path is not None and not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing frozen project artifacts: " + ", ".join(missing))
    return paths


def command_artifacts(args):
    return artifact_paths(args.artifact_root, clean_data=args.clean_data, aml_split=args.aml_split,
                          aml_model=args.aml_model, mlp_checkpoint=args.mlp_checkpoint)


if __name__ == "__main__":
    args = add_arguments(argparse.ArgumentParser(description=__doc__)).parse_args()
    run(args.output, **command_artifacts(args))
