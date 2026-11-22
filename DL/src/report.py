"""Rebuild the capstone Markdown and PDF from Phase 3 outputs."""

import argparse
from html import escape
import os
from pathlib import Path
import re

import pandas as pd
from reportlab import rl_config
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Table, TableStyle

from .compare import BOOST, DL_ROOT, ENSEMBLE, read_json, settings
from .reporting import add_arguments, command_artifacts, run


TITLE = "Water Quality Prediction via Gradient Boosting and Deep Neural Networks"
NAMES = {
    BOOST: "XGBoost",
    "aml_random_forest": "Random Forest",
    "dl_mlp": "Point MLP",
    "dl_mlp_mcdropout": "MC dropout",
    ENSEMBLE: "Deep ensemble",
    "dl_mlp_quantile": "Quantile MLP",
    "aml_xgboost_conformal": "XGBoost conformal",
}
INK = colors.HexColor("#182735")
TEAL = colors.HexColor("#0D6476")
MUTED = colors.HexColor("#435867")
PALE = colors.HexColor("#EAF3F5")


def mean_sd(row, metric, places=4):
    sd = row.get(f"{metric}_SD")
    mean = f"{row[metric]:.{places}f}"
    return mean if pd.isna(sd) else f"{mean} ± {sd:.{places}f}"


def make_markdown(output, reports):
    def table(name):
        return pd.read_csv(output / "tables" / f"phase3_{name}.csv")

    def figure(name, label):
        path = output / "figures" / f"phase3_{name}.png"
        return f"![{label}]({os.path.relpath(path, reports)})"

    provenance = read_json(output / "tables/phase3_provenance.json")
    explanation = read_json(output / "tables/phase3_explain_status.json")
    aml = read_json(DL_ROOT / "results/aml_baseline_metrics.json")
    phase2 = read_json(DL_ROOT / "results/tables/phase2_final_results.json")
    manifest = provenance["manifest_counts"]
    hparams = phase2["best_hyperparameters_used"]
    comparison = table("comparison").query("split == 'test'").set_index("model")
    intervals = table("intervals").set_index("model")
    bootstrap = table("paired_bootstrap").set_index("model")
    resources = table("resources").set_index("model")
    subgroups = table("subgroups").query("`grouping` == 'CCME band'")
    ablation = table("leakage_ablation").query("split == 'test' and metric == 'RMSE'").set_index("model")
    recap = table("aml_stage6_recap").set_index("Model")
    original_xgb = recap.loc["6. XGBoost (Boundary-Clipped [0, 100])", "Test RMSE"]
    original_rf = recap.loc["4. Random Forest Regressor", "Test RMSE"]
    ensemble = comparison.loc[ENSEMBLE]
    boost, forest = comparison.loc[BOOST], comparison.loc["aml_random_forest"]
    ens_ci = bootstrap.loc[ENSEMBLE]
    archive_status = ("The original AML split archive reproduced all three sampled partitions "
                      "in order." if provenance["original_aml_indices_verified"] else
                      "The original AML split archive is absent from the checkout. Full AML "
                      "split provenance remains independently unverified.")
    block_status = (f"The bootstrap resampled {provenance['test_sites']:,} sites with replacement, "
                    "retaining all rows belonging to each drawn site."
                    if provenance["metadata_available"] else
                    "The available prediction exports contain no site IDs. The bootstrap resampled "
                    "individual rows, so the confidence interval is exploratory and may be too narrow "
                    "if errors cluster within sites.")
    feature_status = ("The generated tables include site, year and feature-range breakdowns."
                      if provenance["metadata_available"] else
                      "Site, year and feature-range breakdowns require the original cleaned data.")
    top_aml = table("aml_frozen_importance").iloc[0].feature
    seed = int(ablation.loc["dl_mlp", "seed"])
    boost_abl = ablation.loc[BOOST]
    mlp_abl = ablation.loc["dl_mlp"]
    dropped_features = [f"`{name}`" for name in boost_abl.dropped_features.split(", ")]
    test_seed_ensemble = int(phase2["dl_mlp_ensemble"]["n_members"])
    draw_count = settings()["evaluation"]["bootstrap_draws"]
    summaries = [
        "| Model | Runs | MAE | RMSE | R² | Macro-F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model, row in comparison.iterrows():
        values = [NAMES[model], str(int(row.runs))]
        values.extend(mean_sd(row, m) for m in ("MAE", "RMSE", "R2", "Macro_F1"))
        summaries.append("| " + " | ".join(values) + " |")
    checkpoint_text = (
        f"The supplied XGBoost artifact is {int(resources.loc[BOOST, 'checkpoint_bytes']):,} bytes "
        f"and the seed-{seed} MLP checkpoint is {int(resources.loc['dl_mlp', 'checkpoint_bytes']):,} bytes. "
        "The artifacts use different formats. Random Forest size and comparable AML timings "
        "are absent from the frozen inputs."
        if "checkpoint_bytes" in resources
        and pd.notna(resources.loc[[BOOST, "dl_mlp"], "checkpoint_bytes"]).all() else
        "Comparable model sizes and elapsed times for AML are absent from the frozen metrics."
    )
    interval_rows = [
        "| Model | Runs | Coverage | Mean width | Below / above |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for model, row in intervals.iterrows():
        coverage = f"{row.PICP:.2%}"
        if pd.notna(row.PICP_SD):
            coverage += f" ± {100 * row.PICP_SD:.2f} pp"
        misses = f"{row.below:.2%} / {row.above:.2%}"
        interval_rows.append(f"| {NAMES[model]} | {int(row.runs)} | {coverage} | "
                             f"{mean_sd(row, 'MPIW', 3)} | {misses} |")

    if explanation["mlp_checkpoint_evaluated"]:
        top_mlp = table("mlp_importance").iloc[0].feature
        explain_sections = [
            f"The saved seed-{seed} MLP ranked `{top_mlp}` first by absolute GradientSHAP "
            f"over {explanation['shap_rows']} test rows. Spearman rank agreement with XGBoost "
            f"was {explanation['permutation_spearman']:.2f} across the "
            f"{explanation['common_importance_features']} features in the archived AML permutation "
            "ranking. Correlated and engineered features can divide importance across representations.",
            figure("mlp_shap", "Mean absolute GradientSHAP for the saved point MLP"),
            f"As a chemistry check, dissolved oxygen was changed from "
            f"{explanation['do_low_mg_l']:.2f} to {explanation['do_high_mg_l']:.2f} mg/L and its "
            f"engineered features recomputed. Mean predicted WQI shifted by "
            f"{explanation['do_shift_wqi']:+.3f} points. The magnitude and direction describe the "
            "fitted model over this selected slice. Oxygen changes in a real waterbody can "
            "coincide with temperature and other chemistry changes.",
            "<!-- pagebreak -->",
            "## MLP partial dependence and individual responses",
            figure("mlp_pdp_ice", "Top three MLP features. The dark line is the average prediction."),
            "These plots vary one model input at a time. Because the inputs include transformed "
            "versions of the same chemistry measurements, this can create feature combinations "
            "outside the observed range of joint chemistry.",
        ]
    else:
        explain_sections = [
            "The MLP GradientSHAP, permutation ranking and PDP/ICE comparison require the "
            "matching saved checkpoint and cleaned inputs."
        ]
    report = [
        f"# {TITLE}",
        "AML + Deep Learning course capstone · Phase 3 evaluation",
        f"**Finding.** Random Forest and XGBoost had lower test RMSE than every DL variant on "
        f"the matched exports. The deep ensemble led the DL variants at {ensemble.RMSE:.4f} WQI "
        f"points. XGBoost scored {boost.RMSE:.4f} and Random Forest scored {forest.RMSE:.4f}.",
        "## 1. Problem and target definition",
        "The regression target is the supplied continuous CCME Water Quality Index "
        "(`CCME_Values`). The CCME index combines the scope, frequency and amplitude of "
        "water-quality guideline exceedances on a 0–100 scale. The project also reports "
        "hard CCME tiers derived from those scores for secondary classification metrics. "
        "[CCME WQI User's Manual](https://ccme.ca/en/res/wqimanualen.pdf)",
        "## 2. Data and missing-data treatment",
        "The AML scope is ambient surface water from 1990 onward. The pipeline retains chemistry "
        "units from the cleaned data, with concentration measurements in mg/L, temperature in "
        "degrees Celsius and pH on its original scale. The DL bridge uses eight chemistry "
        "measurements and 13 engineered variables. Reported preprocessing substitutes half the "
        "detection limit for censored readings, sets impossible values to missing, and fits "
        "median imputation and standard scaling on the training rows. Trees use unscaled inputs.",
        "## 3. Frozen AML baseline",
        f"The original Stage 6 table reports test RMSE {original_xgb:.4f} for boundary-clipped "
        f"XGBoost and {original_rf:.4f} for Random Forest. The Phase 1 bridge published separate "
        f"results of {boost.RMSE:.4f} and {forest.RMSE:.4f}. The results below use the bridge "
        "because it includes row IDs for matching to DL. The Stage 6 and bridge metrics are "
        "quoted from their respective saved outputs.",
        "## 4. MLP design",
        f"The final point MLP uses {', '.join(map(str, hparams['hidden_dims']))} hidden units, "
        f"{hparams['activation'].upper()} activations, batch normalization, dropout "
        f"{hparams['dropout']:.1f}, Huber loss and AdamW with early stopping. "
        f"DL tuning used {hparams['tuning']['n_trials']} Optuna trials. The frozen AML bridge "
        f"records {aml['meta']['tuning_budget']}. The trial counts match and the search "
        "objectives, candidate training budgets and validation procedures differ.",
        "## 5. Experimental protocol",
        f"The Phase 1 manifest records the AML site-held-out policy (`Area`, seed "
        f"{aml['meta']['split_seed']}). It contains {manifest['train']:,} training, "
        f"{manifest['val']:,} validation and {manifest['test']:,} test row positions. "
        f"All compared saved predictions have the same respective validation and test row IDs "
        f"and target values. {archive_status}",
        "<!-- pagebreak -->",
        "## 6. Point prediction and residuals",
        "\n".join(summaries),
        f"WQI errors are in index points. The DL mean ± sample SD reflects separate runs. The "
        f"deep ensemble is one model built from {test_seed_ensemble} member seeds. AML metrics "
        f"are imported. The {draw_count:,}-resample paired bootstrap estimated ensemble minus "
        f"XGBoost RMSE at {ens_ci.delta_rmse:+.4f} (95% CI [{ens_ci.ci_low:+.4f}, "
        f"{ens_ci.ci_high:+.4f}]). {block_status}",
        figure("residuals", "Test residuals by model, shown for the first seed or the single ensemble"),
        f"The saved XGBoost residuals had RMSE "
        f"{subgroups[(subgroups.model == BOOST) & (subgroups.group == 'Poor')].RMSE.iloc[0]:.3f} "
        f"in the Poor tier and "
        f"{subgroups[(subgroups.model == BOOST) & (subgroups.group == 'Excellent')].RMSE.iloc[0]:.3f} "
        f"in the Excellent tier. {feature_status}",
        f"Phase 2 logged {int(resources.loc['dl_mlp', 'parameters']):,} parameters, "
        f"{resources.loc['dl_mlp', 'train_seconds']:.2f} s to train and "
        f"{resources.loc['dl_mlp', 'test_inference_seconds']:.4f} s to score the test set for "
        f"the point MLP. The ensemble used {int(resources.loc[ENSEMBLE, 'parameters']):,} parameters. "
        f"{checkpoint_text}",
        "<!-- pagebreak -->",
        "## 7. Predictive uncertainty",
        "All methods target 90% marginal coverage. PICP is the observed fraction inside the "
        "interval. Mean interval width is measured in WQI points. Below and above give the "
        "two miss rates.",
        "\n".join(interval_rows),
        "The XGBoost wrapper takes the finite-sample order statistic of absolute saved validation "
        "residuals, applies that single radius to the saved test predictions and clips the "
        "bounds to the WQI scale. The validation split also served model selection or early "
        "stopping. Site dependence and validation reuse weaken the assumptions for a formal "
        "coverage guarantee here. [Angelopoulos and Bates](https://arxiv.org/abs/2107.07511)",
        figure("calibration", "Interval coverage and empirical quantile calibration"),
        "The ensemble came closest to 90% among DL models. MC dropout and the quantile MLP "
        "both exceeded nominal coverage with much wider intervals. The XGBoost wrapper had "
        "the narrowest mean interval. Ties at WQI 100 make strict and inclusive quantile "
        "calibration frequencies diverge. The exports contain hard tier labels, so probability "
        "calibration metrics require additional probability exports.",
        "<!-- pagebreak -->",
        "## 8. Explainability",
        f"The frozen XGBoost evaluation contains TreeSHAP and permutation importance. "
        f"{top_aml} led its permutation ranking. This is consistent with sensitivity to nutrient "
        "guidelines in the index. The archived ranking describes the fitted model and cannot "
        "establish a causal chemistry effect.",
        figure("aml_frozen_importance", "Top saved XGBoost permutation importances"),
        *explain_sections,
        "## 9. Feature ablation and limitations",
        f"At seed {seed} the matched ablations removed "
        f"{', '.join(dropped_features)}. "
        f"XGBoost RMSE rose from {boost_abl.full:.4f} to {boost_abl.ablated:.4f} "
        f"({boost_abl.delta_ablated_minus_full:+.4f}). MLP RMSE rose from "
        f"{mlp_abl.full:.4f} to {mlp_abl.ablated:.4f} "
        f"({mlp_abl.delta_ablated_minus_full:+.4f}). The remaining chemistry can encode "
        "inputs used to construct WQI. Removing engineered proxies measures sensitivity to those "
        "proxies and leaves possible formula leakage.",
        "The study holds out sites using the original AML protocol. It has no forward-in-time holdout, "
        "so future-period performance and geographic transfer beyond the sampled sites remain untested. "
        "Different preprocessing implementations and the heavy mass at WQI 100 also limit interpretation.",
        "## 10. Conclusion",
        "The MLP did not improve point prediction over either frozen tree baseline. The deep "
        "ensemble offers the closest DL coverage to the nominal level and the lowest DL point "
        "error. The conformal wrapper on XGBoost was narrower at similar observed coverage. "
        "These results describe reconstruction of the supplied CCME index on the matched exports.",
    ]
    return "\n\n".join(report) + "\n"


def markup(text):
    escaped = escape(text)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"`([^`]+)`", r"<font name='Courier'>\1</font>", escaped)
    return re.sub(r"\[([^\]]+)\]\((https://[^ )]+)\)",
                  r"<link href='\2' color='#0D6476'><u>\1</u></link>", escaped)


def make_pdf(md, reports):
    width = A4[0] - 100
    styles = {
        "title": ParagraphStyle("Title", fontName="Helvetica-Bold", fontSize=26,
                                leading=30, textColor=INK, spaceAfter=12),
        "heading": ParagraphStyle("Section", fontName="Helvetica-Bold", fontSize=13,
                                  leading=16, textColor=TEAL, spaceBefore=9, spaceAfter=5,
                                  keepWithNext=True),
        "body": ParagraphStyle("Body", fontName="Helvetica", fontSize=10.2,
                               leading=14.2, textColor=INK, spaceAfter=7, alignment=TA_LEFT),
        "small": ParagraphStyle("Caption", fontName="Helvetica", fontSize=8.2,
                                leading=10.4, textColor=MUTED, spaceAfter=6),
    }

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(TEAL)
        canvas.setLineWidth(.6)
        canvas.line(50, 48, A4[0] - 50, 48)
        canvas.setFillColor(MUTED)
        canvas.setFont("Helvetica", 8)
        canvas.drawString(50, 34, "AML + Deep Learning Capstone")
        canvas.drawRightString(A4[0] - 50, 34, f"{doc.page}")
        canvas.restoreState()

    story = []
    for block in md.strip().split("\n\n"):
        if block == "<!-- pagebreak -->":
            story.append(PageBreak())
        elif block.startswith("# "):
            story.append(Paragraph(markup(block[2:]), styles["title"]))
        elif block.startswith("## "):
            story.append(Paragraph(markup(block[3:]), styles["heading"]))
        elif block.startswith("|"):
            rows = [[part.strip() for part in line.strip("|").split("|")]
                    for line in block.splitlines() if not re.fullmatch(r"[| :\-]+", line)]
            col_widths = [width * .25, *([width * .75 / (len(rows[0]) - 1)] * (len(rows[0]) - 1))]
            grid = Table(rows, colWidths=col_widths, repeatRows=1, hAlign="LEFT", spaceAfter=10)
            grid.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), TEAL),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("TEXTCOLOR", (0, 1), (-1, -1), INK),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.6),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [PALE, colors.white]),
                ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]))
            story.append(grid)
        elif block.startswith("!["):
            caption, path = re.fullmatch(r"!\[([^\]]+)\]\((.+)\)", block).groups()
            full = reports / path
            w, h = ImageReader(str(full)).getSize()
            scale = min(width / w, 245 / h)
            story.extend([Image(str(full), width=w * scale, height=h * scale),
                          Paragraph(markup(caption), styles["small"])])
        else:
            story.append(Paragraph(markup(block), styles["body"]))
    doc = SimpleDocTemplate(str(reports / "DL_Report.pdf"), pagesize=A4, title=TITLE,
                            author="AML + Deep Learning Capstone Team", leftMargin=50,
                            rightMargin=50, topMargin=38, bottomMargin=59)
    ascii85 = rl_config.useA85
    rl_config.useA85 = 0
    try:
        doc.build(story, onFirstPage=footer, onLaterPages=footer,
                  canvasmaker=lambda *a, **k: Canvas(*a, **{**k, "invariant": 1}))
    finally:
        rl_config.useA85 = ascii85


if __name__ == "__main__":
    parser = add_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--reports", type=Path, default=DL_ROOT / "reports")
    args = parser.parse_args()
    run(args.output, **command_artifacts(args))
    args.reports.mkdir(parents=True, exist_ok=True)
    md = make_markdown(args.output, args.reports)
    (args.reports / "DL_Report.md").write_text(md, encoding="utf-8")
    make_pdf(md, args.reports)
