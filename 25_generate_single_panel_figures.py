from pathlib import Path
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import patches
from scipy.stats import spearmanr


ROOT = Path.cwd()
OUT_DIR = ROOT / "publication_figures_single_nature"
PREVIEW_DIR = OUT_DIR / "previews"
TABLE_DIR = OUT_DIR / "tables"
CM_TO_IN = 1 / 2.54
PANEL_SIZE_CM = 6.0
PANEL_SIZE_IN = PANEL_SIZE_CM * CM_TO_IN

COLORS = {
    "blue": "#3C5488",
    "red": "#E64B35",
    "green": "#00A087",
    "gold": "#F39B7F",
    "slate": "#4D4D4D",
    "light_blue": "#91B5D8",
    "light_red": "#F1B6A8",
    "gray": "#8A8A8A",
    "grid": "#E6E6E6",
    "black": "#1A1A1A",
}

CELL_LABELS = {
    "FCGR3A+ Monocytes": "FCGR3A Mono",
    "Dendritic cells": "DC",
    "CD14+ Monocytes": "CD14 Mono",
    "B cells": "B",
    "CD4 T cells": "CD4 T",
    "NK cells": "NK",
    "CD8 T cells": "CD8 T",
}


mpl.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 13,
        "axes.titlesize": 13,
        "axes.labelsize": 13,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
        "figure.titlesize": 13,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)

sns.set_theme(style="ticks")


def ensure_dirs():
    for path in [OUT_DIR, PREVIEW_DIR, TABLE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def save_figure(fig, stem: str):
    pdf_path = OUT_DIR / f"{stem}.pdf"
    png_path = PREVIEW_DIR / f"{stem}.png"
    try:
        fig.tight_layout(pad=0.2)
    except Exception:
        pass
    fig.savefig(pdf_path, transparent=False)
    fig.savefig(png_path, dpi=260, transparent=False)
    plt.close(fig)


def add_axis_caption(ax, title: str, subtitle: Optional[str] = None):
    ax.set_title(title, loc="left", pad=4, fontweight="bold", color=COLORS["black"], fontsize=12)
    if subtitle:
        ax.text(0.0, 0.985, subtitle, transform=ax.transAxes, ha="left", va="top", color=COLORS["gray"], fontsize=9.5)


def style_axes(ax, add_ygrid: bool = False):
    sns.despine(ax=ax, top=True, right=True)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.spines["left"].set_color(COLORS["black"])
    ax.spines["bottom"].set_color(COLORS["black"])
    ax.tick_params(length=3, width=0.8, colors=COLORS["black"])
    ax.xaxis.label.set_color(COLORS["black"])
    ax.yaxis.label.set_color(COLORS["black"])
    if add_ygrid:
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.6)
    else:
        ax.grid(False)


def draw_rounded_box(ax, xywh, text, edgecolor, facecolor="white", fontsize=14, weight=None):
    x, y, w, h = xywh
    rect = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.2,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, fontweight=weight)


def short_perturbation_label(value: str) -> str:
    left = str(value).split("__")[0]
    return left.replace("_", "+").replace("TGFBR2", "TGFBR2").replace("UBASH3B", "UBASH3B")


def shorten_cell_labels(df: pd.DataFrame, column: str = "cell_type") -> pd.DataFrame:
    out = df.copy()
    out[column] = out[column].map(lambda x: CELL_LABELS.get(x, x))
    return out


def load_sender_data():
    pred = pd.read_csv(
        Path(r"E:\扰动\GSE133344_outputs\gnn_graph\gnn_training\gnn_predicted_expression_shifts.csv"),
        index_col=0,
    )
    obs = pd.read_csv(Path(r"E:\扰动\GSE133344_outputs\GSE133344_expression_shifts.csv"), index_col=0)
    gene_map = pd.read_csv(
        Path(r"E:\扰动\GSE133344\GSE133344_filtered_genes.tsv.gz"),
        sep="\t",
        header=None,
        names=["ensembl_id", "gene_symbol"],
    )
    mapper = gene_map.drop_duplicates("ensembl_id").set_index("ensembl_id")["gene_symbol"].astype(str).to_dict()
    pred.columns = [mapper.get(c, c) for c in pred.columns]
    obs.columns = [mapper.get(c, c) for c in obs.columns]
    pred = pred.T.groupby(level=0).mean().T
    obs = obs.T.groupby(level=0).mean().T

    quality = pd.read_csv(Path(r"E:\扰动\GSE133344_outputs\gnn_graph\gnn_training\gnn_prediction_quality_by_perturbation.csv"))
    quality_summary = pd.read_csv(Path(r"E:\扰动\GSE133344_outputs\gnn_graph\gnn_training\gnn_prediction_quality_summary.csv"))
    split_metrics = pd.read_csv(ROOT / "sender_split_stability_outputs" / "tables" / "sender_split_metrics.csv")
    knn = pd.read_csv(ROOT / "knn_baseline_comparison_outputs" / "tables" / "knn_vs_gnn_sender_prediction.csv")
    knn_test = pd.read_csv(ROOT / "commbio_strengthening_outputs" / "tables" / "sender_gnn_vs_knn_exact_test.csv")
    diag = pd.read_csv(ROOT / "sender_receiver_link_outputs" / "tables" / "sender_matrix_diagnostics.csv")
    return pred, obs, quality, quality_summary, split_metrics, knn, knn_test, diag


def load_receiver_data():
    ifn_perf = pd.read_csv(ROOT / "gse96583_receiver_outputs" / "tables" / "receiver_condition_model_performance.csv")
    lps_perf = pd.read_csv(ROOT / "gse226488_lps_receiver_outputs" / "tables" / "receiver_condition_model_performance.csv")
    overlap = pd.read_csv(ROOT / "gse226488_lps_receiver_outputs" / "tables" / "ifn_vs_lps_signature_overlap.csv")
    ifn_sig = pd.read_csv(ROOT / "gse96583_receiver_outputs" / "tables" / "receiver_response_signatures.csv")
    lps_sig = pd.read_csv(ROOT / "gse226488_lps_receiver_outputs" / "tables" / "receiver_response_signatures.csv")
    return ifn_perf, lps_perf, overlap, ifn_sig, lps_sig


def load_linking_data():
    concord = pd.read_csv(ROOT / "commbio_strengthening_outputs" / "tables" / "predicted_vs_observed_concordance_with_permutation.csv")
    hybrid_tests = pd.read_csv(ROOT / "commbio_strengthening_outputs" / "tables" / "hybrid_vs_baselines_exact_tests.csv")
    ifn_rank = pd.read_csv(ROOT / "sender_receiver_link_outputs" / "tables" / "ifn_predicted_hybrid_global_ranking.csv")
    lps_rank = pd.read_csv(ROOT / "sender_receiver_link_outputs" / "tables" / "lps_predicted_hybrid_global_ranking.csv")
    contrastive_ifn = pd.read_csv(ROOT / "sender_receiver_link_outputs" / "tables" / "ifn_predicted_contrastive_global_ranking.csv")
    contrastive_lps = pd.read_csv(ROOT / "sender_receiver_link_outputs" / "tables" / "lps_predicted_contrastive_global_ranking.csv")
    strict = pd.read_csv(ROOT / "receptor_focused_support_outputs" / "tables" / "strict_coverage_summary.csv")
    cross = pd.read_csv(ROOT / "context_sensitive_linking_outputs" / "tables" / "contrastive_top_cell_type_crosstab.csv", index_col=0)
    return concord, hybrid_tests, ifn_rank, lps_rank, contrastive_ifn, contrastive_lps, strict, cross


def build_signature_heatmap(sig: pd.DataFrame, top_n: int = 6) -> pd.DataFrame:
    up = sig[sig["direction"] == "up_in_stim"].copy()
    top = up.groupby("cell_type").head(top_n)
    genes = list(dict.fromkeys(top["gene"].tolist()))
    rows = []
    for cell_type, sub in up.groupby("cell_type"):
        gene_to_val = sub.set_index("gene")["delta_log1p"].to_dict()
        row = {"cell_type": cell_type}
        for gene in genes:
            row[gene] = gene_to_val.get(gene, 0.0)
        rows.append(row)
    return pd.DataFrame(rows).set_index("cell_type")


def fig_ax():
    fig, ax = plt.subplots(figsize=(PANEL_SIZE_IN, PANEL_SIZE_IN))
    return fig, ax


def plot_workflow():
    fig, ax = fig_ax()
    ax.axis("off")
    add_axis_caption(ax, "Workflow")
    draw_rounded_box(ax, (0.07, 0.67, 0.25, 0.13), "Sender\nGSE133344", COLORS["blue"], facecolor="#F7F9FD", fontsize=9.0)
    draw_rounded_box(ax, (0.07, 0.22, 0.25, 0.13), "Prior\nLIANA+CPDB", COLORS["green"], facecolor="#F3FBF8", fontsize=9.0)
    draw_rounded_box(ax, (0.40, 0.42, 0.22, 0.18), "Linking\nhybrid +\ncontrastive", COLORS["slate"], facecolor="#FAFAFA", fontsize=9.0, weight="bold")
    draw_rounded_box(ax, (0.71, 0.67, 0.19, 0.12), "IFN rec\nGSE96583", COLORS["red"], facecolor="#FFF6F2", fontsize=9.0)
    draw_rounded_box(ax, (0.71, 0.45, 0.19, 0.12), "LPS rec\nGSE226488", COLORS["red"], facecolor="#FFF6F2", fontsize=9.0)
    arrow_kw = dict(arrowstyle="-|>", mutation_scale=16, linewidth=1.5, color=COLORS["black"])
    ax.annotate("", xy=(0.40, 0.52), xytext=(0.32, 0.71), arrowprops=arrow_kw)
    ax.annotate("", xy=(0.40, 0.48), xytext=(0.32, 0.29), arrowprops=arrow_kw)
    ax.annotate("", xy=(0.71, 0.73), xytext=(0.62, 0.56), arrowprops=arrow_kw)
    ax.annotate("", xy=(0.71, 0.51), xytext=(0.62, 0.51), arrowprops=arrow_kw)
    ax.text(0.50, 0.09, "sender ranking\nreceiver assignment\nprogram prioritization", ha="center", va="center", fontsize=8.8, color=COLORS["gray"])
    save_figure(fig, "workflow")


def plot_validation_anchors():
    fig, ax = fig_ax()
    ax.axis("off")
    add_axis_caption(ax, "Validation")
    items = [
        ((0.06, 0.66, 0.38, 0.18), "Sender stability", "12 random splits"),
        ((0.56, 0.66, 0.38, 0.18), "Receiver contexts", "IFN-beta and LPS"),
        ((0.06, 0.38, 0.38, 0.18), "Baseline test", "GNN > kNN\nP = 0.0017"),
        ((0.56, 0.38, 0.38, 0.18), "Linking test", "Permutation\nP = 0.0005"),
    ]
    for box, title, text in items:
        draw_rounded_box(ax, box, "", "#D5DDE5", facecolor="#FBFCFD")
        x, y, _, _ = box
        ax.text(x + 0.04, y + 0.12, title, fontweight="bold", fontsize=9.2, color=COLORS["black"])
        ax.text(x + 0.04, y + 0.04, text, color=COLORS["gray"], fontsize=8.6)
    save_figure(fig, "validation_anchors")


def plot_interpretation_scope():
    fig, ax = fig_ax()
    ax.axis("off")
    add_axis_caption(ax, "Scope")
    draw_rounded_box(ax, (0.08, 0.67, 0.84, 0.16), "", COLORS["blue"], facecolor="#F7F9FD")
    draw_rounded_box(ax, (0.08, 0.42, 0.84, 0.14), "", COLORS["red"], facecolor="#FFF6F2")
    draw_rounded_box(ax, (0.08, 0.17, 0.84, 0.14), "", COLORS["green"], facecolor="#F3FBF8")
    ax.text(0.13, 0.76, "Primary claim", color=COLORS["blue"], fontweight="bold", fontsize=9.2)
    ax.text(0.13, 0.69, "Weakly supervised sender-to-receiver\nresponse-program prioritization", fontsize=8.6, color=COLORS["black"])
    ax.text(0.13, 0.50, "Not claimed", color=COLORS["red"], fontweight="bold", fontsize=9.2)
    ax.text(0.13, 0.45, "Phenotype-level causal prediction", fontsize=8.6, color=COLORS["black"])
    ax.text(0.13, 0.25, "Current bottleneck", color=COLORS["green"], fontweight="bold", fontsize=9.2)
    ax.text(0.13, 0.20, "Sparse canonical ligand coverage\nin sender perturbation space", fontsize=8.6, color=COLORS["black"])
    save_figure(fig, "interpretation_scope")


def plot_sender_global_agreement(pred, obs):
    rng = np.random.default_rng(42)
    x = obs.values.ravel()
    y = pred.values.ravel()
    idx = rng.choice(len(x), size=min(35000, len(x)), replace=False)
    x_s = x[idx]
    y_s = y[idx]
    x_lo, x_hi = np.percentile(x_s, [0.5, 99.5])
    y_lo, y_hi = np.percentile(y_s, [0.5, 99.5])
    keep = (x_s >= x_lo) & (x_s <= x_hi) & (y_s >= y_lo) & (y_s <= y_hi)
    x_plot = x_s[keep]
    y_plot = y_s[keep]
    pearson = np.corrcoef(x, y)[0, 1]
    fig, ax = fig_ax()
    hb = ax.hexbin(
        x_plot,
        y_plot,
        gridsize=36,
        cmap=sns.light_palette(COLORS["blue"], as_cmap=True),
        mincnt=1,
        linewidths=0,
    )
    add_axis_caption(ax, "Global agreement", f"Pearson r = {pearson:.3f}; 0.5-99.5% window")
    ax.set_xlabel("Observed shift")
    ax.set_ylabel("Predicted shift")
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.axvline(0, color=COLORS["gray"], lw=0.8, ls="--", zorder=0)
    ax.axhline(0, color=COLORS["gray"], lw=0.8, ls="--", zorder=0)
    style_axes(ax)
    save_figure(fig, "sender_global_agreement")


def plot_sender_quality_distribution(quality, quality_summary):
    fig, ax = fig_ax()
    sns.violinplot(
        data=quality,
        x="split",
        y="pearson_corr",
        hue="split",
        palette=[COLORS["light_blue"], COLORS["light_red"]],
        legend=False,
        inner=None,
        ax=ax,
    )
    sns.stripplot(data=quality, x="split", y="pearson_corr", color="black", size=2.4, alpha=0.45, ax=ax)
    train_p = quality_summary.loc[quality_summary["split"] == "train", "mean_pearson"].iloc[0]
    val_p = quality_summary.loc[quality_summary["split"] == "val", "mean_pearson"].iloc[0]
    add_axis_caption(ax, "Quality", f"train = {train_p:.3f}, val = {val_p:.3f}")
    ax.set_xlabel("")
    ax.set_ylabel("Per-perturbation Pearson")
    style_axes(ax, add_ygrid=True)
    save_figure(fig, "sender_quality_distribution")


def plot_sender_split_stability(split_metrics, diag):
    fig, ax = fig_ax()
    sns.pointplot(data=split_metrics, x="seed", y="val_mean_pearson", color=COLORS["blue"], errorbar=None, ax=ax)
    mean_v = split_metrics["val_mean_pearson"].mean()
    std_v = split_metrics["val_mean_pearson"].std()
    ax.axhline(mean_v, ls="--", lw=1.5, color=COLORS["gray"])
    ax.fill_between(
        np.arange(split_metrics.shape[0]),
        mean_v - std_v,
        mean_v + std_v,
        color=COLORS["light_blue"],
        alpha=0.35,
    )
    unique_pred = int(diag.loc[diag["source"] == "predicted", "n_unique_rows_rounded_6"].iloc[0])
    add_axis_caption(ax, "Split stability", f"12 splits; unique rows = {unique_pred}")
    ax.set_xlabel("Random split")
    ax.set_ylabel("Validation mean Pearson")
    style_axes(ax, add_ygrid=True)
    save_figure(fig, "sender_split_stability")


def plot_sender_knn(knn, knn_test):
    fig, ax = fig_ax()
    summary = (
        knn.groupby("method")["mean_val_pearson"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    sns.barplot(data=summary, x="method", y="mean", palette=[COLORS["blue"], COLORS["light_red"]], ax=ax)
    ax.errorbar(np.arange(summary.shape[0]), summary["mean"], yerr=summary["std"], fmt="none", ecolor="black", capsize=4)
    pval = knn_test["exact_signflip_p"].iloc[0]
    add_axis_caption(ax, "GNN vs kNN", f"Exact sign-flip P = {pval:.4f}")
    ax.set_xlabel("")
    ax.set_ylabel("Validation mean Pearson")
    style_axes(ax, add_ygrid=True)
    save_figure(fig, "sender_gnn_vs_knn")


def plot_receiver_classifier(perf, stem, title, color):
    perf = shorten_cell_labels(perf)
    fig, ax = fig_ax()
    sns.barplot(data=perf, y="cell_type", x="auroc", color=color, ax=ax)
    ax.set_xlim(0.97, 1.0)
    add_axis_caption(ax, title)
    ax.set_xlabel("AUROC")
    ax.set_ylabel("")
    style_axes(ax)
    save_figure(fig, stem)


def plot_receiver_overlap(overlap):
    overlap = shorten_cell_labels(overlap)
    fig, ax = fig_ax()
    sns.barplot(data=overlap.sort_values("top30_up_jaccard", ascending=False), y="cell_type", x="top30_up_jaccard", color=COLORS["green"], ax=ax)
    add_axis_caption(ax, "IFN versus LPS signature overlap")
    ax.set_xlabel("Top30 up-gene Jaccard")
    ax.set_ylabel("")
    style_axes(ax)
    save_figure(fig, "receiver_ifn_lps_overlap")


def plot_signature_heatmap(sig, stem, title, cmap):
    matrix = build_signature_heatmap(sig, top_n=6)
    matrix.index = [CELL_LABELS.get(x, x) for x in matrix.index]
    fig, ax = fig_ax()
    sns.heatmap(matrix, cmap=cmap, ax=ax, cbar_kws={"shrink": 0.65}, linewidths=0.4, linecolor="white")
    ax.set_title(title, loc="left", pad=12, fontweight="bold")
    ax.set_xlabel("Top responsive genes")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    save_figure(fig, stem)


def plot_hybrid_rankings(rank_df, stem, title, color):
    top = rank_df.head(12).copy()
    top["label"] = top["perturbation"].map(short_perturbation_label)
    fig, ax = fig_ax()
    sns.barplot(data=top, y="label", x="hybrid_global_score", color=color, ax=ax)
    ax.set_xlabel("Hybrid global score")
    ax.set_ylabel("")
    ax.set_title(title, loc="left", pad=12, fontweight="bold")
    style_axes(ax)
    save_figure(fig, stem)


def plot_hybrid_baseline_gain(hybrid_tests):
    baseline_map = {
        "combined_score": "Combined",
        "cosine_score": "Cosine",
        "pearson_score": "Pearson",
        "overlap_score": "Overlap",
        "effect_score_scaled": "Effect only",
    }
    plot_df = hybrid_tests.copy()
    plot_df["baseline_label"] = plot_df["baseline"].map(baseline_map)
    fig, ax = fig_ax()
    sns.barplot(
        data=plot_df.sort_values("observed_mean_difference", ascending=False),
        y="baseline_label",
        x="observed_mean_difference",
        color=COLORS["green"],
        ax=ax,
    )
    add_axis_caption(ax, "Hybrid-head gain")
    ax.set_xlabel("Mean concordance advantage")
    ax.set_ylabel("")
    style_axes(ax)
    save_figure(fig, "hybrid_head_gain")


def plot_concordance_heatmap(concord):
    plot_df = concord[concord["subset"] == "all"].pivot(index="cell_type", columns="context", values="spearman")
    plot_df.index = [CELL_LABELS.get(x, x) for x in plot_df.index]
    fig, ax = fig_ax()
    sns.heatmap(plot_df, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0, vmax=1, ax=ax, cbar_kws={"shrink": 0.65}, linewidths=0.4, linecolor="white")
    ax.set_title("Predicted-versus-observed concordance", loc="left", pad=12, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    save_figure(fig, "predicted_observed_concordance")


def plot_context_scatter(ifn_rank, lps_rank, contrastive_ifn, contrastive_lps):
    hybrid = ifn_rank[["perturbation", "hybrid_global_score"]].rename(columns={"hybrid_global_score": "ifn"}).merge(
        lps_rank[["perturbation", "hybrid_global_score"]].rename(columns={"hybrid_global_score": "lps"}),
        on="perturbation",
    )
    contrastive = contrastive_ifn[["perturbation", "contrastive_global_score"]].rename(columns={"contrastive_global_score": "ifn"}).merge(
        contrastive_lps[["perturbation", "contrastive_global_score"]].rename(columns={"contrastive_global_score": "lps"}),
        on="perturbation",
    )

    fig, ax = fig_ax()
    rho = spearmanr(hybrid["ifn"], hybrid["lps"]).statistic
    sns.scatterplot(data=hybrid, x="ifn", y="lps", s=28, color=COLORS["slate"], ax=ax)
    add_axis_caption(ax, "Hybrid context agreement", f"IFN versus LPS rho = {rho:.2f}")
    ax.set_xlabel("IFN hybrid score")
    ax.set_ylabel("LPS hybrid score")
    style_axes(ax)
    save_figure(fig, "hybrid_context_agreement")

    fig, ax = fig_ax()
    rho = spearmanr(contrastive["ifn"], contrastive["lps"]).statistic
    sns.scatterplot(data=contrastive, x="ifn", y="lps", s=28, color=COLORS["gold"], ax=ax)
    add_axis_caption(ax, "Contrastive context reranking", f"IFN versus LPS rho = {rho:.2f}")
    ax.set_xlabel("IFN contrastive score")
    ax.set_ylabel("LPS contrastive score")
    style_axes(ax)
    save_figure(fig, "contrastive_context_reranking")


def plot_receiver_reassignment(cross):
    cross = cross.copy()
    cross.index = [CELL_LABELS.get(x, x) for x in cross.index]
    cross.columns = [CELL_LABELS.get(x, x) for x in cross.columns]
    fig, ax = fig_ax()
    sns.heatmap(cross, annot=True, fmt="g", cmap="OrRd", ax=ax, cbar=False, linewidths=0.4, linecolor="white")
    ax.set_title("Receiver-cell reassignment", loc="left", pad=12, fontweight="bold")
    ax.set_xlabel("LPS top receiver")
    ax.set_ylabel("IFN top receiver")
    save_figure(fig, "receiver_cell_reassignment")


def plot_contrastive_top_hits(contrastive_ifn, contrastive_lps):
    plot_df = pd.concat(
        [
            contrastive_ifn.head(6)[["perturbation", "contrastive_global_score"]].assign(context="IFN"),
            contrastive_lps.head(6)[["perturbation", "contrastive_global_score"]].assign(context="LPS"),
        ]
    ).copy()
    plot_df["perturbation"] = plot_df["perturbation"].map(short_perturbation_label)
    fig, ax = fig_ax()
    sns.barplot(
        data=plot_df,
        y="perturbation",
        x="contrastive_global_score",
        hue="context",
        palette=[COLORS["blue"], COLORS["red"]],
        ax=ax,
    )
    ax.set_title("Top context-sensitive perturbations", loc="left", pad=12, fontweight="bold")
    ax.set_xlabel("Contrastive global score")
    ax.set_ylabel("")
    style_axes(ax)
    save_figure(fig, "contrastive_top_perturbations")


def plot_strict_bottleneck(strict):
    strict_map = {
        "sender_genes": "Sender",
        "strict_ligands": "Ligands",
        "strict_receptors": "Receptors",
        "sender_strict_ligands": "Sender∩L",
        "sender_strict_receptors": "Sender∩R",
    }
    plot_df = strict.copy()
    plot_df["label"] = plot_df["group"].map(strict_map)
    fig, ax = fig_ax()
    sns.barplot(data=plot_df, x="label", y="n", color=COLORS["green"], ax=ax)
    for i, row in plot_df.iterrows():
        ax.text(i, row["n"] + max(plot_df["n"]) * 0.03, str(int(row["n"])), ha="center", va="bottom", fontsize=11)
    ax.set_title("Strict ligand bottleneck", loc="left", pad=12, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Gene count")
    ax.tick_params(axis="x", rotation=30)
    style_axes(ax, add_ygrid=True)
    save_figure(fig, "strict_ligand_bottleneck")


def write_manifest():
    pdfs = sorted(OUT_DIR.glob("*.pdf"))
    lines = ["# Nature-style single-panel publication figures", "", "All figures are editable PDF, 6 x 6 cm, main font size 13.", ""]
    for pdf in pdfs:
        lines.append(f"- [{pdf.name}]({pdf.as_posix()})")
    (OUT_DIR / "figure_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ensure_dirs()
    pred, obs, quality, quality_summary, split_metrics, knn, knn_test, diag = load_sender_data()
    ifn_perf, lps_perf, overlap, ifn_sig, lps_sig = load_receiver_data()
    concord, hybrid_tests, ifn_rank, lps_rank, contrastive_ifn, contrastive_lps, strict, cross = load_linking_data()

    plot_workflow()
    plot_validation_anchors()
    plot_interpretation_scope()
    plot_sender_global_agreement(pred, obs)
    plot_sender_quality_distribution(quality, quality_summary)
    plot_sender_split_stability(split_metrics, diag)
    plot_sender_knn(knn, knn_test)
    plot_receiver_classifier(ifn_perf, "ifn_receiver_classifiers", "IFN AUROC", COLORS["blue"])
    plot_receiver_classifier(lps_perf, "lps_receiver_classifiers", "LPS AUROC", COLORS["red"])
    plot_receiver_overlap(overlap)
    plot_signature_heatmap(ifn_sig, "ifn_receiver_programs", "IFN programs", "Reds")
    plot_signature_heatmap(lps_sig, "lps_receiver_programs", "LPS programs", "OrRd")
    plot_hybrid_rankings(ifn_rank, "ifn_hybrid_rankings", "IFN hybrid top hits", COLORS["blue"])
    plot_hybrid_rankings(lps_rank, "lps_hybrid_rankings", "LPS hybrid top hits", COLORS["red"])
    plot_hybrid_baseline_gain(hybrid_tests)
    plot_concordance_heatmap(concord)
    plot_context_scatter(ifn_rank, lps_rank, contrastive_ifn, contrastive_lps)
    plot_receiver_reassignment(cross)
    plot_contrastive_top_hits(contrastive_ifn, contrastive_lps)
    plot_strict_bottleneck(strict)
    write_manifest()
    print(f"Single-panel figures written to {OUT_DIR}")


if __name__ == "__main__":
    main()
