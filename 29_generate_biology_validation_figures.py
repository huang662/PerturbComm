from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr


ROOT = Path.cwd()
OUT_DIR = ROOT / "publication_figures_single_nature"
PREVIEW_DIR = OUT_DIR / "previews"
CM_TO_IN = 1 / 2.54
PANEL_SIZE_CM = 6.0
PANEL_SIZE_IN = PANEL_SIZE_CM * CM_TO_IN

COLORS = {
    "blue": "#3C5488",
    "red": "#E64B35",
    "green": "#00A087",
    "gold": "#F39B7F",
    "slate": "#4D4D4D",
    "gray": "#8A8A8A",
    "grid": "#E6E6E6",
    "black": "#1A1A1A",
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
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)
sns.set_theme(style="ticks")


def ensure_dirs():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(fig, stem: str):
    try:
        fig.tight_layout(pad=0.2)
    except Exception:
        pass
    fig.savefig(OUT_DIR / f"{stem}.pdf", transparent=False)
    fig.savefig(PREVIEW_DIR / f"{stem}.png", dpi=260, transparent=False)
    plt.close(fig)


def panel_fig():
    return plt.subplots(figsize=(PANEL_SIZE_IN, PANEL_SIZE_IN))


def style_axes(ax, add_ygrid: bool = False):
    sns.despine(ax=ax, top=True, right=True)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.spines["left"].set_color(COLORS["black"])
    ax.spines["bottom"].set_color(COLORS["black"])
    ax.tick_params(length=3, width=0.8, colors=COLORS["black"])
    if add_ygrid:
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.6)
    else:
        ax.grid(False)


def plot_role_enrichment():
    df = pd.read_csv(ROOT / "biological_interpretation_outputs" / "tables" / "top_candidate_role_enrichment_tests.csv")
    df = df[df["label"].str.contains("communication_genes|strict_receptors")].copy()
    df["set"] = df["label"].str.replace("_communication_genes", "", regex=False).str.replace("_strict_receptors", "", regex=False)
    df["feature"] = np.where(df["label"].str.contains("communication_genes"), "communication", "strict receptor")
    df["neglog10p"] = -np.log10(df["fisher_pvalue"].clip(lower=1e-10))
    fig, ax = panel_fig()
    sns.barplot(data=df, x="set", y="neglog10p", hue="feature", palette=[COLORS["blue"], COLORS["green"]], ax=ax)
    ax.axhline(-np.log10(0.05), ls="--", lw=1.5, color=COLORS["gray"])
    ax.set_title("Comm enrichment", loc="left", pad=4, fontweight="bold", fontsize=12)
    ax.set_xlabel("")
    ax.set_ylabel("-log10 Fisher P")
    ax.tick_params(axis="x", rotation=30)
    style_axes(ax, add_ygrid=True)
    save_figure(fig, "communication_role_enrichment")


def plot_hallmark_overlap():
    df = pd.read_csv(ROOT / "biological_interpretation_outputs" / "tables" / "sender_receiver_shared_hallmarks.csv")
    pivot = df.pivot(index="sender_set", columns="receiver_set", values="n_shared_hallmarks")
    pivot.index = [x.replace("_sender", "").replace("ifn_", "IFN ").replace("lps_", "LPS ") for x in pivot.index]
    pivot.columns = [x.replace("_receiver", "").replace("ifn_", "IFN ").replace("lps_", "LPS ") for x in pivot.columns]
    fig, ax = panel_fig()
    sns.heatmap(pivot, annot=True, cmap="YlGnBu", vmin=0, vmax=max(2, np.nanmax(pivot.to_numpy())), cbar=False, ax=ax, linewidths=0.4, linecolor="white")
    ax.set_title("Hallmarks", loc="left", pad=4, fontweight="bold", fontsize=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)
    save_figure(fig, "sender_receiver_hallmark_overlap")


def _load_tgfbr2_points(subtype: str):
    sender = pd.read_csv(ROOT / "external_validation_outputs" / "tables" / "vento_donor_level_sender_receiver_scores.csv")
    subtype_df = pd.read_csv(ROOT / "external_validation_outputs" / "tables" / "vento_donor_level_receiver_subtype_scores.csv")
    sub = subtype_df[subtype_df["receiver_subtype"] == subtype].copy()
    merged = sender.merge(sub, on=["Donor_ID", "Group"], how="inner").dropna()
    return merged


def plot_tgfbr2_scatter(subtype: str, stem: str, title: str):
    df = _load_tgfbr2_points(subtype)
    ifn_col = "receiver_ifn_score_y" if "receiver_ifn_score_y" in df.columns else "receiver_ifn_score"
    fig, ax = panel_fig()
    sns.scatterplot(data=df, x="sender_score_tgfbr2_axis", y=ifn_col, hue="Group", palette="tab10", s=60, ax=ax, legend=False)
    rho, pval = spearmanr(df["sender_score_tgfbr2_axis"], df[ifn_col])
    ax.set_title(title, loc="left", pad=4, fontweight="bold", fontsize=12)
    ax.text(0.02, 0.96, f"rho = {rho:.3f}\nP = {pval:.4f}", transform=ax.transAxes, ha="left", va="top", color=COLORS["gray"], fontsize=9.5)
    ax.set_xlabel("Myeloid TGFBR2")
    ax.set_ylabel(f"{subtype} IFN")
    style_axes(ax)
    save_figure(fig, stem)


def plot_static_overlap():
    df = pd.read_csv(ROOT / "static_ccc_comparison_outputs" / "tables" / "static_vs_perturbcomm_top_overlap.csv")
    df = df.copy()
    baseline_map = {
        "static_omnipath_score": "OmniPath",
        "static_cpdb_score": "CellPhoneDB",
    }
    df["baseline_short"] = df["baseline"].map(baseline_map).fillna(df["baseline"])
    fig, ax = panel_fig()
    sns.barplot(data=df, x="context", y="overlap_fraction_vs_perturbcomm", hue="baseline_short", palette=[COLORS["blue"], COLORS["red"]], ax=ax)
    ax.set_title("Static CCC overlap", loc="left", pad=4, fontweight="bold", fontsize=12)
    ax.set_xlabel("")
    ax.set_ylabel("Top-20 overlap fraction")
    leg = ax.legend(
        title="Baseline",
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=2,
        handlelength=1.2,
        columnspacing=0.8,
        borderaxespad=0.0,
        fontsize=9.5,
        title_fontsize=9.5,
    )
    if leg is not None:
        for txt in leg.get_texts():
            txt.set_color(COLORS["black"])
    style_axes(ax, add_ygrid=True)
    save_figure(fig, "static_ccc_overlap")


def plot_string_summary():
    df = pd.read_csv(ROOT / "biological_interpretation_outputs" / "tables" / "string_network_shortest_path_summary.csv")
    plot_df = pd.DataFrame(
        {
            "metric": ["Observed", "Null mean"],
            "value": [df["observed_mean_shortest_path"].iloc[0], df["null_mean_shortest_path"].iloc[0]],
        }
    )
    fig, ax = panel_fig()
    sns.barplot(data=plot_df, x="metric", y="value", palette=[COLORS["gold"], COLORS["gray"]], ax=ax)
    ax.set_title("STRING distance", loc="left", pad=4, fontweight="bold", fontsize=12)
    ax.text(0.02, 0.98, f"empirical P = {df['empirical_pvalue'].iloc[0]:.3f}", transform=ax.transAxes, ha="left", va="top", color=COLORS["gray"])
    ax.set_xlabel("")
    ax.set_ylabel("Mean shortest path")
    style_axes(ax, add_ygrid=True)
    save_figure(fig, "string_network_distance_summary")


def write_manifest():
    pdfs = sorted(OUT_DIR.glob("*.pdf"))
    lines = ["# Nature-style single-panel publication figures", "", "All figures are editable PDF, 6 x 6 cm, main font size 13.", ""]
    for pdf in pdfs:
        lines.append(f"- [{pdf.name}]({pdf.as_posix()})")
    (OUT_DIR / "figure_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ensure_dirs()
    plot_role_enrichment()
    plot_hallmark_overlap()
    plot_tgfbr2_scatter("NK", "tgfbr2_vs_nk_ifn", "NK IFN")
    plot_tgfbr2_scatter("CD4_T", "tgfbr2_vs_cd4_ifn", "CD4 IFN")
    plot_tgfbr2_scatter("CD8_T", "tgfbr2_vs_cd8_ifn", "CD8 IFN")
    plot_static_overlap()
    plot_string_summary()
    write_manifest()
    print(f"Biology validation figures written to {OUT_DIR}")


if __name__ == "__main__":
    main()
