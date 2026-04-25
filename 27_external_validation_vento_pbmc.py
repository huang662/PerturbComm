from pathlib import Path
from typing import Iterable, List

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr


ROOT = Path.cwd()
OUT_DIR = ROOT / "external_validation_outputs"
TABLE_DIR = OUT_DIR / "tables"
DATA_PATH = ROOT / "external_validation_data" / "vento_pbmc_processed.h5ad"


def ensure_dirs() -> None:
    for path in [OUT_DIR, TABLE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def normalize_genes(genes: Iterable[str]) -> List[str]:
    seen = set()
    ordered = []
    for gene in genes:
        g = str(gene).strip()
        if g and g not in seen:
            seen.add(g)
            ordered.append(g)
    return ordered


def split_genes(value: str) -> List[str]:
    if pd.isna(value) or not str(value).strip():
        return []
    text = str(value).replace("__", ";").replace("_", ";")
    return [part.strip() for part in text.split(";") if part.strip() and part.strip().lower() != "nan"]


def top_component_genes(df: pd.DataFrame, n: int = 10) -> List[str]:
    top = df.head(n).copy()
    return normalize_genes(g for text in top["perturbation_genes"] for g in split_genes(text))


def receiver_gene_set(sig_df: pd.DataFrame, top_n_per_cell: int = 8) -> List[str]:
    up = sig_df[sig_df["direction"] == "up_in_stim"].copy()
    top = up.groupby("cell_type").head(top_n_per_cell)
    return normalize_genes(top["gene"])


def compute_score(adata, genes: List[str]) -> np.ndarray:
    gene_index = [gene for gene in genes if gene in adata.var_names]
    if not gene_index:
        return np.zeros(adata.n_obs, dtype=float)
    matrix = adata[:, gene_index].X
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    else:
        matrix = np.asarray(matrix)
    if matrix.ndim == 1:
        return matrix.astype(float)
    return matrix.mean(axis=1).astype(float)


def group_test(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    rows = []
    control = df.loc[df["Group"] == "Control", value_col].dropna().to_numpy()
    for group, sub in df.groupby("Group"):
        values = sub[value_col].dropna().to_numpy()
        if group == "Control" or len(values) == 0 or len(control) == 0:
            continue
        stat = mannwhitneyu(values, control, alternative="two-sided")
        rows.append(
            {
                "group": group,
                "value_col": value_col,
                "group_n": len(values),
                "control_n": len(control),
                "group_mean": float(np.mean(values)),
                "control_mean": float(np.mean(control)),
                "group_minus_control": float(np.mean(values) - np.mean(control)),
                "mannwhitney_pvalue": float(stat.pvalue),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ensure_dirs()
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing external validation dataset: {DATA_PATH}")

    ifn_rank = pd.read_csv(ROOT / "sender_receiver_link_outputs" / "tables" / "ifn_predicted_hybrid_global_ranking.csv")
    comm_rank = pd.read_csv(ROOT / "sender_receiver_link_outputs" / "tables" / "top30_communication_like_candidates.csv")
    ifn_sig = pd.read_csv(ROOT / "gse96583_receiver_outputs" / "tables" / "receiver_response_signatures.csv")
    lps_sig = pd.read_csv(ROOT / "gse226488_lps_receiver_outputs" / "tables" / "receiver_response_signatures.csv")

    sender_top_genes = top_component_genes(ifn_rank, n=10)
    sender_comm_genes = top_component_genes(comm_rank[comm_rank["context"] == "ifn"], n=10)
    tgfbr2_gene_set = ["TGFBR2", "TGFB1", "TGFB2", "TGFB3"]
    ifn_receiver_genes = receiver_gene_set(ifn_sig, top_n_per_cell=8)
    lps_receiver_genes = receiver_gene_set(lps_sig, top_n_per_cell=8)

    adata = ad.read_h5ad(DATA_PATH, backed="r")
    obs = adata.obs.copy()
    obs["sender_score_all"] = compute_score(adata, sender_top_genes)
    obs["sender_score_comm"] = compute_score(adata, sender_comm_genes)
    obs["sender_score_tgfbr2_axis"] = compute_score(adata, tgfbr2_gene_set)
    obs["receiver_ifn_score"] = compute_score(adata, ifn_receiver_genes)
    obs["receiver_lps_score"] = compute_score(adata, lps_receiver_genes)

    annotation = obs["Annotation"].astype(str)
    obs["sender_compartment"] = np.where(annotation.str.contains("mono|DC|pDC|cDC", case=False, regex=True), "myeloid", "other")
    obs["receiver_compartment"] = np.where(annotation.str.contains("NK|CD8|CD4", case=False, regex=True), "lymphoid", "other")
    obs["receiver_subtype"] = np.select(
        [
            annotation.str.contains("NK", case=False, regex=True),
            annotation.str.contains("CD8", case=False, regex=True),
            annotation.str.contains("CD4", case=False, regex=True),
        ],
        ["NK", "CD8_T", "CD4_T"],
        default="other",
    )

    obs.to_csv(TABLE_DIR / "vento_cell_level_scores_preview.csv", index=True)

    sender_donor = (
        obs[obs["sender_compartment"] == "myeloid"]
        .groupby(["Donor_ID", "Group"], observed=True)[["sender_score_all", "sender_score_comm", "sender_score_tgfbr2_axis"]]
        .mean()
        .reset_index()
    )
    receiver_donor = (
        obs[obs["receiver_compartment"] == "lymphoid"]
        .groupby(["Donor_ID", "Group"], observed=True)[["receiver_ifn_score", "receiver_lps_score"]]
        .mean()
        .reset_index()
    )
    receiver_subtype = (
        obs[obs["receiver_subtype"].isin(["NK", "CD8_T", "CD4_T"])]
        .groupby(["Donor_ID", "Group", "receiver_subtype"], observed=True)[["receiver_ifn_score", "receiver_lps_score"]]
        .mean()
        .reset_index()
    )

    merged = sender_donor.merge(receiver_donor, on=["Donor_ID", "Group"], how="inner").dropna()
    merged.to_csv(TABLE_DIR / "vento_donor_level_sender_receiver_scores.csv", index=False)
    receiver_subtype.to_csv(TABLE_DIR / "vento_donor_level_receiver_subtype_scores.csv", index=False)

    corr_rows = []
    for sender_col, receiver_col in [
        ("sender_score_all", "receiver_ifn_score"),
        ("sender_score_comm", "receiver_ifn_score"),
        ("sender_score_tgfbr2_axis", "receiver_ifn_score"),
        ("sender_score_all", "receiver_lps_score"),
        ("sender_score_comm", "receiver_lps_score"),
    ]:
        rho, pval = spearmanr(merged[sender_col], merged[receiver_col])
        corr_rows.append({"sender_metric": sender_col, "receiver_metric": receiver_col, "spearman_rho": float(rho), "pvalue": float(pval), "n_donors": merged.shape[0]})
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(TABLE_DIR / "vento_sender_receiver_donor_correlations.csv", index=False)

    subtype_corr_rows = []
    for subtype, sub in receiver_subtype.groupby("receiver_subtype"):
        sub_merged = sender_donor.merge(sub, on=["Donor_ID", "Group"], how="inner").dropna()
        for sender_col in ["sender_score_all", "sender_score_comm", "sender_score_tgfbr2_axis"]:
            for receiver_col in ["receiver_ifn_score", "receiver_lps_score"]:
                rho, pval = spearmanr(sub_merged[sender_col], sub_merged[receiver_col])
                subtype_corr_rows.append(
                    {
                        "receiver_subtype": subtype,
                        "sender_metric": sender_col,
                        "receiver_metric": receiver_col,
                        "spearman_rho": float(rho),
                        "pvalue": float(pval),
                        "n_donors": sub_merged.shape[0],
                    }
                )
    subtype_corr_df = pd.DataFrame(subtype_corr_rows)
    subtype_corr_df.to_csv(TABLE_DIR / "vento_sender_receiver_subtype_correlations.csv", index=False)

    group_tests = []
    for value_col in ["sender_score_all", "sender_score_comm", "sender_score_tgfbr2_axis"]:
        group_tests.append(group_test(sender_donor, value_col))
    for value_col in ["receiver_ifn_score", "receiver_lps_score"]:
        group_tests.append(group_test(receiver_donor, value_col))
    group_tests_df = pd.concat([df for df in group_tests if not df.empty], ignore_index=True)
    group_tests_df.to_csv(TABLE_DIR / "vento_group_comparison_tests.csv", index=False)

    gene_availability = pd.DataFrame(
        {
            "gene_set": ["sender_top", "sender_comm", "tgfbr2_axis", "ifn_receiver", "lps_receiver"],
            "n_requested": [len(sender_top_genes), len(sender_comm_genes), len(tgfbr2_gene_set), len(ifn_receiver_genes), len(lps_receiver_genes)],
            "n_present_in_vento": [
                sum(g in adata.var_names for g in sender_top_genes),
                sum(g in adata.var_names for g in sender_comm_genes),
                sum(g in adata.var_names for g in tgfbr2_gene_set),
                sum(g in adata.var_names for g in ifn_receiver_genes),
                sum(g in adata.var_names for g in lps_receiver_genes),
            ],
            "genes_present": [
                ";".join([g for g in sender_top_genes if g in adata.var_names]),
                ";".join([g for g in sender_comm_genes if g in adata.var_names]),
                ";".join([g for g in tgfbr2_gene_set if g in adata.var_names]),
                ";".join([g for g in ifn_receiver_genes if g in adata.var_names]),
                ";".join([g for g in lps_receiver_genes if g in adata.var_names]),
            ],
        }
    )
    gene_availability.to_csv(TABLE_DIR / "vento_gene_set_availability.csv", index=False)

    summary_lines = [
        "External validation on Vento COVID-19 autoimmunity PBMCs complete.",
        f"Dataset shape: {adata.n_obs} cells x {adata.n_vars} genes",
        f"Donors: {merged['Donor_ID'].nunique()}",
        f"Groups: {', '.join(sorted(merged['Group'].unique()))}",
        f"Myeloid sender correlation with lymphoid IFN score (communication set): {corr_df.loc[(corr_df['sender_metric']=='sender_score_comm') & (corr_df['receiver_metric']=='receiver_ifn_score'), 'spearman_rho'].iloc[0]:.3f}",
        f"TGFBR2-axis correlation with lymphoid IFN score: {corr_df.loc[(corr_df['sender_metric']=='sender_score_tgfbr2_axis') & (corr_df['receiver_metric']=='receiver_ifn_score'), 'spearman_rho'].iloc[0]:.3f}",
    ]
    (OUT_DIR / "run_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
