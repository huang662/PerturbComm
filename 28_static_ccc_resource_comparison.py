from pathlib import Path
from typing import Dict, Iterable, List, Set

import anndata as ad
import numpy as np
import pandas as pd


ROOT = Path.cwd()
OUT_DIR = ROOT / "static_ccc_comparison_outputs"
TABLE_DIR = OUT_DIR / "tables"


def ensure_dirs() -> None:
    for path in [OUT_DIR, TABLE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def split_items(value: str, delimiters: str = ";") -> List[str]:
    if pd.isna(value) or not str(value).strip():
        return []
    text = str(value)
    for delim in delimiters:
        text = text.replace(delim, ";")
    return [part.strip().upper() for part in text.split(";") if part.strip() and part.strip().lower() != "nan"]


def perturbation_genes(text: str) -> List[str]:
    return split_items(str(text).replace("__", ";").replace("_", ";"))


def load_receiver_detected_genes(h5ad_path: Path) -> Set[str]:
    adata = ad.read_h5ad(h5ad_path, backed="r")
    var_genes = set(map(str.upper, adata.var_names.astype(str)))
    if "gene_symbol" in adata.var.columns:
        gene_symbols = set(adata.var["gene_symbol"].astype(str).str.upper())
        var_genes |= gene_symbols
    return {g for g in var_genes if g and g != "NAN"}


def build_cpdb_partner_map(cpdb_df: pd.DataFrame) -> Dict[str, Set[str]]:
    partner_map: Dict[str, Set[str]] = {}
    for interactors in cpdb_df["interactors"].dropna():
        parts = [x.strip().upper() for x in str(interactors).replace("+", "-").split("-") if x.strip()]
        uniq = list(dict.fromkeys(parts))
        for gene in uniq:
            others = {x for x in uniq if x != gene}
            partner_map.setdefault(gene, set()).update(others)
    return partner_map


def score_static_baselines(
    sender_ann: pd.DataFrame,
    detected_ifn: Set[str],
    detected_lps: Set[str],
    cpdb_partner_map: Dict[str, Set[str]],
) -> pd.DataFrame:
    rows = []
    for _, row in sender_ann.iterrows():
        genes = perturbation_genes(row["perturbation_genes"])
        matched_comm = set(split_items(row.get("matched_communication_genes", "")))
        matched_ligands = set(split_items(row.get("matched_ligands", "")))
        matched_receptors = set(split_items(row.get("matched_receptors", "")))
        omni_partners = set(split_items(row.get("candidate_partner_receptors", ""))) | set(split_items(row.get("candidate_partner_ligands", "")))
        cpdb_partners = set()
        for gene in genes:
            cpdb_partners |= cpdb_partner_map.get(gene, set())

        ifn_omni_detected = omni_partners & detected_ifn
        lps_omni_detected = omni_partners & detected_lps
        ifn_cpdb_detected = cpdb_partners & detected_ifn
        lps_cpdb_detected = cpdb_partners & detected_lps

        base = {
            "perturbation": row["perturbation"],
            "perturbation_genes": row["perturbation_genes"],
            "matched_communication_genes": ";".join(sorted(matched_comm)),
            "matched_ligands": ";".join(sorted(matched_ligands)),
            "matched_receptors": ";".join(sorted(matched_receptors)),
            "omnipath_partners": ";".join(sorted(omni_partners)),
            "cpdb_partners": ";".join(sorted(cpdb_partners)),
            "n_matched_comm": len(matched_comm),
            "n_matched_ligands": len(matched_ligands),
            "n_matched_receptors": len(matched_receptors),
        }
        rows.append(
            {
                **base,
                "context": "ifn",
                "omnipath_detected_partners": ";".join(sorted(ifn_omni_detected)),
                "cpdb_detected_partners": ";".join(sorted(ifn_cpdb_detected)),
                "n_omnipath_detected_partners": len(ifn_omni_detected),
                "n_cpdb_detected_partners": len(ifn_cpdb_detected),
                "static_omnipath_score": len(matched_comm) + 0.75 * len(ifn_omni_detected) + 0.5 * len(matched_receptors),
                "static_cpdb_score": len(matched_comm) + 0.75 * len(ifn_cpdb_detected) + 0.5 * len(matched_receptors),
            }
        )
        rows.append(
            {
                **base,
                "context": "lps",
                "omnipath_detected_partners": ";".join(sorted(lps_omni_detected)),
                "cpdb_detected_partners": ";".join(sorted(lps_cpdb_detected)),
                "n_omnipath_detected_partners": len(lps_omni_detected),
                "n_cpdb_detected_partners": len(lps_cpdb_detected),
                "static_omnipath_score": len(matched_comm) + 0.75 * len(lps_omni_detected) + 0.5 * len(matched_receptors),
                "static_cpdb_score": len(matched_comm) + 0.75 * len(lps_cpdb_detected) + 0.5 * len(matched_receptors),
            }
        )
    return pd.DataFrame(rows)


def overlap_summary(static_df: pd.DataFrame, perturbcomm_df: pd.DataFrame, score_col: str, context: str, top_n: int = 20) -> Dict[str, object]:
    static_top = set(static_df[static_df["context"] == context].sort_values(score_col, ascending=False).head(top_n)["perturbation"])
    pc_top = set(perturbcomm_df[perturbcomm_df["context"] == context].head(top_n)["perturbation"])
    return {
        "context": context,
        "baseline": score_col,
        "top_n": top_n,
        "static_top_n": len(static_top),
        "perturbcomm_top_n": len(pc_top),
        "overlap_n": len(static_top & pc_top),
        "overlap_fraction_vs_perturbcomm": len(static_top & pc_top) / max(len(pc_top), 1),
        "perturbcomm_unique": ";".join(sorted(pc_top - static_top)),
        "static_unique": ";".join(sorted(static_top - pc_top)),
    }


def main() -> None:
    ensure_dirs()
    sender_ann = pd.read_csv(ROOT / "communication_prior_outputs" / "tables" / "gse133344_sender_communication_annotations.csv")
    cpdb_df = pd.read_csv(ROOT / "cellphonedb_interaction_input.csv")
    ifn_rank = pd.read_csv(ROOT / "sender_receiver_link_outputs" / "tables" / "ifn_predicted_hybrid_global_ranking.csv")
    lps_rank = pd.read_csv(ROOT / "sender_receiver_link_outputs" / "tables" / "lps_predicted_hybrid_global_ranking.csv")
    ifn_rank["communication_like"] = ifn_rank["communication_like"].astype(str).str.lower() == "true"
    lps_rank["communication_like"] = lps_rank["communication_like"].astype(str).str.lower() == "true"
    perturbcomm_top = pd.concat(
        [
            ifn_rank[ifn_rank["communication_like"]].assign(context="ifn"),
            lps_rank[lps_rank["communication_like"]].assign(context="lps"),
        ],
        ignore_index=True,
    )

    detected_ifn = load_receiver_detected_genes(ROOT / "gse96583_receiver_outputs" / "gse96583_batch2_receiver_processed.h5ad")
    detected_lps = load_receiver_detected_genes(ROOT / "gse226488_lps_receiver_outputs" / "gse226488_lps_receiver_processed.h5ad")
    cpdb_partner_map = build_cpdb_partner_map(cpdb_df)

    static_df = score_static_baselines(sender_ann, detected_ifn, detected_lps, cpdb_partner_map)
    static_df.to_csv(TABLE_DIR / "static_ccc_resource_scores.csv", index=False)

    overlaps = []
    for context in ["ifn", "lps"]:
        overlaps.append(overlap_summary(static_df, perturbcomm_top, "static_omnipath_score", context, top_n=20))
        overlaps.append(overlap_summary(static_df, perturbcomm_top, "static_cpdb_score", context, top_n=20))
    overlap_df = pd.DataFrame(overlaps)
    overlap_df.to_csv(TABLE_DIR / "static_vs_perturbcomm_top_overlap.csv", index=False)

    top_static = []
    for context in ["ifn", "lps"]:
        for baseline in ["static_omnipath_score", "static_cpdb_score"]:
            subset = static_df[static_df["context"] == context].sort_values(baseline, ascending=False).head(20).copy()
            subset["baseline"] = baseline
            top_static.append(subset)
    pd.concat(top_static, ignore_index=True).to_csv(TABLE_DIR / "top20_static_ccc_candidates.csv", index=False)

    summary_lines = [
        "Static CCC resource comparison complete.",
        f"IFN detected receiver genes: {len(detected_ifn)}",
        f"LPS detected receiver genes: {len(detected_lps)}",
    ]
    for _, row in overlap_df.iterrows():
        summary_lines.append(
            f"{row['context']} {row['baseline']} overlap with PerturbComm top{row['top_n']}: {row['overlap_n']} ({row['overlap_fraction_vs_perturbcomm']:.2f})"
        )
    (OUT_DIR / "run_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
