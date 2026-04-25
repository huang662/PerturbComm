from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import gseapy as gp
import networkx as nx
import numpy as np
import pandas as pd
import requests
from scipy.stats import fisher_exact
from requests.exceptions import RequestException


ROOT = Path.cwd()
OUT_DIR = ROOT / "biological_interpretation_outputs"
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"

RNG = np.random.default_rng(42)


def ensure_dirs() -> None:
    for path in [OUT_DIR, TABLE_DIR, FIG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)


def split_genes(value: str) -> List[str]:
    if pd.isna(value) or not str(value).strip():
        return []
    text = str(value).replace("__", ";").replace("_", ";")
    return [part.strip() for part in text.split(";") if part.strip() and part.strip().lower() != "nan"]


def normalize_gene_list(genes: Iterable[str]) -> List[str]:
    seen = set()
    ordered = []
    for gene in genes:
        g = str(gene).strip().upper()
        if g and g not in seen:
            seen.add(g)
            ordered.append(g)
    return ordered


def load_rankings() -> Dict[str, pd.DataFrame]:
    return {
        "ifn_hybrid": read_csv(ROOT / "sender_receiver_link_outputs" / "tables" / "ifn_predicted_hybrid_global_ranking.csv"),
        "lps_hybrid": read_csv(ROOT / "sender_receiver_link_outputs" / "tables" / "lps_predicted_hybrid_global_ranking.csv"),
        "ifn_contrastive": read_csv(ROOT / "sender_receiver_link_outputs" / "tables" / "ifn_predicted_contrastive_global_ranking.csv"),
        "lps_contrastive": read_csv(ROOT / "sender_receiver_link_outputs" / "tables" / "lps_predicted_contrastive_global_ranking.csv"),
        "comm_candidates": read_csv(ROOT / "sender_receiver_link_outputs" / "tables" / "top30_communication_like_candidates.csv"),
    }


def load_resources() -> Dict[str, pd.DataFrame]:
    return {
        "comm_roles": read_csv(ROOT / "communication_prior_outputs" / "tables" / "communication_gene_roles.csv"),
        "sender_comm": read_csv(ROOT / "communication_prior_outputs" / "tables" / "gse133344_sender_communication_annotations.csv"),
        "strict_ligands": read_csv(ROOT / "omnipath_secreted_ligands.tsv", sep="\t"),
        "strict_receptors": read_csv(ROOT / "omnipath_receptors.tsv", sep="\t"),
        "ifn_sig": read_csv(ROOT / "gse96583_receiver_outputs" / "tables" / "receiver_response_signatures.csv"),
        "lps_sig": read_csv(ROOT / "gse226488_lps_receiver_outputs" / "tables" / "receiver_response_signatures.csv"),
        "tgfbr2_links": read_csv(ROOT / "receptor_focused_support_outputs" / "tables" / "tgfbr2_linked_candidates.csv"),
        "tgfbr2_expr": read_csv(ROOT / "receptor_focused_support_outputs" / "tables" / "tgfbr2_receiver_expression.csv"),
        "tgfbr2_support": read_csv(ROOT / "receptor_focused_support_outputs" / "tables" / "tgfbr2_external_ligand_support.csv"),
    }


def top_component_genes(rank_df: pd.DataFrame, n: int = 20) -> Tuple[pd.DataFrame, List[str]]:
    top = rank_df.head(n).copy()
    genes = normalize_gene_list(g for text in top["perturbation_genes"] for g in split_genes(text))
    return top, genes


def receiver_gene_set(sig_df: pd.DataFrame, top_n_per_cell: int = 8, direction: str = "up_in_stim") -> List[str]:
    subset = sig_df[sig_df["direction"] == direction].copy()
    subset = subset.groupby("cell_type").head(top_n_per_cell)
    return normalize_gene_list(subset["gene"])


def first_sentence(text: str) -> str:
    if not text:
        return ""
    sent = str(text).split(". ")[0].strip()
    return sent[:260]


def ncbi_gene_summary(gene: str) -> Dict[str, str]:
    try:
        term = f"{gene}[Gene Name] AND Homo sapiens[Organism]"
        esearch = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={"db": "gene", "term": term, "retmode": "json"},
            timeout=15,
        ).json()
        ids = esearch.get("esearchresult", {}).get("idlist", [])
        if not ids:
            return {"gene_symbol": gene, "ncbi_gene_id": "", "description": "", "summary": ""}
        uid = ids[0]
        esummary = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
            params={"db": "gene", "id": uid, "retmode": "json"},
            timeout=15,
        ).json()
        obj = esummary.get("result", {}).get(uid, {})
        return {
            "gene_symbol": gene,
            "ncbi_gene_id": uid,
            "description": obj.get("description", ""),
            "summary": first_sentence(obj.get("summary", "")),
        }
    except RequestException:
        return {"gene_symbol": gene, "ncbi_gene_id": "", "description": "", "summary": ""}


def pubmed_support(gene: str, term: str) -> Tuple[int, str]:
    try:
        query = f"{gene}[Title/Abstract] AND {term}[Title/Abstract]"
        payload = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={"db": "pubmed", "term": query, "retmode": "json", "retmax": 3},
            timeout=15,
        ).json()
        result = payload.get("esearchresult", {})
        count = int(result.get("count", 0))
        pmids = ";".join(result.get("idlist", []))
        return count, pmids
    except RequestException:
        return 0, ""


def build_gene_annotation_table(gene_sets: Dict[str, List[str]], resources: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    all_genes = normalize_gene_list(g for genes in gene_sets.values() for g in genes)
    comm_roles = resources["comm_roles"].copy()
    comm_roles["gene"] = comm_roles["gene"].astype(str).str.upper()
    role_map = comm_roles.groupby("gene")["roles"].apply(lambda x: ";".join(sorted(set(";".join(x).split(";"))))).to_dict()

    strict_ligands = set(resources["strict_ligands"]["genesymbol"].astype(str).str.upper())
    strict_receptors = set(resources["strict_receptors"]["genesymbol"].astype(str).str.upper())

    rows = []
    for gene in all_genes:
        info = ncbi_gene_summary(gene)
        immune_count, immune_pmids = pubmed_support(gene, "immune")
        ifn_count, ifn_pmids = pubmed_support(gene, "interferon")
        lps_count, lps_pmids = pubmed_support(gene, "lipopolysaccharide")
        row = {
            "gene": gene,
            "description": info["description"],
            "summary_first_sentence": info["summary"],
            "communication_roles": role_map.get(gene, ""),
            "is_strict_ligand": gene in strict_ligands,
            "is_strict_receptor": gene in strict_receptors,
            "immune_pubmed_count": immune_count,
            "immune_pmids_top3": immune_pmids,
            "ifn_pubmed_count": ifn_count,
            "ifn_pmids_top3": ifn_pmids,
            "lps_pubmed_count": lps_count,
            "lps_pmids_top3": lps_pmids,
        }
        for key, genes in gene_sets.items():
            row[f"in_{key}"] = gene in genes
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["in_ifn_hybrid", "in_lps_hybrid", "in_ifn_contrastive", "in_lps_contrastive", "immune_pubmed_count"],
        ascending=[False, False, False, False, False],
    )


def fisher_enrichment(top_genes: List[str], universe: List[str], positive: Iterable[str], label: str) -> Dict[str, object]:
    top_set = set(top_genes)
    universe_set = set(universe)
    pos_set = set(positive) & universe_set
    a = len(top_set & pos_set)
    b = len(top_set - pos_set)
    c = len((universe_set - top_set) & pos_set)
    d = len(universe_set - top_set - pos_set)
    odds_ratio, pvalue = fisher_exact([[a, b], [c, d]], alternative="greater")
    return {
        "label": label,
        "top_positive": a,
        "top_total": len(top_set),
        "universe_positive": len(pos_set),
        "universe_total": len(universe_set),
        "odds_ratio": odds_ratio,
        "fisher_pvalue": pvalue,
    }


def run_enrichr(genes: List[str], prefix: str) -> pd.DataFrame:
    if not genes:
        return pd.DataFrame()
    libraries = [
        "MSigDB_Hallmark_2020",
        "Reactome_2022",
        "KEGG_2021_Human",
        "GO_Biological_Process_2023",
    ]
    frames = []
    for library in libraries:
        enr = gp.enrichr(gene_list=genes, gene_sets=[library], organism="human", outdir=None, cutoff=0.5)
        res = enr.results.copy()
        res["library"] = library
        res["analysis_set"] = prefix
        frames.append(res)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def select_significant_hallmarks(enr_df: pd.DataFrame) -> set:
    if enr_df.empty:
        return set()
    subset = enr_df[(enr_df["library"] == "MSigDB_Hallmark_2020") & (enr_df["Adjusted P-value"] < 0.05)]
    return set(subset["Term"].astype(str))


def fetch_string_network(genes: List[str]) -> pd.DataFrame:
    if len(genes) < 2:
        return pd.DataFrame()
    response = requests.post(
        "https://string-db.org/api/json/network",
        data={
            "identifiers": "\r".join(genes),
            "species": 9606,
            "network_type": "functional",
            "required_score": 400,
        },
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    if not payload:
        return pd.DataFrame()
    return pd.DataFrame(payload)


def build_graph(string_df: pd.DataFrame) -> nx.Graph:
    graph = nx.Graph()
    for _, row in string_df.iterrows():
        a = row.get("preferredName_A")
        b = row.get("preferredName_B")
        score = row.get("score", 0.0)
        if a and b:
            graph.add_edge(str(a).upper(), str(b).upper(), score=float(score))
    return graph


def mean_shortest_path(graph: nx.Graph, source_genes: List[str], target_genes: List[str]) -> float:
    targets = [g for g in target_genes if g in graph.nodes]
    if not targets:
        return np.nan
    path_lengths = []
    for src in source_genes:
        if src not in graph.nodes:
            continue
        distances = nx.single_source_shortest_path_length(graph, src, cutoff=6)
        target_dists = [distances[tgt] for tgt in targets if tgt in distances]
        if target_dists:
            path_lengths.append(min(target_dists))
    return float(np.mean(path_lengths)) if path_lengths else np.nan


def random_path_null(graph: nx.Graph, universe: List[str], target_genes: List[str], n_source: int, n_iter: int = 100) -> np.ndarray:
    valid = [g for g in universe if g in graph.nodes]
    if len(valid) < n_source:
        return np.array([])
    values = []
    for _ in range(n_iter):
        sampled = RNG.choice(valid, size=n_source, replace=False).tolist()
        values.append(mean_shortest_path(graph, sampled, target_genes))
    return np.array(values, dtype=float)


def maybe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    return pd.read_csv(path) if path.exists() else None


def save_tgfbr2_case(resources: Dict[str, pd.DataFrame], ifn_sig: pd.DataFrame) -> pd.DataFrame:
    links = resources["tgfbr2_links"].copy()
    expr = resources["tgfbr2_expr"].copy()
    support = resources["tgfbr2_support"].copy()
    ifn_top = ifn_sig[ifn_sig["direction"] == "up_in_stim"].groupby("cell_type").head(5)
    tgfbr2_case = links.copy()
    tgfbr2_case["ifn_receiver_gene_overlap"] = tgfbr2_case["top_receiver_overlap_genes"].fillna("")
    tgfbr2_case.to_csv(TABLE_DIR / "tgfbr2_case_candidates.csv", index=False)
    expr.to_csv(TABLE_DIR / "tgfbr2_case_receiver_expression.csv", index=False)
    support.to_csv(TABLE_DIR / "tgfbr2_case_external_support.csv", index=False)
    ifn_top.to_csv(TABLE_DIR / "ifn_receiver_top5_for_case_study.csv", index=False)
    return tgfbr2_case


def main() -> None:
    ensure_dirs()
    rankings = load_rankings()
    resources = load_resources()

    ifn_hybrid_top, ifn_hybrid_genes = top_component_genes(rankings["ifn_hybrid"], n=20)
    lps_hybrid_top, lps_hybrid_genes = top_component_genes(rankings["lps_hybrid"], n=20)
    ifn_contrastive_top, ifn_contrastive_genes = top_component_genes(rankings["ifn_contrastive"], n=20)
    lps_contrastive_top, lps_contrastive_genes = top_component_genes(rankings["lps_contrastive"], n=20)
    comm_top, comm_top_genes = top_component_genes(rankings["comm_candidates"], n=20)

    universe_genes = normalize_gene_list(
        g
        for frame in [rankings["ifn_hybrid"], rankings["lps_hybrid"], rankings["ifn_contrastive"], rankings["lps_contrastive"]]
        for text in frame["perturbation_genes"]
        for g in split_genes(text)
    )
    ifn_receiver_genes = receiver_gene_set(resources["ifn_sig"], top_n_per_cell=8)
    lps_receiver_genes = receiver_gene_set(resources["lps_sig"], top_n_per_cell=8)

    sender_gene_sets = {
        "ifn_hybrid": ifn_hybrid_genes,
        "lps_hybrid": lps_hybrid_genes,
        "ifn_contrastive": ifn_contrastive_genes,
        "lps_contrastive": lps_contrastive_genes,
        "comm_hybrid": comm_top_genes,
    }
    gene_sets = {
        **sender_gene_sets,
        "ifn_receiver": ifn_receiver_genes,
        "lps_receiver": lps_receiver_genes,
    }
    annotation_path = TABLE_DIR / "top_candidate_gene_annotation_table.csv"
    annotation_df = maybe_read_csv(annotation_path)
    if annotation_df is None:
        annotation_df = build_gene_annotation_table(sender_gene_sets, resources)
        annotation_df.to_csv(annotation_path, index=False)

    strict_ligands = set(resources["strict_ligands"]["genesymbol"].astype(str).str.upper())
    strict_receptors = set(resources["strict_receptors"]["genesymbol"].astype(str).str.upper())
    comm_roles = resources["comm_roles"].copy()
    comm_role_genes = set(comm_roles["gene"].astype(str).str.upper())

    enrichment_rows = []
    for name, genes in {
        "ifn_hybrid": ifn_hybrid_genes,
        "lps_hybrid": lps_hybrid_genes,
        "ifn_contrastive": ifn_contrastive_genes,
        "lps_contrastive": lps_contrastive_genes,
        "comm_hybrid": comm_top_genes,
    }.items():
        enrichment_rows.append(fisher_enrichment(genes, universe_genes, comm_role_genes, f"{name}_communication_genes"))
        enrichment_rows.append(fisher_enrichment(genes, universe_genes, strict_ligands, f"{name}_strict_ligands"))
        enrichment_rows.append(fisher_enrichment(genes, universe_genes, strict_receptors, f"{name}_strict_receptors"))
    role_enrichment_path = TABLE_DIR / "top_candidate_role_enrichment_tests.csv"
    role_enrichment_df = pd.DataFrame(enrichment_rows)
    role_enrichment_df.to_csv(role_enrichment_path, index=False)

    enrich_path = TABLE_DIR / "pathway_enrichment_results.csv"
    enrich_df = maybe_read_csv(enrich_path)
    if enrich_df is None:
        all_enrich = []
        for name, genes in {
            "ifn_hybrid_sender": ifn_hybrid_genes,
            "lps_hybrid_sender": lps_hybrid_genes,
            "ifn_contrastive_sender": ifn_contrastive_genes,
            "lps_contrastive_sender": lps_contrastive_genes,
            "ifn_receiver": ifn_receiver_genes,
            "lps_receiver": lps_receiver_genes,
        }.items():
            all_enrich.append(run_enrichr(genes, name))
        enrich_df = pd.concat([df for df in all_enrich if not df.empty], ignore_index=True)
        enrich_df.to_csv(enrich_path, index=False)

    hallmark_sets = {}
    for name in enrich_df["analysis_set"].dropna().unique():
        hallmark_sets[name] = select_significant_hallmarks(enrich_df[enrich_df["analysis_set"] == name])
    pathway_overlap_rows = []
    for sender_name in ["ifn_hybrid_sender", "lps_hybrid_sender", "ifn_contrastive_sender", "lps_contrastive_sender"]:
        for receiver_name in ["ifn_receiver", "lps_receiver"]:
            sender_terms = hallmark_sets.get(sender_name, set())
            receiver_terms = hallmark_sets.get(receiver_name, set())
            shared = sorted(sender_terms & receiver_terms)
            pathway_overlap_rows.append(
                {
                    "sender_set": sender_name,
                    "receiver_set": receiver_name,
                    "n_sender_hallmarks": len(sender_terms),
                    "n_receiver_hallmarks": len(receiver_terms),
                    "n_shared_hallmarks": len(shared),
                    "shared_hallmarks": ";".join(shared),
                }
            )
    pathway_overlap_path = TABLE_DIR / "sender_receiver_shared_hallmarks.csv"
    pathway_overlap_df = pd.DataFrame(pathway_overlap_rows)
    pathway_overlap_df.to_csv(pathway_overlap_path, index=False)

    network_sender_genes = normalize_gene_list(ifn_hybrid_genes + ifn_contrastive_genes + ["TGFBR2"])
    network_receiver_genes = normalize_gene_list(ifn_receiver_genes[:12] + ["STAT1", "IRF7", "IRF9", "ISG15", "MX1"])
    string_path = TABLE_DIR / "string_sender_receiver_network_edges.csv"
    string_df = maybe_read_csv(string_path)
    if string_df is None:
        string_df = fetch_string_network(normalize_gene_list(network_sender_genes + network_receiver_genes))
        string_df.to_csv(string_path, index=False)
    graph = build_graph(string_df)

    network_sender_in_graph = [g for g in network_sender_genes if g in graph.nodes]
    network_receiver_in_graph = [g for g in network_receiver_genes if g in graph.nodes]
    observed_path = mean_shortest_path(graph, network_sender_in_graph, network_receiver_in_graph)
    null_paths = random_path_null(
        graph,
        universe_genes + network_receiver_in_graph,
        network_receiver_in_graph,
        len(network_sender_in_graph),
        n_iter=50,
    )
    network_summary = pd.DataFrame(
        [
            {
                "observed_mean_shortest_path": observed_path,
                "null_mean_shortest_path": float(np.nanmean(null_paths)) if null_paths.size else np.nan,
                "null_std_shortest_path": float(np.nanstd(null_paths)) if null_paths.size else np.nan,
                "empirical_pvalue": float((np.sum(null_paths <= observed_path) + 1) / (len(null_paths) + 1)) if null_paths.size else np.nan,
                "sender_gene_count": len(network_sender_in_graph),
                "receiver_gene_count": len(network_receiver_in_graph),
                "graph_nodes": graph.number_of_nodes(),
                "graph_edges": graph.number_of_edges(),
            }
        ]
    )
    network_summary.to_csv(TABLE_DIR / "string_network_shortest_path_summary.csv", index=False)
    pd.DataFrame({"null_mean_shortest_path": null_paths}).to_csv(TABLE_DIR / "string_network_null_distribution.csv", index=False)

    tgfbr2_case = save_tgfbr2_case(resources, resources["ifn_sig"])

    top_candidate_table = pd.concat(
        [
            ifn_hybrid_top.assign(head="hybrid", context="ifn"),
            lps_hybrid_top.assign(head="hybrid", context="lps"),
            ifn_contrastive_top.assign(head="contrastive", context="ifn"),
            lps_contrastive_top.assign(head="contrastive", context="lps"),
        ],
        ignore_index=True,
    )
    top_candidate_table.to_csv(TABLE_DIR / "top20_candidates_all_heads_contexts.csv", index=False)

    summary_lines = [
        "Biological interpretation and network validation complete.",
        f"Annotated genes: {annotation_df.shape[0]}",
        f"IFN hybrid top-gene count: {len(ifn_hybrid_genes)}",
        f"LPS hybrid top-gene count: {len(lps_hybrid_genes)}",
        f"IFN contrastive top-gene count: {len(ifn_contrastive_genes)}",
        f"LPS contrastive top-gene count: {len(lps_contrastive_genes)}",
        f"Observed STRING mean shortest path: {network_summary['observed_mean_shortest_path'].iloc[0]:.3f}",
        f"STRING empirical p-value: {network_summary['empirical_pvalue'].iloc[0]:.4f}",
        f"TGFBR2-linked candidates saved: {tgfbr2_case.shape[0]}",
    ]
    (OUT_DIR / "run_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
