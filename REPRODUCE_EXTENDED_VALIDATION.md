# Reproduce Extended Validation

This document covers the four additions requested for the submission-strengthening round:

1. A second receiver signaling context based on `GSE226488` LPS-stimulated PBMCs
2. Sender random-split stability analysis
3. A nearest-neighbor baseline comparison
4. A one-command reproducibility entrypoint
5. A context-sensitive contrastive reranking head
6. Publication-ready vector figure generation

## Required local inputs

- [11_build_gse96583_receiver_model.py](C:/Users/Admin/Documents/Codex/2026-04-25-a-b-ai-1-perturb-seq/11_build_gse96583_receiver_model.py)
- [E:/扰动/GSE133344_outputs](E:/扰动/GSE133344_outputs)
- [E:/扰动/GSE133344](E:/扰动/GSE133344)
- [GSM7077865_D1_filtered_feature_bc_matrix.tar.gz](C:/Users/Admin/Documents/Codex/2026-04-25-a-b-ai-1-perturb-seq/GSM7077865_D1_filtered_feature_bc_matrix.tar.gz)
- [GSM7077866_G1_filtered_feature_bc_matrix.tar.gz](C:/Users/Admin/Documents/Codex/2026-04-25-a-b-ai-1-perturb-seq/GSM7077866_G1_filtered_feature_bc_matrix.tar.gz)
- [gse226488_series_matrix.txt.gz](C:/Users/Admin/Documents/Codex/2026-04-25-a-b-ai-1-perturb-seq/gse226488_series_matrix.txt.gz)

## One-command run

From the workspace root:

```powershell
.\run_extended_validation_pipeline.ps1
```

## Step-by-step scripts

```powershell
python 18_build_gse226488_lps_receiver_model.py
python 19_sender_random_split_stability.py
python 20_compare_knn_baseline.py
python 21_build_extension_bundle.py
python 22_commbio_analysis_strengthening.py
python 23_context_sensitive_linking_head.py
python 24_generate_publication_figures.py
```

## Key outputs

- LPS receiver module: [gse226488_lps_receiver_outputs](C:/Users/Admin/Documents/Codex/2026-04-25-a-b-ai-1-perturb-seq/gse226488_lps_receiver_outputs)
- Sender stability: [sender_split_stability_outputs](C:/Users/Admin/Documents/Codex/2026-04-25-a-b-ai-1-perturb-seq/sender_split_stability_outputs)
- kNN baseline: [knn_baseline_comparison_outputs](C:/Users/Admin/Documents/Codex/2026-04-25-a-b-ai-1-perturb-seq/knn_baseline_comparison_outputs)
- Aggregate bundle: [extension_bundle_outputs](C:/Users/Admin/Documents/Codex/2026-04-25-a-b-ai-1-perturb-seq/extension_bundle_outputs)
- Context-sensitive head: [context_sensitive_linking_outputs](C:/Users/Admin/Documents/Codex/2026-04-25-a-b-ai-1-perturb-seq/context_sensitive_linking_outputs)
- Publication figures: [publication_figures](C:/Users/Admin/Documents/Codex/2026-04-25-a-b-ai-1-perturb-seq/publication_figures)
