$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

function Run-Step {
    param(
        [string]$Label,
        [string]$Command
    )
    Write-Host ""
    Write-Host "=== $Label ==="
    Invoke-Expression $Command
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed with exit code ${LASTEXITCODE}: $Label"
    }
}

Run-Step "Receiver IFN module" "python 11_build_gse96583_receiver_model.py"
Run-Step "Communication prior" "python 13_build_communication_prior.py"
Run-Step "Sender-receiver IFN linking" "python 12_link_sender_to_receiver_program.py"
Run-Step "Benchmark summary" "python 14_build_method_benchmark.py"
Run-Step "Observed-vs-predicted validation" "python 15_validate_linking_and_curate_results.py"
Run-Step "Ligand refinement" "python 16_ligand_refinement_external_validation.py"
Run-Step "Receptor-focused support" "python 17_receptor_focused_support.py"
Run-Step "Receiver LPS module" "python 18_build_gse226488_lps_receiver_model.py"
Run-Step "Sender split stability" "python 19_sender_random_split_stability.py"
Run-Step "kNN baseline comparison" "python 20_compare_knn_baseline.py"
Run-Step "Extension bundle" "python 21_build_extension_bundle.py"
Run-Step "CommBio strengthening" "python 22_commbio_analysis_strengthening.py"
Run-Step "Context-sensitive contrastive head" "python 23_context_sensitive_linking_head.py"
Run-Step "Publication figure generation" "python 24_generate_publication_figures.py"

Write-Host ""
Write-Host "All extended validation steps completed."
