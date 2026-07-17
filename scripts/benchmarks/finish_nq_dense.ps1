param([Parameter(Mandatory = $true)][int]$DensePid)

$ErrorActionPreference = "Stop"
$root = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$artifactRoot = Join-Path $root "artifacts\benchmarks\natural_questions"
$retrievalRoot = Join-Path $artifactRoot "retrieval"
$completionPath = Join-Path $artifactRoot "indexes\e5-small-v2-candidates\followup-completion.json"
$python = (Get-Command python.exe -ErrorAction Stop).Source

Wait-Process -Id $DensePid -ErrorAction SilentlyContinue

$denseReport = Join-Path $retrievalRoot "natural_questions-e5-small-v2.json"
$hybridReport = Join-Path $retrievalRoot "natural_questions-bm25-e5-rrf.json"
$bm25Report = Join-Path $root "reports\public-benchmarks\retrieval\natural_questions-bm25.json"
$publishedDense = Join-Path $root "reports\public-benchmarks\retrieval\natural_questions-e5-small-v2.json"
$publishedHybrid = Join-Path $root "reports\public-benchmarks\retrieval\natural_questions-bm25-e5-rrf.json"

if (-not (Test-Path -LiteralPath $denseReport)) {
  throw "Natural Questions dense report was not produced."
}

Push-Location $root
try {
  & $python -m scripts.benchmarks.run_hybrid_retrieval --benchmark natural_questions
  if ($LASTEXITCODE -ne 0) { throw "Natural Questions hybrid scoring failed." }

  & $python -m scripts.benchmarks.publish_retrieval $denseReport $hybridReport
  if ($LASTEXITCODE -ne 0) { throw "Natural Questions report publication failed." }

  & $python -m scripts.benchmarks.compare_retrieval_gate `
    --baseline $bm25Report --candidate $publishedDense *>&1 |
    Set-Content -LiteralPath (Join-Path $retrievalRoot "e5-vs-bm25-gate.log")
  $e5Gate = $LASTEXITCODE

  & $python -m scripts.benchmarks.compare_retrieval_gate `
    --baseline $publishedDense --candidate $publishedHybrid *>&1 |
    Set-Content -LiteralPath (Join-Path $retrievalRoot "hybrid-vs-e5-gate.log")
  $hybridGate = $LASTEXITCODE

  $previousErrorAction = $ErrorActionPreference
  $ErrorActionPreference = "Continue"
  & $python -m unittest discover -s tests -p "test_*.py" 2>&1 |
    Set-Content -LiteralPath (Join-Path $retrievalRoot "verification.log")
  $tests = $LASTEXITCODE
  $ErrorActionPreference = $previousErrorAction

  & $python -m compileall -q rag scripts tests
  $compile = $LASTEXITCODE
  & $python -m scripts.benchmarks.check
  $contracts = $LASTEXITCODE
  & $python -m scripts.benchmarks.audit_suite
  $suiteAudit = $LASTEXITCODE
  & git diff --check
  $diffCheck = $LASTEXITCODE

  $status = if (
    $tests -eq 0 -and $compile -eq 0 -and $contracts -eq 0 -and
    $suiteAudit -eq 0 -and $diffCheck -eq 0
  ) { "complete" } else { "verification_failed" }

  [ordered]@{
    status = $status
    completed_at = (Get-Date).ToUniversalTime().ToString("o")
    dense_report = $denseReport
    hybrid_report = $hybridReport
    e5_vs_bm25_gate_exit = $e5Gate
    hybrid_vs_e5_gate_exit = $hybridGate
    tests_exit = $tests
    compile_exit = $compile
    contracts_exit = $contracts
    suite_audit_exit = $suiteAudit
    diff_check_exit = $diffCheck
  } | ConvertTo-Json | Set-Content -LiteralPath $completionPath
} finally {
  Pop-Location
}
