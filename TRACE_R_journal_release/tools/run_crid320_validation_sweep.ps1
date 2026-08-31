param(
    [string]$OutRoot = "E:\TRACE_R_experiments\crid320_restorer_validation_20260829"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Evaluator = Join-Path $Root "tools\evaluate_crid320_validation.py"
$TrainRoot = "E:\TRACE_R_experiments\crid320_matched_restorers_20260829"
$Detector = "E:\TRACE_R_experiments\crid320_detector_v1_20260829\weights\best.pt"
$LmHead = Join-Path $Root "weights\instructir\lm_instructir-7d.pt"
$Epochs = @(4, 8, 12, 16, 20)

$Models = @(
    @{ Name = "rmrp"; Prefix = "rmrp"; Extra = @() },
    @{ Name = "demoe"; Prefix = "demoe"; Extra = @() },
    @{ Name = "nafnet"; Prefix = "nafnet"; Extra = @() },
    @{ Name = "dfpir"; Prefix = "dfpir"; Extra = @() },
    @{ Name = "instructir"; Prefix = "instructir"; Extra = @("--lm-head-weights", $LmHead) }
)

foreach ($Spec in $Models) {
    foreach ($Epoch in $Epochs) {
        $Checkpoint = Join-Path $TrainRoot ("{0}\{1}_field_epoch_{2:d3}.pth" -f $Spec.Name, $Spec.Prefix, $Epoch)
        if (-not (Test-Path $Checkpoint)) {
            throw "Missing CRID-320 checkpoint: $Checkpoint"
        }
        $Stem = [System.IO.Path]::GetFileNameWithoutExtension($Checkpoint)
        $Result = Join-Path $OutRoot ("{0}\{1}\validation_selection.json" -f $Spec.Name, $Stem)
        if (Test-Path $Result) {
            Write-Host "[CRID-320 validation] skipping cached $($Spec.Name) epoch $Epoch"
            continue
        }
        Write-Host "[CRID-320 validation] $($Spec.Name) epoch $Epoch eta=1.0"
        $Args = @(
            $Evaluator,
            "--model", $Spec.Name,
            "--checkpoint", $Checkpoint,
            "--detector", $Detector,
            "--out", $OutRoot,
            "--eta", "1.0",
            "--device", "cuda"
        ) + $Spec.Extra
        & $Python @Args
        if ($LASTEXITCODE -ne 0) {
            throw "Validation failed for $($Spec.Name) epoch $Epoch"
        }
    }
}

Write-Host "CRID-320 eta=1 checkpoint sweep complete. Test split remains sealed."
