param(
    [switch]$SmokeOnly,
    [int]$Epochs = 20,
    [int]$SamplesPerEpoch = 360,
    [int]$PatchSize = 256
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$DetectorRoot = "E:\TRACE_R_experiments\crid320_detector_v1_20260829"
$Detector = Join-Path $DetectorRoot "weights\best.pt"
$DetectorFreeze = Join-Path $DetectorRoot "detector_selection_freeze.json"
$OutRoot = "E:\TRACE_R_experiments\crid320_matched_restorers_20260829"

if (-not (Test-Path $DetectorFreeze)) {
    throw "The validation-selected CRID-320 detector is not frozen yet: $DetectorFreeze"
}
if (-not (Test-Path $Detector)) {
    throw "Missing validation-selected detector: $Detector"
}

$Models = @(
    @{
        Name = "rmrp"
        Init = "E:\TRACE_R_experiments\matched_budget_trace_v53_20260827\train\rmrp_epoch_008.pth"
        Extra = @()
    },
    @{
        Name = "demoe"
        Init = Join-Path $Root "experiments\matched_final_candidate_index_v28_epoch70_20260821\demoe\demoe_epoch_070.pth"
        Extra = @()
    },
    @{
        Name = "nafnet"
        Init = "E:\TRACE_R_experiments\official_nafnet_matched_v68_20260828\train\nafnet_epoch_028.pth"
        Extra = @()
    },
    @{
        Name = "dfpir"
        Init = Join-Path $Root "experiments\matched_final_candidate_index_v28_epoch70_20260821\dfpir\dfpir_epoch_070.pth"
        Extra = @()
    },
    @{
        Name = "instructir"
        Init = Join-Path $Root "experiments\matched_final_candidate_index_v28_epoch70_20260821\instructir\instructir_epoch_070.pth"
        Extra = @("--lm-head-weights", (Join-Path $Root "weights\instructir\lm_instructir-7d.pt"))
    }
)

foreach ($Spec in $Models) {
    if (-not (Test-Path $Spec.Init)) {
        throw "Missing initial checkpoint for $($Spec.Name): $($Spec.Init)"
    }
    $Out = Join-Path $OutRoot $Spec.Name
    $Complete = Join-Path $Out "adaptation_complete.json"
    if ((-not $SmokeOnly) -and (Test-Path $Complete)) {
        Write-Host "[CRID-320] skipping completed $($Spec.Name)"
        continue
    }
    $Args = @(
        (Join-Path $Root "tools\train_crid320_restorer.py"),
        "--model", $Spec.Name,
        "--init-weights", $Spec.Init,
        "--detector", $Detector,
        "--out", $Out,
        "--epochs", $Epochs,
        "--samples-per-epoch", $SamplesPerEpoch,
        "--patch-size", $PatchSize,
        "--save-every", 4,
        "--device", "cuda"
    ) + $Spec.Extra
    if ($Spec.Name -in @("dfpir", "instructir")) {
        $Args += @("--amp")
    }
    if (-not $SmokeOnly) {
        $Resume = Get-ChildItem $Out -Filter "$($Spec.Name)_field_epoch_*.pth" -ErrorAction SilentlyContinue |
            Sort-Object Name -Descending |
            Select-Object -First 1
        if ($null -ne $Resume) {
            Write-Host "[CRID-320] resuming $($Spec.Name) from $($Resume.FullName)"
            $Args += @("--resume-from", $Resume.FullName)
        }
    }
    if ($SmokeOnly) {
        $Args += @("--smoke-steps", 1)
        $Args[$Args.IndexOf("--out") + 1] = Join-Path $OutRoot ("smoke_" + $Spec.Name)
    }
    Write-Host "[CRID-320] training $($Spec.Name)"
    & $Python @Args
    if ($LASTEXITCODE -ne 0) {
        throw "CRID-320 adaptation failed for $($Spec.Name)"
    }
}

Write-Host "CRID-320 matched adaptation completed. Test split remains sealed."
