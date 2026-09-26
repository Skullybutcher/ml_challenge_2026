$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $repoRoot
$pythonPath = (Resolve-Path -LiteralPath (Join-Path $repoRoot '..\venv_py311\Scripts\python.exe')).Path
$dataPath = (Resolve-Path -LiteralPath (Join-Path $repoRoot '..\dataset')).Path
$scriptPath = Join-Path $repoRoot 'code\business_entity_resolution\src\pipeline.py'
$affinityRunner = Join-Path $repoRoot 'utils\run_with_cpu_affinity.py'
$outPath = [System.IO.Path]::GetFullPath((Join-Path $repoRoot 'out_100k'))
$logPath = Join-Path $repoRoot 'run_100k.log'
$exitPath = Join-Path $repoRoot 'run_100k.exit'

$activeRun = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -and $_.CommandLine.Contains($scriptPath) -and $_.CommandLine.Contains($outPath) }
if ($activeRun) {
    throw "A pipeline run for $outPath is already active (PID $($activeRun.ProcessId)); refusing a duplicate launch."
}

$attemptStamp = Get-Date -Format 'yyyyMMdd_HHmmss'
if (Test-Path -LiteralPath $logPath) {
    Move-Item -LiteralPath $logPath -Destination (Join-Path $repoRoot "run_100k_attempt_$attemptStamp.log")
}
if (Test-Path -LiteralPath $exitPath) {
    Move-Item -LiteralPath $exitPath -Destination (Join-Path $repoRoot "run_100k_attempt_$attemptStamp.exit")
}

$env:PYTHONHASHSEED = '42'
$env:PYTHONFAULTHANDLER = '1'

# Windows PowerShell treats native stderr as error records. Continue collecting
# those records so Python's traceback and exit code survive a failed run.
$ErrorActionPreference = 'Continue'
& $pythonPath $affinityRunner --mask 0xffff0000 $scriptPath `
    --data-dir $dataPath `
    --out-dir $outPath `
    --sample-s1 100000 `
    --skip-test `
    --n-splits 5 `
    --train-chunk-size 2500 `
    --use-rare `
    --resume-train-chunks *> $logPath
$runExitCode = $LASTEXITCODE
$ErrorActionPreference = 'Stop'
Set-Content -LiteralPath $exitPath -Value $runExitCode -Encoding ascii
exit $runExitCode
