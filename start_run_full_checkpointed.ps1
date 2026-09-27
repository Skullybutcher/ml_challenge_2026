param(
    [ValidateRange(1, 1000000)][int]$TestChunkSize = 12500
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $repoRoot

$pythonPath = (Resolve-Path -LiteralPath (Join-Path $repoRoot '..\venv_py311\Scripts\python.exe')).Path
$dataPath = (Resolve-Path -LiteralPath (Join-Path $repoRoot '..\dataset')).Path
$scriptPath = Join-Path $repoRoot 'utils\run_pipeline_plateau.py'
$affinityRunner = Join-Path $repoRoot 'utils\run_with_cpu_affinity.py'
$outPath = [System.IO.Path]::GetFullPath((Join-Path $repoRoot 'out_full'))
$logPath = Join-Path $repoRoot 'run_full.log'
$exitPath = Join-Path $repoRoot 'run_full.exit'

$activeRun = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -and $_.CommandLine.Contains($scriptPath) -and $_.CommandLine.Contains($outPath) }
if ($activeRun) {
    throw "A pipeline run for $outPath is already active (PID $($activeRun.ProcessId)); refusing a duplicate launch."
}

$attemptStamp = Get-Date -Format 'yyyyMMdd_HHmmss'
foreach ($path in @($logPath, $exitPath)) {
    if (Test-Path -LiteralPath $path) {
        $extension = [System.IO.Path]::GetExtension($path)
        $baseName = [System.IO.Path]::GetFileNameWithoutExtension($path)
        Move-Item -LiteralPath $path -Destination (Join-Path $repoRoot "${baseName}_attempt_${attemptStamp}${extension}")
    }
}

$env:PYTHONHASHSEED = '42'
$env:PYTHONFAULTHANDLER = '1'
$env:PYTHONUNBUFFERED = '1'
$ErrorActionPreference = 'Continue'
& $pythonPath $affinityRunner --mask 0xffff0000 $scriptPath `
    --data-dir $dataPath `
    --out-dir $outPath `
    --sample-s1 100000 `
    --n-splits 5 `
    --train-chunk-size 2500 `
    --test-chunk-size $TestChunkSize `
    --use-rare `
    --rare-max-df 1000 `
    --resume-train-chunks *> $logPath
$runExitCode = $LASTEXITCODE
$ErrorActionPreference = 'Stop'
Set-Content -LiteralPath $exitPath -Value $runExitCode -Encoding ascii
exit $runExitCode
