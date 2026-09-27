param(
    [ValidateRange(1, 1000000)][int]$TestChunkSize = 6250
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $repoRoot

$runLauncher = Join-Path $repoRoot 'start_run_full_checkpointed.ps1'
$memoryWatcher = Join-Path $repoRoot 'utils\watch_process_tree.ps1'
$tripwireWatcher = Join-Path $repoRoot 'watch_run_full_tripwire_all.ps1'
$gpuWatcher = Join-Path $repoRoot 'watch_run_full_gpu.ps1'
$monitorLog = Join-Path $repoRoot 'run_full.monitor.log'
$tripwireLog = Join-Path $repoRoot 'run_full.tripwire.log'
$tripwireMarker = Join-Path $repoRoot 'run_full.tripwire'
$thermalLog = Join-Path $repoRoot 'run_full.thermal.log'
$pidPath = Join-Path $repoRoot 'run_full.pid'
$watcherPidPath = Join-Path $repoRoot 'run_full.watcher.pid'
$tripwirePidPath = Join-Path $repoRoot 'run_full.tripwire.pid'
$thermalPidPath = Join-Path $repoRoot 'run_full.thermal.pid'
$pwsh = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'

if (Test-Path -LiteralPath (Join-Path $repoRoot 'run_full.exit')) {
    throw 'run_full.exit exists; review or archive the prior attempt before launching again.'
}
if (Test-Path -LiteralPath $tripwireMarker) {
    throw 'run_full.tripwire exists; review the recorded pair-count tripwire before launching again.'
}

$run = Start-Process -FilePath $pwsh -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$runLauncher`" -TestChunkSize $TestChunkSize" -WorkingDirectory $repoRoot -WindowStyle Hidden -PassThru
Set-Content -LiteralPath $pidPath -Value $run.Id -Encoding ascii
Start-Sleep -Seconds 2

$memory = Start-Process -FilePath $pwsh -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$memoryWatcher`" -RootProcessId $($run.Id) -LogPath `"$monitorLog`" -TreeLimitGB 44 -SystemLimitGB 47 -IntervalSeconds 1" -WorkingDirectory $repoRoot -WindowStyle Hidden -PassThru
$tripwire = Start-Process -FilePath $pwsh -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$tripwireWatcher`" -RootProcessId $($run.Id) -RunLogPath `"$(Join-Path $repoRoot 'run_full.log')`" -TripwireLogPath `"$tripwireLog`" -TripwireMarkerPath `"$tripwireMarker`" -PairLimit 50000000 -IntervalSeconds 2" -WorkingDirectory $repoRoot -WindowStyle Hidden -PassThru
$gpu = Start-Process -FilePath $pwsh -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$gpuWatcher`" -RootProcessId $($run.Id) -ThermalLogPath `"$thermalLog`" -GpuLimitC 82 -IntervalSeconds 15" -WorkingDirectory $repoRoot -WindowStyle Hidden -PassThru
Set-Content -LiteralPath $watcherPidPath -Value $memory.Id -Encoding ascii
Set-Content -LiteralPath $tripwirePidPath -Value $tripwire.Id -Encoding ascii
Set-Content -LiteralPath $thermalPidPath -Value $gpu.Id -Encoding ascii

Start-Sleep -Seconds 3
$runProcess = Get-Process -Id $run.Id -ErrorAction SilentlyContinue
if (-not $runProcess) { throw "Run 2 launcher PID $($run.Id) stopped during startup; inspect run_full.log and run_full.exit." }
Write-Output "Run 2 launcher PID=$($run.Id); test chunk size=$TestChunkSize; memory watchdog PID=$($memory.Id); all-chunk pair tripwire PID=$($tripwire.Id); GPU thermal watchdog PID=$($gpu.Id)."
