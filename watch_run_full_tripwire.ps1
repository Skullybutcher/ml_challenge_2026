param(
    [Parameter(Mandatory = $true)][int]$RootProcessId,
    [Parameter(Mandatory = $true)][string]$RunLogPath,
    [Parameter(Mandatory = $true)][string]$TripwireLogPath,
    [Parameter(Mandatory = $true)][string]$TripwireMarkerPath,
    [long]$PairLimit = 50000000,
    [int]$IntervalSeconds = 2
)

$ErrorActionPreference = 'SilentlyContinue'
Add-Content -LiteralPath $TripwireLogPath -Value "tripwire start root=$RootProcessId limit_pairs=$PairLimit time=$([DateTime]::Now.ToString('o'))"

function Stop-ProcessTree {
    param([int]$Root)
    $all = @(Get-CimInstance Win32_Process)
    $ids = [System.Collections.Generic.HashSet[int]]::new()
    [void]$ids.Add($Root)
    $changed = $true
    while ($changed) {
        $changed = $false
        foreach ($proc in $all) {
            $childId = [int]$proc.ProcessId
            if ($ids.Contains([int]$proc.ParentProcessId) -and -not $ids.Contains($childId)) {
                [void]$ids.Add($childId)
                $changed = $true
            }
        }
    }
    foreach ($processId in $ids) {
        if ($processId -ne $Root) { Stop-Process -Id $processId -Force }
    }
    Stop-Process -Id $Root -Force
}

while ($true) {
    if (-not (Get-Process -Id $RootProcessId -ErrorAction SilentlyContinue)) { break }
    if (Test-Path -LiteralPath $RunLogPath) {
        $lines = @(Get-Content -LiteralPath $RunLogPath)
        $testStart = -1
        for ($i = 0; $i -lt $lines.Count; $i++) {
            if ($lines[$i] -match 'Test inference in') { $testStart = $i }
        }
        if ($testStart -ge 0) {
            for ($i = $testStart + 1; $i -lt $lines.Count; $i++) {
                if ($lines[$i] -match 'chunk 1/\d+: pairs=([\d,]+)') {
                    $pairCount = [long](($Matches[1] -replace ',', ''))
                    if ($pairCount -gt $PairLimit) {
                        $message = "TRIPWIRE: first test chunk has $pairCount pairs (> $PairLimit); stopping to protect the run. time=$([DateTime]::Now.ToString('o'))"
                        Set-Content -LiteralPath $TripwireMarkerPath -Value $message
                        Add-Content -LiteralPath $TripwireLogPath -Value $message
                        Stop-ProcessTree -Root $RootProcessId
                        exit 2
                    }
                    Add-Content -LiteralPath $TripwireLogPath -Value "first test chunk pairs=$pairCount limit=$PairLimit; tripwire passed time=$([DateTime]::Now.ToString('o'))"
                    exit 0
                }
            }
        }
    }
    Start-Sleep -Seconds $IntervalSeconds
}
Add-Content -LiteralPath $TripwireLogPath -Value "tripwire end root=$RootProcessId time=$([DateTime]::Now.ToString('o'))"
