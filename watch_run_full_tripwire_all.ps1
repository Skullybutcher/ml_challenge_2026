param(
    [Parameter(Mandatory = $true)][int]$RootProcessId,
    [Parameter(Mandatory = $true)][string]$RunLogPath,
    [Parameter(Mandatory = $true)][string]$TripwireLogPath,
    [Parameter(Mandatory = $true)][string]$TripwireMarkerPath,
    [long]$PairLimit = 50000000,
    [int]$IntervalSeconds = 2
)

$ErrorActionPreference = 'SilentlyContinue'
$checkedChunks = [System.Collections.Generic.HashSet[int]]::new()
Add-Content -LiteralPath $TripwireLogPath -Value "all-test-chunk tripwire start root=$RootProcessId limit_pairs=$PairLimit time=$([DateTime]::Now.ToString('o'))"

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

while (Get-Process -Id $RootProcessId -ErrorAction SilentlyContinue) {
    if (Test-Path -LiteralPath $RunLogPath) {
        $lines = @(Get-Content -LiteralPath $RunLogPath)
        $testStart = -1
        for ($i = 0; $i -lt $lines.Count; $i++) {
            if ($lines[$i] -match 'Test inference in') { $testStart = $i }
        }
        if ($testStart -ge 0) {
            for ($i = $testStart + 1; $i -lt $lines.Count; $i++) {
                if ($lines[$i] -match 'chunk\s+(\d+)/(\d+): pairs=([\d,]+)') {
                    $chunkIndex = [int]$Matches[1]
                    if ($checkedChunks.Add($chunkIndex)) {
                        $chunkCount = [int]$Matches[2]
                        $pairCount = [long](($Matches[3] -replace ',', ''))
                        if ($pairCount -gt $PairLimit) {
                            $message = "TRIPWIRE: test chunk $chunkIndex/$chunkCount has $pairCount pairs (> $PairLimit); stopping to protect the run. time=$([DateTime]::Now.ToString('o'))"
                            Set-Content -LiteralPath $TripwireMarkerPath -Value $message
                            Add-Content -LiteralPath $TripwireLogPath -Value $message
                            Stop-ProcessTree -Root $RootProcessId
                            exit 2
                        }
                        Add-Content -LiteralPath $TripwireLogPath -Value "test chunk $chunkIndex/$chunkCount pairs=$pairCount limit=$PairLimit; tripwire passed time=$([DateTime]::Now.ToString('o'))"
                    }
                }
            }
        }
    }
    Start-Sleep -Seconds $IntervalSeconds
}
Add-Content -LiteralPath $TripwireLogPath -Value "all-test-chunk tripwire end root=$RootProcessId time=$([DateTime]::Now.ToString('o'))"