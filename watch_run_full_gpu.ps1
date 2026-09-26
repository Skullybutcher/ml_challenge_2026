param(
    [Parameter(Mandatory = $true)][int]$RootProcessId,
    [Parameter(Mandatory = $true)][string]$ThermalLogPath,
    [double]$GpuLimitC = 82.0,
    [int]$IntervalSeconds = 15
)

$ErrorActionPreference = 'SilentlyContinue'
$nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
Add-Content -LiteralPath $ThermalLogPath -Value "GPU thermal watch start root=$RootProcessId limit_c=$GpuLimitC interval_s=$IntervalSeconds sensor=$(if ($nvidiaSmi) { 'nvidia-smi' } else { 'unavailable' }) time=$([DateTime]::Now.ToString('o'))"

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
    if ($nvidiaSmi) {
        $reading = (& $nvidiaSmi.Source --query-gpu=name,temperature.gpu,utilization.gpu,memory.used --format=csv,noheader 2>$null | Select-Object -First 1)
        if ($reading -match '^\s*([^,]+),\s*(\d+)\s*,\s*(\d+)\s*%\s*,\s*([^,]+)') {
            $gpuTemp = [double]$Matches[2]
            Add-Content -LiteralPath $ThermalLogPath -Value "$([DateTime]::Now.ToString('o')) gpu=$($Matches[1].Trim()) temp_c=$gpuTemp util_pct=$($Matches[3]) memory=$($Matches[4].Trim())"
            if ($gpuTemp -ge $GpuLimitC) {
                $message = "GPU thermal watchdog stopping tree at temp_c=$gpuTemp limit_c=$GpuLimitC time=$([DateTime]::Now.ToString('o'))"
                Add-Content -LiteralPath $ThermalLogPath -Value $message
                Stop-ProcessTree -Root $RootProcessId
                break
            }
        } else {
            Add-Content -LiteralPath $ThermalLogPath -Value "GPU sensor query returned an unreadable value: $reading time=$([DateTime]::Now.ToString('o'))"
        }
    } else {
        Add-Content -LiteralPath $ThermalLogPath -Value "GPU temperature sensor unavailable time=$([DateTime]::Now.ToString('o'))"
    }
    Start-Sleep -Seconds $IntervalSeconds
}
Add-Content -LiteralPath $ThermalLogPath -Value "GPU thermal watch end root=$RootProcessId time=$([DateTime]::Now.ToString('o'))"
