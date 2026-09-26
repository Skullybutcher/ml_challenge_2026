param(
    [Parameter(Mandatory = $true)][int]$RootProcessId,
    [Parameter(Mandatory = $true)][string]$LogPath,
    [double]$TreeLimitGB = 44.0,
    [double]$SystemLimitGB = 47.0,
    [int]$IntervalSeconds = 1
)

$ErrorActionPreference = 'Stop'
$peakTree = 0.0
$peakSystem = 0.0

function Write-MonitorLine {
    param([Parameter(Mandatory = $true)][string]$Line)
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($Line + [Environment]::NewLine)
    for ($attempt = 0; $attempt -lt 10; $attempt++) {
        $stream = $null
        try {
            $stream = [System.IO.File]::Open(
                $LogPath,
                [System.IO.FileMode]::Append,
                [System.IO.FileAccess]::Write,
                [System.IO.FileShare]::ReadWrite
            )
            $stream.Write($bytes, 0, $bytes.Length)
            $stream.Flush()
            return $true
        } catch [System.IO.IOException] {
            if ($attempt -lt 9) { Start-Sleep -Milliseconds 100 }
        } finally {
            if ($null -ne $stream) { $stream.Dispose() }
        }
    }
    # A transient monitor-log lock must not stop the training process.
    return $false
}

if ($TreeLimitGB -le 0 -or $SystemLimitGB -le 0 -or $IntervalSeconds -le 0) {
    throw 'Memory limits and polling interval must be positive.'
}
[void](Write-MonitorLine "watch start root=$RootProcessId time=$([DateTime]::Now.ToString('o')) tree_limit_gb=$TreeLimitGB system_limit_gb=$SystemLimitGB interval_s=$IntervalSeconds")
$hasSample = $false

while ($true) {
    try {
        $all = @(Get-CimInstance Win32_Process -ErrorAction Stop)
        $root = $all | Where-Object { $_.ProcessId -eq $RootProcessId }
        if (-not $root) {
            if (-not $hasSample) {
                throw "Root PID $RootProcessId was absent before the first successful sample."
            }
            [void](Write-MonitorLine "watch end root=$RootProcessId peak_tree_rss_gb=$([math]::Round($peakTree,2)) peak_system_used_gb=$([math]::Round($peakSystem,2)) time=$([DateTime]::Now.ToString('o'))")
            break
        }

        $ids = [System.Collections.Generic.HashSet[int]]::new()
        [void]$ids.Add($RootProcessId)
        $changed = $true
        while ($changed) {
            $changed = $false
            foreach ($process in $all) {
                $childId = [int]$process.ProcessId
                if ($ids.Contains([int]$process.ParentProcessId) -and -not $ids.Contains($childId)) {
                    [void]$ids.Add($childId)
                    $changed = $true
                }
            }
        }

        $tree = @($all | Where-Object { $ids.Contains([int]$_.ProcessId) })
        $treeRssBytes = ($tree | Measure-Object -Property WorkingSetSize -Sum).Sum
        if ($null -eq $treeRssBytes -or $tree.Count -eq 0) {
            throw 'Process-tree memory query returned no values.'
        }
        $treeRss = $treeRssBytes / 1GB
        $os = Get-CimInstance Win32_OperatingSystem -ErrorAction Stop
        if ($null -eq $os -or $os.TotalVisibleMemorySize -le 0 -or $os.FreePhysicalMemory -lt 0) {
            throw 'System memory query returned invalid values.'
        }
        $systemUsed = (($os.TotalVisibleMemorySize - $os.FreePhysicalMemory) * 1KB) / 1GB
        $peakTree = [math]::Max($peakTree, $treeRss)
        $peakSystem = [math]::Max($peakSystem, $systemUsed)
        $hasSample = $true
        [void](Write-MonitorLine "$([DateTime]::Now.ToString('o')) tree_pids=$(@($ids) -join ',') tree_rss_gb=$([math]::Round($treeRss,2)) peak_tree_rss_gb=$([math]::Round($peakTree,2)) system_used_gb=$([math]::Round($systemUsed,2)) peak_system_used_gb=$([math]::Round($peakSystem,2))")

        if ($treeRss -ge $TreeLimitGB -or $systemUsed -ge $SystemLimitGB) {
            [void](Write-MonitorLine "watchdog stopping tree=$([math]::Round($treeRss,2))GB system=$([math]::Round($systemUsed,2))GB time=$([DateTime]::Now.ToString('o'))")
            foreach ($processId in $ids) { Stop-Process -Id $processId -Force }
            break
        }
    } catch {
        [void](Write-MonitorLine "WATCHDOG ERROR; stopping root PID $RootProcessId. $($_.Exception.Message) time=$([DateTime]::Now.ToString('o'))")
        Stop-Process -Id $RootProcessId -Force
        exit 1
    }
    Start-Sleep -Seconds $IntervalSeconds
}
