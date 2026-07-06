$ErrorActionPreference = "Continue"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$targets = Get-CimInstance Win32_Process |
    Where-Object {
        (
            $_.CommandLine -like "*start-ngn6-watchdog.ps1*" -and
            $_.CommandLine -like "*$projectRoot*"
        ) -or (
            $_.CommandLine -like "*ngn6_bot.cli run*" -and
            $_.CommandLine -like "*$projectRoot*"
        )
    }

foreach ($process in $targets) {
    Stop-Process -Id $process.ProcessId -Force
    [pscustomobject]@{
        ProcessId = $process.ProcessId
        Name = $process.Name
        Stopped = $true
    }
}
