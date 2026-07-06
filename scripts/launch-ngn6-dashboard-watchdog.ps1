param(
    [string]$Python = "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe",
    [string]$Config = "config/ngn6.yaml",
    [string]$HostName = "0.0.0.0",
    [int]$Port = 8080,
    [int]$RestartDelaySeconds = 30
)

$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$scriptPath = Join-Path $projectRoot "scripts\start-ngn6-dashboard-watchdog.ps1"
$existing = Get-CimInstance Win32_Process |
    Where-Object {
        $_.CommandLine -like "*start-ngn6-dashboard-watchdog.ps1*" -and
        $_.CommandLine -like "*$projectRoot*"
    }

if ($existing) {
    $existing | Select-Object ProcessId, Name, CommandLine
    return
}

$arguments = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", "`"$scriptPath`"",
    "-Python", "`"$Python`"",
    "-Config", "`"$Config`"",
    "-HostName", "`"$HostName`"",
    "-Port", "$Port",
    "-RestartDelaySeconds", "$RestartDelaySeconds"
)

Start-Process powershell.exe -WindowStyle Hidden -ArgumentList $arguments
Start-Sleep -Seconds 2

Get-CimInstance Win32_Process |
    Where-Object {
        $_.CommandLine -like "*start-ngn6-dashboard-watchdog.ps1*" -and
        $_.CommandLine -like "*$projectRoot*"
    } |
    Select-Object ProcessId, Name, CommandLine
