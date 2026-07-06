param(
    [string]$TaskName = "NGN6 Bot Paper Watchdog",
    [string]$Python = "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe",
    [string]$Config = "config/ngn6.yaml",
    [int]$RestartDelaySeconds = 30
)

$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$scriptPath = Join-Path $projectRoot "scripts\start-ngn6-watchdog.ps1"
$arguments = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-WindowStyle", "Hidden",
    "-File", "`"$scriptPath`"",
    "-Python", "`"$Python`"",
    "-Config", "`"$Config`"",
    "-RestartDelaySeconds", "$RestartDelaySeconds"
) -join " "

$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $arguments -WorkingDirectory $projectRoot
$trigger = New-ScheduledTaskTrigger -AtLogOn
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Description "Starts the NGN6 paper trading watchdog at Windows logon." `
    -Force | Out-Null

Get-ScheduledTask -TaskName $TaskName | Select-Object TaskName, State
