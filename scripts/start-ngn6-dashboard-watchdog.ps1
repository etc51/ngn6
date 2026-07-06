param(
    [string]$Python = "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe",
    [string]$Config = "config/ngn6.yaml",
    [string]$HostName = "0.0.0.0",
    [int]$Port = 8080,
    [int]$RestartDelaySeconds = 30
)

$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$configPath = Join-Path $projectRoot $Config
$logDir = Join-Path $projectRoot "logs"
$watchdogLog = Join-Path $logDir "ngn6-dashboard-watchdog.log"
$stdoutLog = Join-Path $logDir "ngn6-dashboard.stdout.log"
$stderrLog = Join-Path $logDir "ngn6-dashboard.stderr.log"

New-Item -ItemType Directory -Force $logDir | Out-Null
Set-Location $projectRoot
$env:PYTHONUNBUFFERED = "1"

function Write-WatchdogLog {
    param([string]$Message)
    $stamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    "$stamp $Message" | Add-Content -Path $watchdogLog -Encoding UTF8
}

Write-WatchdogLog "dashboard_watchdog_start project=$projectRoot config=$configPath host=$HostName port=$Port python=$Python"

while ($true) {
    try {
        Write-WatchdogLog "dashboard_start"
        $process = Start-Process `
            -FilePath $Python `
            -ArgumentList @("-m", "ngn6_bot.cli", "dashboard", "--config", "`"$configPath`"", "--host", $HostName, "--port", "$Port") `
            -WorkingDirectory $projectRoot `
            -NoNewWindow `
            -Wait `
            -PassThru `
            -RedirectStandardOutput $stdoutLog `
            -RedirectStandardError $stderrLog
        $exitCode = $process.ExitCode
        Write-WatchdogLog "dashboard_exit code=$exitCode"
    }
    catch {
        Write-WatchdogLog "dashboard_crash error=$($_.Exception.Message)"
    }
    Start-Sleep -Seconds $RestartDelaySeconds
}
