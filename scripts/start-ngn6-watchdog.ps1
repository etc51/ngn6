param(
    [string]$Python = "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe",
    [string]$Config = "config/ngn6.yaml",
    [int]$RestartDelaySeconds = 30
)

$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$configPath = Join-Path $projectRoot $Config
$logDir = Join-Path $projectRoot "logs"
$watchdogLog = Join-Path $logDir "ngn6-watchdog.log"
$stdoutLog = Join-Path $logDir "ngn6-run.stdout.log"
$stderrLog = Join-Path $logDir "ngn6-run.stderr.log"

New-Item -ItemType Directory -Force $logDir | Out-Null
Set-Location $projectRoot
$env:PYTHONUNBUFFERED = "1"

function Write-WatchdogLog {
    param([string]$Message)
    $stamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    "$stamp $Message" | Add-Content -Path $watchdogLog -Encoding UTF8
}

Write-WatchdogLog "watchdog_start project=$projectRoot config=$configPath python=$Python"

while ($true) {
    try {
        Write-WatchdogLog "bot_start"
        $process = Start-Process `
            -FilePath $Python `
            -ArgumentList @("-m", "ngn6_bot.cli", "run", "--config", "`"$configPath`"") `
            -WorkingDirectory $projectRoot `
            -NoNewWindow `
            -Wait `
            -PassThru `
            -RedirectStandardOutput $stdoutLog `
            -RedirectStandardError $stderrLog
        $exitCode = $process.ExitCode
        Write-WatchdogLog "bot_exit code=$exitCode"
    }
    catch {
        Write-WatchdogLog "bot_crash error=$($_.Exception.Message)"
    }
    Start-Sleep -Seconds $RestartDelaySeconds
}
