param(
    [string]$ShortcutName = "NGN6 Bot Paper Watchdog.lnk",
    [string]$Python = "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe",
    [string]$Config = "config/ngn6.yaml",
    [int]$RestartDelaySeconds = 30
)

$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$scriptPath = Join-Path $projectRoot "scripts\start-ngn6-watchdog.ps1"
$startupDir = [Environment]::GetFolderPath("Startup")
$shortcutPath = Join-Path $startupDir $ShortcutName

$arguments = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-WindowStyle", "Hidden",
    "-File", "`"$scriptPath`"",
    "-Python", "`"$Python`"",
    "-Config", "`"$Config`"",
    "-RestartDelaySeconds", "$RestartDelaySeconds"
) -join " "

$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = "powershell.exe"
$shortcut.Arguments = $arguments
$shortcut.WorkingDirectory = $projectRoot
$shortcut.WindowStyle = 7
$shortcut.Description = "Starts the NGN6 paper trading watchdog."
$shortcut.Save()

[pscustomobject]@{
    Shortcut = $shortcutPath
    Target = $shortcut.TargetPath
    Arguments = $shortcut.Arguments
}
