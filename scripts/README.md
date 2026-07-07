# scripts

## Purpose
Operational scripts for Windows watchdogs, dashboard launchers, startup registration, and history downloads.

## Contents
- `start-ngn6-watchdog.ps1` / `stop-ngn6-watchdog.ps1` - local Windows bot watchdog control.
- `start-ngn6-dashboard-watchdog.ps1` / `stop-ngn6-dashboard-watchdog.ps1` - local Windows dashboard watchdog control.
- `launch-*.ps1` - hidden-window launch wrappers.
- `register-ngn6-watchdog-*.ps1` - Windows startup/task registration helpers.
- `download_tbank_neoassets_history.py` - historical market-data downloader.

## Rules
- Current production-like collection should run on VPS, not local Windows.
- Do not re-enable local startup without an explicit request.
- Scripts must avoid hard-coded secrets.
- Prefer hidden background processes for watchdogs unless interactive UI is required.

## Quick Checks
- Local NGN6 processes: inspect `ngn6_bot.cli run`, `ngn6_bot.cli dashboard`, and watchdog command lines.
- VPS services are managed through systemd, not these PowerShell scripts.
