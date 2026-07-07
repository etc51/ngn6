# deployment

## Purpose
Linux service templates for running the bot and dashboard under systemd.

## Contents
- `ngn6-bot.service` - long-running market-data and decision loop.
- `ngn6-dashboard.service` - HTTP dashboard service.

## Rules
- Treat these files as templates; the active VPS uses `/home/codex/ngn6-bot`.
- Keep bot and dashboard as separate services.
- Do not put secrets in unit files. Load secrets from environment files on the server.
- Restart the bot only after tests pass and the deployed `.commit_hash` is updated.

## Quick Checks
- Server status: `systemctl status ngn6-bot ngn6-dashboard`
- Server logs: `journalctl -u ngn6-bot -n 100 --no-pager`
