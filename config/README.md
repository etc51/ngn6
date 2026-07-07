# config

## Purpose
Runtime configuration for the NGN6 bot. The main file is `ngn6.yaml`.

## Contents
- `ngn6.yaml` - instrument, paper mode, ML gates, risk limits, execution costs, review schedule, dashboard port, daily oracle schedule, and data paths.

## Rules
- Keep `bot.dry_run: true` and `trading.live_enabled: false` unless there is an explicit audited live-trading change.
- Do not store real tokens or account secrets here.
- Keep data/report paths relative to the project root so the same config works on the VPS.
- Tighten safety gates in config first; loosen them only with tests and a commit explaining why.

## Quick Checks
- Validate config: `python -m ngn6_bot.cli smoke --config config/ngn6.yaml`
- Confirm promotion status: `python -m ngn6_bot.cli promotion-check --config config/ngn6.yaml --model active`
