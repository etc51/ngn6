# ngn6_bot

## Purpose
Main Python package for the NGN6 intraday paper-trading bot.

## Contents
- `bot.py` - main runtime loop, entry/exit control, safety gates, schedulers.
- `cli.py` - command-line entry points for run, dashboard, smoke, tests, reports, training, and promotion checks.
- `config.py` - config loading and fail-closed runtime validation.
- `tbank.py` - T-Invest gateway and market-data access.
- `models.py` - shared domain dataclasses and enums.
- `signals.py`, `indicators.py`, `orderbook.py`, `tradeflow.py` - signal and market-structure logic.
- `risk.py`, `costs.py`, `execution.py`, `paper.py` - risk, cost, execution, and paper portfolio logic.
- `recorder.py` - runtime JSONL writers for market structure and decisions.
- `dashboard.py`, `review.py`, `charting.py` - local dashboard and chart/report generation.
- `strategy_audit.py` - read-only per-trade paper forensics across decisions, candles, and microstructure.
- `runtime_metadata.py` - commit-hash attribution for runtime reports.
- `learning/` - ML, training, labels, promotion, shadow mode.
- `monitoring/` - drift and monitoring helpers.

## Rules
- Runtime trading must fail closed: no live execution, no fallback entries, no candidate model trading.
- Every runtime JSON/JSONL report should include `commit_hash`.
- Keep generated data out of this package. Runtime outputs belong under project-level `data/`, `logs/`, and `reports/`.
- Prefer config-driven behavior over hard-coded thresholds.
- Add tests when changing entry, exit, risk, model gating, or report formats.

## Quick Checks
- Full tests: `python -m pytest -q`
- Lint: `ruff check .`
- Smoke: `python -m ngn6_bot.cli smoke --config config/ngn6.yaml`
- Strategy audit: `python -m ngn6_bot.cli strategy-audit --config config/ngn6.yaml --fetch-candles`
