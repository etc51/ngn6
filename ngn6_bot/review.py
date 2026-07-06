from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from ngn6_bot.charting import fetch_day_candles, plot_indicator_chart
from ngn6_bot.config import RuntimeConfig
from ngn6_bot.models import Candle, MarketState
from ngn6_bot.recorder import read_jsonl_tail


class ReviewScheduler:
    def __init__(self, config: RuntimeConfig, logger):
        self.config = config
        self.logger = logger
        self.enabled = bool(config.get("review", "enabled", default=True))
        self.times = set(config.get("review", "times", default=["12:00", "19:00"]))
        self.timeframes = list(config.get("review", "timeframes", default=["1min", "15min"]))
        self.output_dir = Path(config.get("review", "output_dir", default="reports/review"))
        self.state_file = Path(config.get("review", "state_file", default="data/review_runs.json"))
        self.generated = self._load_generated()

    def maybe_run(self, now: datetime, state: MarketState) -> None:
        if not self.enabled:
            return
        local_now = now.astimezone(ZoneInfo(self.config.timezone))
        schedule_key = local_now.strftime("%H:%M")
        if schedule_key not in self.times:
            return
        run_key = f"{local_now.date().isoformat()}_{schedule_key}"
        if run_key in self.generated:
            return

        paths = []
        try:
            paths = self.generate_from_state(state, local_now.date(), schedule_key.replace(":", ""))
            self.logger.info(
                "review_charts_generated",
                extra={
                    "event": "review_charts_generated",
                    "details": {"run_key": run_key, "paths": [str(path) for path in paths]},
                },
            )
        except Exception as exc:
            self.logger.exception(
                "review_charts_failed",
                extra={"event": "review_charts_failed", "details": {"run_key": run_key, "error": str(exc)}},
            )
        finally:
            self.generated.add(run_key)
            self._save_generated()

    def generate_from_state(self, state: MarketState, trading_date: date, label: str) -> list[Path]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        decisions = _read_recent_decisions(self.config)
        paths: list[Path] = []
        for timeframe in self.timeframes:
            candles = _candles_for_date(_state_candles(state, timeframe), trading_date, self.config.timezone)
            if len(candles) < 5:
                self.logger.warning(
                    "review_chart_skipped",
                    extra={
                        "event": "review_chart_skipped",
                        "details": {"date": trading_date.isoformat(), "timeframe": timeframe, "candles": len(candles)},
                    },
                )
                continue
            output = self.output_dir / (
                f"{self.config.get('instrument', 'ticker')}_{trading_date.isoformat()}_{label}_{timeframe}_review.png"
            )
            paths.append(
                plot_indicator_chart(
                    self.config,
                    candles,
                    trading_date,
                    timeframe,
                    output,
                    decisions=decisions,
                    title_suffix=f"review {label}",
                )
            )
        return paths

    def _load_generated(self) -> set[str]:
        try:
            payload = json.loads(self.state_file.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return set()
        return set(payload.get("generated", []))

    def _save_generated(self) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {"generated": sorted(self.generated)}
        self.state_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def generate_review_from_api(
    config: RuntimeConfig,
    logger,
    trading_date: date | None = None,
    label: str | None = None,
    timeframes: list[str] | None = None,
) -> list[Path]:
    tz = ZoneInfo(config.timezone)
    target_date = trading_date or datetime.now(tz).date()
    run_label = label or datetime.now(tz).strftime("%H%M")
    output_dir = Path(config.get("review", "output_dir", default="reports/review"))
    output_dir.mkdir(parents=True, exist_ok=True)
    decisions = _read_recent_decisions(config)
    paths: list[Path] = []
    for timeframe in timeframes or list(config.get("review", "timeframes", default=["1min", "15min"])):
        _, candles = fetch_day_candles(config, logger, target_date, timeframe)
        output = output_dir / (
            f"{config.get('instrument', 'ticker')}_{target_date.isoformat()}_{run_label}_{timeframe}_review.png"
        )
        paths.append(
            plot_indicator_chart(
                config,
                candles,
                target_date,
                timeframe,
                output,
                decisions=decisions,
                title_suffix=f"review {run_label}",
            )
        )
    return paths


def _read_recent_decisions(config: RuntimeConfig) -> list[dict]:
    path = Path(config.get("data_collection", "decisions_file", default="data/decisions.jsonl"))
    return read_jsonl_tail(path, 1000)


def _state_candles(state: MarketState, timeframe: str) -> list[Candle]:
    return {
        "1min": list(state.candles_1m),
        "5min": list(state.candles_5m),
        "15min": list(state.candles_15m),
    }[timeframe]


def _candles_for_date(candles: list[Candle], trading_date: date, timezone_name: str) -> list[Candle]:
    tz = ZoneInfo(timezone_name)
    return [candle for candle in candles if candle.timestamp.astimezone(tz).date() == trading_date]
