from __future__ import annotations

import csv
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from ngn6_bot.config import RuntimeConfig
from ngn6_bot.models import Candle
from ngn6_bot.runtime_metadata import with_commit_hash


@dataclass(frozen=True)
class OracleTrade:
    trading_date: str
    side: str
    label: str
    entry_time: str
    entry_price: float
    stop_price: float
    take_profit: float
    exit_time: str
    exit_price: float
    mfe_pct: float
    mae_pct: float
    score: float
    review_action: str
    reason: str


@dataclass(frozen=True)
class DailyOracleResult:
    figi: str
    trading_date: str
    candles_1m: int
    candles_15m: int
    oracle_trades: list[OracleTrade]
    sideways_labels: int
    json_path: Path
    labels_csv_path: Path


class DailyOracleScheduler:
    def __init__(self, config: RuntimeConfig, logger, reload_feedback_model=None):
        self.config = config
        self.logger = logger
        self.reload_feedback_model = reload_feedback_model
        self.enabled = bool(config.get("daily_oracle", "enabled", default=True))
        self.run_time = _parse_local_time(str(config.get("daily_oracle", "run_time", default="23:50")))
        self.minutes = int(config.get("daily_oracle", "minutes", default=12_000))
        self.training_minutes = int(config.get("daily_oracle", "training_minutes", default=90_000))
        self.output_dir = Path(config.get("daily_oracle", "output_dir", default="reports/daily_oracle"))
        self.state_file = Path(config.get("daily_oracle", "state_file", default="data/daily_oracle_runs.json"))
        self.retry_delay = timedelta(
            minutes=float(config.get("daily_oracle", "retry_delay_minutes", default=30))
        )
        self.generated = self._load_generated()
        self.failed_at: dict[str, datetime] = {}

    def maybe_run(self, now: datetime) -> None:
        if not self.enabled:
            return
        local_now = _utc(now).astimezone(ZoneInfo(self.config.timezone))
        trading_date = _candidate_training_date(local_now, self.run_time)
        if trading_date is None:
            return
        run_key = trading_date.isoformat()
        if run_key in self.generated:
            return
        last_failure = self.failed_at.get(run_key)
        if last_failure is not None and _utc(now) - last_failure < self.retry_delay:
            return
        try:
            result = generate_daily_oracle_from_api(
                self.config,
                self.logger,
                trading_date=trading_date,
                minutes=self.minutes,
                output_dir=self.output_dir,
            )
            training = _train_feedback_from_api(
                self.config,
                self.logger,
                minutes=self.training_minutes,
            )
            if self.reload_feedback_model is not None:
                self.reload_feedback_model()
            self.logger.info(
                "daily_oracle_retrained",
                extra={
                    "event": "daily_oracle_retrained",
                    "details": {
                        "date": run_key,
                        "oracle_trades": len(result.oracle_trades),
                        "sideways_labels": result.sideways_labels,
                        "labels_csv": str(result.labels_csv_path),
                        "model_path": str(training.report.path),
                        "examples": training.total_examples,
                        "classes": training.report.classes,
                        "promotion_score": getattr(training.report, "promotion_score", None),
                        "promoted": getattr(training.report, "promoted", None),
                    },
                },
            )
            self.generated.add(run_key)
            self._save_generated()
        except Exception as exc:
            self.failed_at[run_key] = _utc(now)
            self.logger.exception(
                "daily_oracle_failed",
                extra={"event": "daily_oracle_failed", "details": {"date": run_key, "error": str(exc)}},
            )

    def _load_generated(self) -> set[str]:
        try:
            payload = json.loads(self.state_file.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return set()
        return set(payload.get("generated", []))

    def _save_generated(self) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(
            json.dumps(
                with_commit_hash({"generated": sorted(self.generated)}),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


def generate_daily_oracle_from_api(
    config: RuntimeConfig,
    logger,
    *,
    trading_date: date,
    minutes: int = 12_000,
    output_dir: str | Path = "reports/daily_oracle",
) -> DailyOracleResult:
    from ngn6_bot.backtest import fetch_1m_history

    figi, candles_1m = fetch_1m_history(config, logger, minutes)
    return generate_daily_oracle(
        config,
        figi=figi,
        candles_1m=candles_1m,
        trading_date=trading_date,
        output_dir=output_dir,
    )


def _train_feedback_from_api(config: RuntimeConfig, logger, *, minutes: int):
    from ngn6_bot.learning.training import train_feedback_from_api

    return train_feedback_from_api(config, logger, minutes=minutes)


def generate_daily_oracle(
    config: RuntimeConfig,
    *,
    figi: str,
    candles_1m: list[Candle],
    trading_date: date,
    output_dir: str | Path = "reports/daily_oracle",
) -> DailyOracleResult:
    tz = ZoneInfo(config.timezone)
    day_1m = [
        candle
        for candle in candles_1m
        if _utc(candle.timestamp).astimezone(tz).date() == trading_date
    ]
    candles_15m = _aggregate_candles(day_1m, "15min")
    decisions = _read_decisions_for_date(
        Path(config.get("data_collection", "decisions_file", default="data/decisions.jsonl")),
        trading_date,
        tz,
    )
    oracle_trades = _select_oracle_trades(config, candles_15m, trading_date, tz, decisions)
    sideways = _sideways_intervals(config, candles_15m, oracle_trades, trading_date, tz)

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / f"{config.get('instrument', 'ticker')}_{trading_date.isoformat()}_oracle.json"
    labels_csv_path = output_root / f"{config.get('instrument', 'ticker')}_{trading_date.isoformat()}_oracle_labels.csv"

    payload = {
        "schema_version": 1,
        "figi": figi,
        "trading_date": trading_date.isoformat(),
        "candles_1m": len(day_1m),
        "candles_15m": len(candles_15m),
        "oracle_trades": [asdict(item) for item in oracle_trades],
        "sideways_labels": sideways,
        "note": "Oracle labels use future candles only for post-market review. Training features are rebuilt at label time only.",
    }
    payload = with_commit_hash(payload)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_labels_csv(labels_csv_path, oracle_trades, sideways, candles_15m, config, trading_date, tz)
    latest_path = output_root / "latest_oracle_labels.csv"
    shutil.copyfile(labels_csv_path, latest_path)
    return DailyOracleResult(
        figi=figi,
        trading_date=trading_date.isoformat(),
        candles_1m=len(day_1m),
        candles_15m=len(candles_15m),
        oracle_trades=oracle_trades,
        sideways_labels=sideways,
        json_path=json_path,
        labels_csv_path=labels_csv_path,
    )


def _aggregate_candles(candles: list[Candle], timeframe: str) -> list[Candle]:
    minutes = {"5min": 5, "15min": 15}[timeframe]
    aggregated: dict[datetime, Candle] = {}
    for candle in candles:
        timestamp = _utc(candle.timestamp)
        bucket_minute = timestamp.minute - (timestamp.minute % minutes)
        bucket_start = timestamp.replace(minute=bucket_minute, second=0, microsecond=0)
        existing = aggregated.get(bucket_start)
        if existing is None:
            aggregated[bucket_start] = Candle(
                timestamp=bucket_start,
                open=candle.open,
                high=candle.high,
                low=candle.low,
                close=candle.close,
                volume=candle.volume,
                timeframe=timeframe,
            )
            continue
        aggregated[bucket_start] = Candle(
            timestamp=bucket_start,
            open=existing.open,
            high=max(existing.high, candle.high),
            low=min(existing.low, candle.low),
            close=candle.close,
            volume=existing.volume + candle.volume,
            timeframe=timeframe,
        )
    return [aggregated[key] for key in sorted(aggregated)]


def _select_oracle_trades(
    config: RuntimeConfig,
    candles: list[Candle],
    trading_date: date,
    tz: ZoneInfo,
    decisions: list[dict[str, Any]],
) -> list[OracleTrade]:
    horizon = int(config.get("daily_oracle", "horizon_bars", default=8))
    min_mfe_pct = float(config.get("daily_oracle", "min_mfe_pct", default=0.60))
    max_mae_pct = float(config.get("daily_oracle", "max_mae_pct", default=0.35))
    max_trades = int(config.get("daily_oracle", "max_trades_per_day", default=6))
    min_spacing = int(config.get("daily_oracle", "min_spacing_bars", default=3))
    stop_buffer_pct = float(config.get("daily_oracle", "stop_buffer_pct", default=0.08))
    reward_pct = float(config.get("daily_oracle", "take_profit_pct", default=0.80))
    candidates: list[tuple[int, OracleTrade]] = []

    for index in range(0, max(0, len(candles) - horizon)):
        entry = candles[index]
        future = candles[index + 1 : index + horizon + 1]
        if len(future) < horizon:
            continue
        entry_price = float(entry.close)
        high = max(float(candle.high) for candle in future)
        low = min(float(candle.low) for candle in future)
        long_mfe = (high - entry_price) / entry_price * 100
        long_mae = (entry_price - low) / entry_price * 100
        short_mfe = (entry_price - low) / entry_price * 100
        short_mae = (high - entry_price) / entry_price * 100
        long_score = long_mfe - long_mae * 1.5
        short_score = short_mfe - short_mae * 1.5
        if long_mfe >= min_mfe_pct and long_mae <= max_mae_pct and long_score > short_score:
            candidates.append(
                (
                    index,
                    _build_trade(
                        candles,
                        index,
                        future,
                        trading_date,
                        tz,
                        decisions,
                        side="long",
                        label="LONG_CONTINUATION",
                        mfe_pct=long_mfe,
                        mae_pct=long_mae,
                        score=long_score,
                        stop_price=entry_price * (1 - (long_mae + stop_buffer_pct) / 100),
                        take_profit=entry_price * (1 + reward_pct / 100),
                    ),
                )
            )
        if short_mfe >= min_mfe_pct and short_mae <= max_mae_pct and short_score > long_score:
            candidates.append(
                (
                    index,
                    _build_trade(
                        candles,
                        index,
                        future,
                        trading_date,
                        tz,
                        decisions,
                        side="short",
                        label="FAST_SHORT",
                        mfe_pct=short_mfe,
                        mae_pct=short_mae,
                        score=short_score,
                        stop_price=entry_price * (1 + (short_mae + stop_buffer_pct) / 100),
                        take_profit=entry_price * (1 - reward_pct / 100),
                    ),
                )
            )

    selected: list[tuple[int, OracleTrade]] = []
    for index, trade in sorted(candidates, key=lambda item: item[1].score, reverse=True):
        if any(abs(index - selected_index) < min_spacing for selected_index, _ in selected):
            continue
        selected.append((index, trade))
        if len(selected) >= max_trades:
            break
    return [trade for _, trade in sorted(selected, key=lambda item: item[0])]


def _build_trade(
    candles: list[Candle],
    index: int,
    future: list[Candle],
    trading_date: date,
    tz: ZoneInfo,
    decisions: list[dict[str, Any]],
    *,
    side: str,
    label: str,
    mfe_pct: float,
    mae_pct: float,
    score: float,
    stop_price: float,
    take_profit: float,
) -> OracleTrade:
    entry = candles[index]
    exit_candle = _first_exit_candle(future, side, stop_price, take_profit) or future[-1]
    exit_price = _exit_price(side, exit_candle, stop_price, take_profit)
    review_action, reason = _review_action(decisions, _utc(entry.timestamp), side)
    return OracleTrade(
        trading_date=trading_date.isoformat(),
        side=side,
        label=label,
        entry_time=_utc(entry.timestamp).astimezone(tz).isoformat(),
        entry_price=round(float(entry.close), 6),
        stop_price=round(float(stop_price), 6),
        take_profit=round(float(take_profit), 6),
        exit_time=_utc(exit_candle.timestamp).astimezone(tz).isoformat(),
        exit_price=round(float(exit_price), 6),
        mfe_pct=round(float(mfe_pct), 4),
        mae_pct=round(float(mae_pct), 4),
        score=round(float(score), 4),
        review_action=review_action,
        reason=reason,
    )


def _first_exit_candle(
    future: list[Candle],
    side: str,
    stop_price: float,
    take_profit: float,
) -> Candle | None:
    for candle in future:
        if side == "long" and (float(candle.low) <= stop_price or float(candle.high) >= take_profit):
            return candle
        if side == "short" and (float(candle.high) >= stop_price or float(candle.low) <= take_profit):
            return candle
    return None


def _exit_price(side: str, candle: Candle, stop_price: float, take_profit: float) -> float:
    if side == "long":
        if float(candle.low) <= stop_price:
            return stop_price
        if float(candle.high) >= take_profit:
            return take_profit
    else:
        if float(candle.high) >= stop_price:
            return stop_price
        if float(candle.low) <= take_profit:
            return take_profit
    return float(candle.close)


def _review_action(
    decisions: list[dict[str, Any]],
    entry_time: datetime,
    side: str,
) -> tuple[str, str]:
    window = timedelta(minutes=20)
    nearby = [
        item
        for item in decisions
        if abs((_parse_timestamp(item.get("timestamp")) - entry_time).total_seconds()) <= window.total_seconds()
    ]
    opens = [item for item in nearby if str(item.get("action")) == "open_accepted"]
    if any(str(item.get("side")).lower() == side for item in opens):
        return "matched_bot_entry", "bot_opened_same_side_near_oracle_entry"
    if opens:
        return "opposite_or_wrong_entry", "bot_opened_different_side_near_oracle_entry"
    return "missed_valid_entry", "no_bot_entry_near_oracle_entry"


def _sideways_intervals(
    config: RuntimeConfig,
    candles: list[Candle],
    oracle_trades: list[OracleTrade],
    trading_date: date,
    tz: ZoneInfo,
) -> int:
    del trading_date, tz
    horizon = int(config.get("daily_oracle", "sideways_horizon_bars", default=4))
    max_mfe_pct = float(config.get("daily_oracle", "sideways_max_mfe_pct", default=0.25))
    oracle_times = {_parse_timestamp(trade.entry_time) for trade in oracle_trades}
    count = 0
    for index in range(0, max(0, len(candles) - horizon), max(1, horizon)):
        entry = candles[index]
        entry_time = _utc(entry.timestamp)
        if any(abs((entry_time - item).total_seconds()) < 45 * 60 for item in oracle_times):
            continue
        future = candles[index + 1 : index + horizon + 1]
        if not future:
            continue
        price = float(entry.close)
        up = (max(float(candle.high) for candle in future) - price) / price * 100
        down = (price - min(float(candle.low) for candle in future)) / price * 100
        if max(up, down) <= max_mfe_pct:
            count += 1
    return count


def _write_labels_csv(
    path: Path,
    trades: list[OracleTrade],
    sideways_count: int,
    candles_15m: list[Candle],
    config: RuntimeConfig,
    trading_date: date,
    tz: ZoneInfo,
) -> None:
    fieldnames = [
        "date",
        "start_time",
        "end_time",
        "label",
        "timeframe",
        "source",
        "entry_time",
        "entry_price",
        "stop_price",
        "take_profit",
        "exit_time",
        "exit_price",
        "mfe_pct",
        "mae_pct",
        "score",
        "review_action",
        "reason",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for trade in trades:
            start = _parse_timestamp(trade.entry_time).astimezone(tz)
            end = min(_parse_timestamp(trade.exit_time).astimezone(tz), start + timedelta(minutes=45))
            writer.writerow(
                {
                    "date": trading_date.isoformat(),
                    "start_time": start.time().replace(microsecond=0).isoformat(),
                    "end_time": end.time().replace(microsecond=0).isoformat(),
                    "label": trade.label,
                    "timeframe": "15min",
                    "source": "daily_oracle",
                    "entry_time": trade.entry_time,
                    "entry_price": trade.entry_price,
                    "stop_price": trade.stop_price,
                    "take_profit": trade.take_profit,
                    "exit_time": trade.exit_time,
                    "exit_price": trade.exit_price,
                    "mfe_pct": trade.mfe_pct,
                    "mae_pct": trade.mae_pct,
                    "score": trade.score,
                    "review_action": trade.review_action,
                    "reason": trade.reason,
                }
            )
        _write_sideways_rows(writer, sideways_count, candles_15m, config, trading_date, tz)


def _write_sideways_rows(writer, count: int, candles: list[Candle], config: RuntimeConfig, trading_date: date, tz: ZoneInfo) -> None:
    if count <= 0:
        return
    horizon = int(config.get("daily_oracle", "sideways_horizon_bars", default=4))
    written = 0
    for index in range(0, max(0, len(candles) - horizon), max(1, horizon)):
        start = _utc(candles[index].timestamp).astimezone(tz)
        end = start + timedelta(minutes=15 * horizon)
        writer.writerow(
            {
                "date": trading_date.isoformat(),
                "start_time": start.time().replace(microsecond=0).isoformat(),
                "end_time": end.time().replace(microsecond=0).isoformat(),
                "label": "SIDEWAYS",
                "timeframe": "15min",
                "source": "daily_oracle",
                "entry_time": "",
                "entry_price": "",
                "stop_price": "",
                "take_profit": "",
                "exit_time": "",
                "exit_price": "",
                "mfe_pct": "",
                "mae_pct": "",
                "score": "",
                "review_action": "no_trade_zone",
                "reason": "low_future_range_post_market_label",
            }
        )
        written += 1
        if written >= count:
            return


def _read_decisions_for_date(path: Path, trading_date: date, tz: ZoneInfo) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            timestamp = _parse_timestamp(item.get("timestamp"))
            if timestamp.astimezone(tz).date() != trading_date:
                continue
            rows.append(item)
    return rows


def _parse_local_time(value: str) -> time:
    return time.fromisoformat(value.strip())


def _candidate_training_date(local_now: datetime, run_time: time) -> date | None:
    local_date = local_now.date()
    current_time = local_now.time().replace(second=0, microsecond=0)
    if current_time >= run_time:
        return _latest_weekday(local_date)
    return _latest_weekday(local_date - timedelta(days=1))


def _latest_weekday(value: date) -> date:
    current = value
    while current.weekday() >= 5:
        current -= timedelta(days=1)
    return current


def _parse_timestamp(value: Any) -> datetime:
    parsed = datetime.fromisoformat(str(value))
    return _utc(parsed)


def _utc(timestamp: datetime) -> datetime:
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)
