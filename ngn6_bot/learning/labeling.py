from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from ngn6_bot.backtest import (
    _blocked_by_reentry_cooldown,
    _close_trade,
    _execution_cost_config,
    _exit_price,
    _exit_reason,
    fetch_1m_history,
)
from ngn6_bot.charting import plot_indicator_chart
from ngn6_bot.config import RuntimeConfig
from ngn6_bot.costs import trade_covers_costs
from ngn6_bot.indicators import add_indicators, candles_to_frame
from ngn6_bot.microstructure_replay import MicrostructureReplay, neutral_orderbook, neutral_trade_flow
from ngn6_bot.models import Candle, MarketState, Position, Side
from ngn6_bot.risk import (
    calculate_position_lots,
    must_flatten_before_clearing,
    stop_with_buffer,
    update_trailing_stop,
)
from ngn6_bot.signals import generate_signal


@dataclass(frozen=True)
class LabelingChartResult:
    figi: str
    paths: list[Path]
    decisions: list[dict[str, Any]]
    regimes: list[dict[str, Any]]
    backtest_trades: int = 0


def generate_labeling_charts(
    config: RuntimeConfig,
    logger,
    *,
    days: int = 5,
    minutes: int = 12_000,
    timeframe: str = "15min",
    output_dir: str | Path = "reports/labeling",
    backtest_report: str | Path | None = None,
) -> LabelingChartResult:
    figi, candles_1m = fetch_1m_history(config, logger, minutes)
    tz = ZoneInfo(config.timezone)
    target_dates = _latest_trading_dates(candles_1m, tz, days)
    decisions = _replay_strategy_decisions(config, candles_1m, target_dates)
    candles_by_date = _aggregate_for_dates(candles_1m, target_dates, timeframe, config.timezone)
    backtest_trades = _load_backtest_report_trades(backtest_report)
    backtest_trades_by_date = _group_backtest_trades_by_date(backtest_trades, tz)
    regimes = []
    paths: list[Path] = []
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    title_suffix = "feedback labeling"
    if backtest_report is not None:
        title_suffix = "feedback labeling + backtest trades"

    for trading_date in target_dates:
        day_candles = candles_by_date.get(trading_date, [])
        if len(day_candles) < 5:
            continue
        day_regimes = _sideways_regimes(config, day_candles, trading_date)
        day_backtest_trades = (
            backtest_trades_by_date.get(trading_date, []) if backtest_report is not None else None
        )
        regimes.extend(day_regimes)
        output = output_root / (
            f"{config.get('instrument', 'ticker')}_{trading_date.isoformat()}_{timeframe}_feedback.png"
        )
        paths.append(
            plot_indicator_chart(
                config,
                day_candles,
                trading_date,
                timeframe,
                output,
                decisions=decisions,
                regimes=day_regimes,
                backtest_trades=day_backtest_trades,
                title_suffix=title_suffix,
            )
        )

    return LabelingChartResult(
        figi=figi,
        paths=paths,
        decisions=decisions,
        regimes=regimes,
        backtest_trades=len(backtest_trades),
    )


def _load_backtest_report_trades(path: str | Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    trades = payload.get("trades", [])
    if not isinstance(trades, list):
        raise ValueError(f"Backtest report {path} has no trades list.")
    result: list[dict[str, Any]] = []
    for number, trade in enumerate(trades, start=1):
        if not isinstance(trade, dict):
            continue
        item = dict(trade)
        item.setdefault("number", number)
        result.append(item)
    return result


def _group_backtest_trades_by_date(
    trades: list[dict[str, Any]],
    tz: ZoneInfo,
) -> dict[date, list[dict[str, Any]]]:
    grouped: dict[date, list[dict[str, Any]]] = defaultdict(list)
    for trade in trades:
        seen_dates: set[date] = set()
        for key in ("entry_time", "exit_time"):
            timestamp = _parse_timestamp(trade.get(key))
            if timestamp is None:
                continue
            local_date = timestamp.astimezone(tz).date()
            if local_date not in seen_dates:
                grouped[local_date].append(trade)
                seen_dates.add(local_date)
    return dict(grouped)


def _parse_timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _replay_strategy_decisions(
    config: RuntimeConfig,
    candles: list[Candle],
    target_dates: list[date],
) -> list[dict[str, Any]]:
    from ngn6_bot.bot import TradingBot

    state = MarketState()
    bot = TradingBot(config, logger=_NullLogger(), runtime_services=False)
    signal_config = bot._signal_config()
    risk_config = bot._risk_config()
    cost_config = _execution_cost_config(config)
    target_set = set(target_dates)
    decisions: list[dict[str, Any]] = []
    active_position = Position()
    take_profit_cooldown_minutes = float(
        config.get("signals", "reentry_cooldown_after_take_profit_minutes", default=0)
    )
    exit_cooldown_minutes = float(
        config.get("signals", "reentry_cooldown_after_exit_minutes", default=0)
    )
    loss_cooldown_minutes = float(
        config.get("signals", "reentry_cooldown_after_loss_minutes", default=0)
    )
    last_full_exit: tuple[Side, datetime, str, float] | None = None
    microstructure = MicrostructureReplay.from_config(config)

    for candle in candles:
        state.update_candle(candle)
        bot.state = state
        now = candle.timestamp if candle.timestamp.tzinfo else candle.timestamp.replace(tzinfo=timezone.utc)
        local_date = now.astimezone(ZoneInfo(config.timezone)).date()

        if active_position.is_open:
            exit_reason = _exit_reason(active_position, candle, risk_config, now)
            if exit_reason:
                exit_price = _exit_price(active_position, candle, exit_reason)
                if local_date in target_set:
                    decisions.append(
                        {
                            "timestamp": now.isoformat(),
                            "action": "close_accepted",
                            "reason": exit_reason,
                            "price": exit_price,
                            "side": active_position.side.value,
                        }
                    )
                trade = _close_trade(active_position, candle, exit_price, exit_reason, cost_config)
                last_full_exit = (active_position.side, now, exit_reason, trade.pnl_pct)
                active_position = Position()
                state.position = active_position
            else:
                update_trailing_stop(active_position, candle.close, risk_config)
            continue

        if local_date not in target_set:
            continue
        if len(state.candles_1m) < 40:
            continue
        if must_flatten_before_clearing(now, risk_config):
            continue

        execution_df = bot._indicator_frame("1min")
        confirmation_df = bot._indicator_frame("5min")
        context_df = bot._indicator_frame("15min")
        micro_snapshot = microstructure.at(now)
        if micro_snapshot is None:
            orderbook = neutral_orderbook(candle.close)
            trade_flow = neutral_trade_flow()
        else:
            orderbook = micro_snapshot.orderbook
            trade_flow = micro_snapshot.trade_flow
        signal = generate_signal(
            execution_df=execution_df,
            confirmation_df=confirmation_df,
            context_df=context_df,
            orderbook=orderbook,
            trade_flow=trade_flow,
            config=signal_config,
            now=now,
        )
        signal = bot._apply_feedback(
            signal,
            execution_df=execution_df,
            confirmation_df=confirmation_df,
            context_df=context_df,
            orderbook_features=orderbook,
            trade_flow=trade_flow,
            now=now,
        )
        if signal.side == Side.FLAT:
            continue
        if _blocked_by_reentry_cooldown(
            signal.side,
            now,
            last_full_exit,
            take_profit_cooldown_minutes,
            exit_cooldown_minutes=exit_cooldown_minutes,
            loss_cooldown_minutes=loss_cooldown_minutes,
        ):
            continue

        stop_price = stop_with_buffer(signal, risk_config)
        if stop_price is None:
            continue
        cost_check = trade_covers_costs(
            price=signal.price,
            expected_move_pct=bot._expected_move_pct(signal, risk_config),
            spread_bps=orderbook.spread_bps or 0.0,
            config=cost_config,
        )
        if not cost_check.accepted:
            continue
        normalized_signal = replace(signal, stop_price=stop_price)
        lots = calculate_position_lots(normalized_signal, risk_config)
        if lots <= 0:
            continue

        decisions.append(
            {
                "timestamp": now.isoformat(),
                "action": "open_accepted",
                "reason": signal.reason,
                "price": candle.close,
                "side": signal.side.value,
                "confidence": signal.confidence,
                "stop_price": stop_price,
                "take_profit1": signal.take_profit1,
                "take_profit2": signal.take_profit2,
            }
        )
        active_position = Position(
            side=signal.side,
            lots=lots,
            avg_price=candle.close,
            opened_at=now,
            stop_price=stop_price,
            trailing_stop=stop_price,
            take_profit1=signal.take_profit1,
            take_profit2=signal.take_profit2,
        )
        state.position = active_position

    return decisions


def _aggregate_for_dates(
    candles: list[Candle],
    target_dates: list[date],
    timeframe: str,
    timezone_name: str,
) -> dict[date, list[Candle]]:
    tz = ZoneInfo(timezone_name)
    target_set = set(target_dates)
    source = candles if timeframe == "1min" else _aggregate_candles(candles, timeframe)
    grouped: dict[date, list[Candle]] = defaultdict(list)
    for candle in source:
        local_date = candle.timestamp.astimezone(tz).date()
        if local_date in target_set:
            grouped[local_date].append(candle)
    return dict(grouped)


def _aggregate_candles(candles: list[Candle], timeframe: str) -> list[Candle]:
    minutes = {"5min": 5, "15min": 15}[timeframe]
    aggregated: dict[datetime, Candle] = {}
    for candle in candles:
        bucket_minute = candle.timestamp.minute - (candle.timestamp.minute % minutes)
        bucket_start = candle.timestamp.replace(minute=bucket_minute, second=0, microsecond=0)
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


def _sideways_regimes(
    config: RuntimeConfig,
    candles: list[Candle],
    trading_date: date,
) -> list[dict[str, Any]]:
    df = add_indicators(
        candles_to_frame(candles),
        ema_fast=int(config.get("indicators", "ema_fast")),
        ema_slow=int(config.get("indicators", "ema_slow")),
        rsi_period=int(config.get("indicators", "rsi_period")),
        bollinger_period=int(config.get("indicators", "bollinger_period")),
        bollinger_std=float(config.get("indicators", "bollinger_std")),
        volume_ma_period=int(config.get("indicators", "volume_ma_period")),
    )
    if df.empty:
        return []

    regimes: list[dict[str, Any]] = []
    current_start = None
    last_time = None
    for timestamp, row in df.iterrows():
        close = float(row["close"])
        if close <= 0:
            continue
        ema_spread_pct = abs(float(row["ema_fast"]) - float(row["ema_slow"])) / close * 100
        bb_width = float(row.get("bb_width_pct", 0.0) or 0.0)
        recent = df.loc[:timestamp].tail(8)
        range_pct = (
            (float(recent["high"].max()) - float(recent["low"].min())) / close * 100
            if len(recent) >= 4
            else 999.0
        )
        rsi = float(row.get("rsi", 50.0) or 50.0)
        sideways = ema_spread_pct <= 0.14 and range_pct <= 1.2 and 38 <= rsi <= 62
        sideways = sideways or (bb_width and bb_width <= 1.35 and ema_spread_pct <= 0.18)
        if sideways and current_start is None:
            current_start = timestamp
        if not sideways and current_start is not None:
            if last_time is not None and _minutes_between(current_start, last_time) >= 30:
                regimes.append(_regime_payload(current_start, last_time, trading_date))
            current_start = None
        last_time = timestamp
    if current_start is not None and last_time is not None and _minutes_between(current_start, last_time) >= 30:
        regimes.append(_regime_payload(current_start, last_time, trading_date))
    return regimes


def _regime_payload(start: datetime, end: datetime, trading_date: date) -> dict[str, Any]:
    return {
        "date": trading_date.isoformat(),
        "label": "SIDEWAYS?",
        "start": start.isoformat(),
        "end": end.isoformat(),
        "color": "#64748b",
        "alpha": 0.16,
    }


def _latest_trading_dates(candles: list[Candle], tz: ZoneInfo, days: int) -> list[date]:
    dates = sorted({candle.timestamp.astimezone(tz).date() for candle in candles})
    trading_dates = [item for item in dates if item.weekday() < 5]
    return trading_dates[-days:]


def _minutes_between(start: datetime, end: datetime) -> float:
    return (end - start).total_seconds() / 60


class _NullLogger:
    def info(self, *args, **kwargs):
        return None

    def debug(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def exception(self, *args, **kwargs):
        return None
