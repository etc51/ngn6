from __future__ import annotations

import json
import copy
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ngn6_bot.bot import TradingBot
from ngn6_bot.config import RuntimeConfig
from ngn6_bot.costs import ExecutionCostConfig, cost_pct, round_trip_cost_ticks, trade_covers_costs
from ngn6_bot.indicators import add_indicators, candles_to_frame
from ngn6_bot.microstructure_replay import MicrostructureReplay, neutral_orderbook, neutral_trade_flow
from ngn6_bot.models import Candle, MarketState, Position, Side
from ngn6_bot.risk import (
    calculate_position_lots,
    move_stop_to_breakeven,
    must_flatten_before_clearing,
    stop_with_buffer,
    update_trailing_stop,
)
from ngn6_bot.signals import generate_signal
from ngn6_bot.tbank import TInvestGateway, candle_interval_for_polling


@dataclass(frozen=True)
class BacktestTrade:
    side: str
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    lots: int
    pnl_pct: float
    reason: str


@dataclass(frozen=True)
class BacktestMetrics:
    trades: int
    win_rate_pct: float
    avg_trade_pct: float
    gross_profit_pct: float
    gross_loss_pct: float
    profit_factor: float | None
    max_drawdown_pct: float
    final_equity_pct: float


@dataclass(frozen=True)
class BacktestReport:
    ticker: str
    figi: str
    candles: int
    started_at: str
    finished_at: str
    metrics: BacktestMetrics
    trades: list[BacktestTrade]
    limitations: list[str]

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


@dataclass(frozen=True)
class WalkForwardFold:
    fold: int
    started_at: str
    finished_at: str
    candles: int
    metrics: BacktestMetrics


@dataclass(frozen=True)
class WalkForwardReport:
    ticker: str
    figi: str
    folds: list[WalkForwardFold]
    limitations: list[str]

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


def fetch_1m_history(config: RuntimeConfig, logger, minutes_back: int) -> tuple[str, list[Candle]]:
    with TInvestGateway(config.token, config.raw, logger) as gateway:
        figi, _ = gateway.resolve_instrument()
        candles = _fetch_chunked_1m(gateway, figi, minutes_back)
        return figi, candles


def _fetch_chunked_1m(gateway: TInvestGateway, figi: str, minutes_back: int) -> list[Candle]:
    # T-Invest limits the requested range for 1m candles. Keep chunks conservative.
    chunk_minutes = 24 * 60
    interval = candle_interval_for_polling("1min")
    end = datetime.now(timezone.utc)
    start = end - timedelta(minutes=minutes_back)
    candles_by_time: dict[datetime, Candle] = {}
    current = start
    while current < end:
        chunk_end = min(current + timedelta(minutes=chunk_minutes), end)
        response = gateway.client.market_data.get_candles(
            figi=figi,
            from_=current,
            to=chunk_end,
            interval=interval,
        )
        for candle in response.candles:
            if getattr(candle, "is_complete", True):
                parsed = Candle(
                    timestamp=candle.time,
                    open=_quotation_to_float(candle.open),
                    high=_quotation_to_float(candle.high),
                    low=_quotation_to_float(candle.low),
                    close=_quotation_to_float(candle.close),
                    volume=float(candle.volume),
                    timeframe="1min",
                )
                candles_by_time[parsed.timestamp] = parsed
        current = chunk_end
    return [candles_by_time[key] for key in sorted(candles_by_time)]


def run_replay_backtest(
    config: RuntimeConfig,
    candles: list[Candle],
    figi: str,
    *,
    promoted_only: bool = False,
) -> BacktestReport:
    config = _backtest_runtime_config(config, promoted_only=promoted_only)
    state = MarketState()
    bot = TradingBot(config, logger=_NullLogger(), runtime_services=False)
    risk_config = bot._risk_config()
    signal_config = bot._signal_config()
    cost_config = _execution_cost_config(config)
    microstructure = MicrostructureReplay.from_config(config)
    microstructure_hits = 0
    microstructure_misses = 0
    trades: list[BacktestTrade] = []
    equity_pct = 0.0
    peak_equity = 0.0
    max_drawdown = 0.0
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

    for candle in candles:
        state.update_candle(candle)
        bot.state = state
        now = candle.timestamp if candle.timestamp.tzinfo else candle.timestamp.replace(tzinfo=timezone.utc)

        if active_position.is_open:
            if bot._ml_control_enabled() and len(state.candles_1m) >= 40:
                execution_df = _indicator_frame(list(state.candles_1m), config)
                confirmation_df = _indicator_frame(list(state.candles_5m), config)
                context_df = _indicator_frame(list(state.candles_15m), config)
                orderbook, trade_flow, hit = _microstructure_at(
                    microstructure,
                    now,
                    candle.close,
                )
                if hit:
                    microstructure_hits += 1
                elif microstructure.available:
                    microstructure_misses += 1
                ml_exit_reason, _ = bot.feedback_model.exit_reason_from_prediction(
                    position_side=active_position.side,
                    execution_df=execution_df,
                    confirmation_df=confirmation_df,
                    context_df=context_df,
                    orderbook=orderbook,
                    trade_flow=trade_flow,
                    now=now,
                )
                if ml_exit_reason:
                    trade = _close_trade(
                        active_position,
                        candle,
                        candle.close,
                        ml_exit_reason,
                        cost_config,
                    )
                    trades.append(trade)
                    equity_pct += trade.pnl_pct
                    peak_equity = max(peak_equity, equity_pct)
                    max_drawdown = min(max_drawdown, equity_pct - peak_equity)
                    last_full_exit = (active_position.side, now, ml_exit_reason, trade.pnl_pct)
                    active_position = Position()
                    state.position = active_position
                    continue
            exit_reason = _exit_reason(active_position, candle, risk_config, now)
            if exit_reason:
                exit_price = _exit_price(active_position, candle, exit_reason)
                trade = _close_trade(active_position, candle, exit_price, exit_reason, cost_config)
                trades.append(trade)
                equity_pct += trade.pnl_pct
                peak_equity = max(peak_equity, equity_pct)
                max_drawdown = min(max_drawdown, equity_pct - peak_equity)
                last_full_exit = (active_position.side, now, exit_reason, trade.pnl_pct)
                active_position = Position()
                state.position = active_position
                continue
            move_stop_to_breakeven(active_position, candle.close, risk_config)
            update_trailing_stop(active_position, candle.close, risk_config)
            continue

        if len(state.candles_1m) < 40 or must_flatten_before_clearing(now, risk_config):
            continue

        execution_df = _indicator_frame(list(state.candles_1m), config)
        confirmation_df = _indicator_frame(list(state.candles_5m), config)
        context_df = _indicator_frame(list(state.candles_15m), config)
        orderbook, trade_flow, hit = _microstructure_at(microstructure, now, candle.close)
        if hit:
            microstructure_hits += 1
        elif microstructure.available:
            microstructure_misses += 1
        if bot._ml_control_enabled():
            signal = bot.feedback_model.signal_from_prediction(
                execution_df=execution_df,
                confirmation_df=confirmation_df,
                context_df=context_df,
                orderbook=orderbook,
                trade_flow=trade_flow,
                now=now,
                price=candle.close,
            )
        else:
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
        if bot._entry_signal_block_reason(signal):
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

        cost_check = trade_covers_costs(
            price=signal.price,
            expected_move_pct=bot._expected_move_pct(signal, risk_config),
            spread_bps=orderbook.spread_bps or 0.0,
            config=cost_config,
        )
        if not cost_check.accepted:
            continue

        stop_price = stop_with_buffer(signal, risk_config)
        normalized_signal = replace(signal, stop_price=stop_price)
        lots = calculate_position_lots(normalized_signal, risk_config)
        if lots <= 0 or stop_price is None:
            continue
        risk_reason, _ = bot._final_entry_risk_check(
            signal=normalized_signal,
            lots=lots,
            now=now,
            risk_config=risk_config,
            orderbook_features=orderbook,
            cost_check=cost_check,
        )
        if risk_reason:
            continue
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

    if active_position.is_open and candles:
        last = candles[-1]
        trade = _close_trade(active_position, last, last.close, "end_of_backtest", cost_config)
        trades.append(trade)
        equity_pct += trade.pnl_pct
        peak_equity = max(peak_equity, equity_pct)
        max_drawdown = min(max_drawdown, equity_pct - peak_equity)

    metrics = _metrics(trades, equity_pct, max_drawdown)
    started_at = candles[0].timestamp.isoformat() if candles else ""
    finished_at = candles[-1].timestamp.isoformat() if candles else ""
    return BacktestReport(
        ticker=str(config.get("instrument", "ticker")),
        figi=figi,
        candles=len(candles),
        started_at=started_at,
        finished_at=finished_at,
        metrics=metrics,
        trades=trades,
        limitations=_backtest_limitations(microstructure, microstructure_hits, microstructure_misses),
    )


def save_report(report: BacktestReport, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(report.to_json(), encoding="utf-8")


def run_walk_forward(
    config: RuntimeConfig,
    candles: list[Candle],
    figi: str,
    folds: int,
    *,
    promoted_only: bool = False,
) -> WalkForwardReport:
    if folds <= 0:
        raise ValueError("folds must be positive.")
    fold_size = max(1, len(candles) // folds)
    results: list[WalkForwardFold] = []
    for index in range(folds):
        start = index * fold_size
        end = len(candles) if index == folds - 1 else (index + 1) * fold_size
        fold_candles = candles[start:end]
        if not fold_candles:
            continue
        report = run_replay_backtest(config, fold_candles, figi, promoted_only=promoted_only)
        results.append(
            WalkForwardFold(
                fold=index + 1,
                started_at=report.started_at,
                finished_at=report.finished_at,
                candles=report.candles,
                metrics=report.metrics,
            )
        )
    return WalkForwardReport(
        ticker=str(config.get("instrument", "ticker")),
        figi=figi,
        folds=results,
        limitations=[
            "This is chronological out-of-sample folding without parameter optimization.",
            "Historical order book replay is unavailable in this data path; configured slippage and commission are still deducted.",
        ],
    )


def save_walk_forward_report(report: WalkForwardReport, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(report.to_json(), encoding="utf-8")


def _backtest_limitations(
    microstructure: MicrostructureReplay,
    hits: int,
    misses: int,
) -> list[str]:
    if microstructure.available:
        limitations = [
            f"Backtest replays logged order-book/trade-flow snapshots when available: {hits} matched candle evaluations.",
            "Logged microstructure is sampled live data, not an exchange-native historical queue replay.",
            "Backtest deducts configured slippage and commission; actual queue position is still approximate.",
        ]
        if misses:
            limitations.append(
                f"{misses} candle evaluations had no nearby microstructure snapshot and used neutral fallback."
            )
        return limitations
    return [
        "No logged microstructure replay file was found; backtest uses candles with neutral order book/trade-flow fallback.",
        "With microstructure.require_for_entry enabled, candle-only backtests will block entries until replay logs exist.",
        "Backtest deducts configured slippage and commission, but historical spread and queue position are not replayed.",
    ]


def _backtest_runtime_config(config: RuntimeConfig, *, promoted_only: bool) -> RuntimeConfig:
    raw = copy.deepcopy(config.raw)
    raw.setdefault("bot", {})["dry_run"] = True
    raw.setdefault("trading", {})["dry_run"] = True
    raw["trading"]["live_enabled"] = False
    raw.setdefault("execution", {})["live_enabled"] = False
    raw["execution"]["require_live_orderbook"] = False
    raw["execution"]["block_stale_entries"] = True
    raw["execution"]["allow_fallback_entries"] = False
    raw.setdefault("signals", {})["allow_fallback_entries"] = False
    raw.setdefault("market_data", {})["required_entry_orderbook_source"] = "replay"
    raw["market_data"]["block_entries_when_stale"] = True
    raw["market_data"]["max_entry_staleness_seconds"] = float(
        raw.get("backtest", {}).get("microstructure_replay_max_age_seconds", 75.0)
    )
    if promoted_only:
        raw.setdefault("learning", {})["enabled"] = True
        raw["learning"]["mode"] = "shadow_then_control"
        raw["learning"]["control_require_ensemble_model"] = True
        raw["learning"]["control_require_schema_v2"] = True
        raw["learning"]["control_require_promoted_model"] = True
        raw["learning"]["active_can_trade_only_if_promoted"] = True
        raw["execution"]["allow_trade_without_promoted_model"] = False
    return RuntimeConfig(raw=raw, path=config.path)


def _indicator_frame(candles: list[Candle], config: RuntimeConfig):
    return add_indicators(
        candles_to_frame(candles),
        ema_fast=int(config.get("indicators", "ema_fast")),
        ema_slow=int(config.get("indicators", "ema_slow")),
        rsi_period=int(config.get("indicators", "rsi_period")),
        atr_period=int(config.get("indicators", "atr_period", default=14)),
        adx_period=int(config.get("indicators", "adx_period", default=14)),
        macd_fast=int(config.get("indicators", "macd_fast", default=12)),
        macd_slow=int(config.get("indicators", "macd_slow", default=26)),
        macd_signal=int(config.get("indicators", "macd_signal", default=9)),
        bollinger_period=int(config.get("indicators", "bollinger_period")),
        bollinger_std=float(config.get("indicators", "bollinger_std")),
        volume_ma_period=int(config.get("indicators", "volume_ma_period")),
    )


def _microstructure_at(microstructure: MicrostructureReplay, now: datetime, price: float):
    micro_snapshot = microstructure.at(now)
    if micro_snapshot is None:
        return neutral_orderbook(price), neutral_trade_flow(), False
    return micro_snapshot.orderbook, micro_snapshot.trade_flow, True


def _exit_reason(position: Position, candle: Candle, risk_config, now: datetime) -> str | None:
    if must_flatten_before_clearing(now, risk_config):
        return "pre_clearing_flatten"
    if position.side == Side.LONG:
        if position.stop_price is not None and candle.low <= position.stop_price:
            return "hard_stop_hit"
        if position.take_profit2 is not None and candle.high >= position.take_profit2:
            return "take_profit_2_5r"
        if position.take_profit1 is not None and candle.high >= position.take_profit1:
            return "take_profit"
        if risk_config.partial_take_profit_pct > 0:
            target = position.avg_price * (1 + risk_config.partial_take_profit_pct / 100)
            if candle.high >= target:
                return "take_profit"
    if position.side == Side.SHORT:
        if position.stop_price is not None and candle.high >= position.stop_price:
            return "hard_stop_hit"
        if position.take_profit2 is not None and candle.low <= position.take_profit2:
            return "take_profit_2_5r"
        if position.take_profit1 is not None and candle.low <= position.take_profit1:
            return "take_profit"
        if risk_config.partial_take_profit_pct > 0:
            target = position.avg_price * (1 - risk_config.partial_take_profit_pct / 100)
            if candle.low <= target:
                return "take_profit"
    return None


def _blocked_by_reentry_cooldown(
    side: Side,
    now: datetime,
    last_full_exit: tuple[Side, datetime, str] | tuple[Side, datetime, str, float] | None,
    cooldown_minutes: float,
    *,
    exit_cooldown_minutes: float = 0.0,
    loss_cooldown_minutes: float = 0.0,
) -> bool:
    if last_full_exit is None:
        return False
    exit_side, exit_time, exit_reason = last_full_exit[:3]
    exit_pnl = float(last_full_exit[3]) if len(last_full_exit) >= 4 else 0.0
    elapsed = now - exit_time
    if loss_cooldown_minutes > 0 and exit_pnl < 0 and elapsed <= timedelta(
        minutes=loss_cooldown_minutes
    ):
        return True
    if exit_cooldown_minutes > 0 and elapsed <= timedelta(minutes=exit_cooldown_minutes):
        return True
    if cooldown_minutes <= 0:
        return False
    if exit_side != side or not _is_take_profit_exit(exit_reason):
        return False
    return elapsed <= timedelta(minutes=cooldown_minutes)


def _is_take_profit_exit(reason: str) -> bool:
    return "take_profit" in reason


def _exit_price(position: Position, candle: Candle, reason: str) -> float:
    if reason in {"hard_stop", "hard_stop_hit"} and position.stop_price is not None:
        return position.stop_price
    if reason == "take_profit":
        if position.take_profit1 is not None:
            return position.take_profit1
        if position.side == Side.LONG:
            return position.avg_price * 1.012
        return position.avg_price * 0.988
    if reason in {"legacy_take_profit2", "take_profit_2_5r"} and position.take_profit2 is not None:
        return position.take_profit2
    return candle.close


def _close_trade(
    position: Position,
    candle: Candle,
    exit_price: float,
    reason: str,
    cost_config: ExecutionCostConfig,
) -> BacktestTrade:
    if position.side == Side.LONG:
        pnl_pct = (exit_price - position.avg_price) / position.avg_price * 100
    else:
        pnl_pct = (position.avg_price - exit_price) / position.avg_price * 100
    costs_pct = cost_pct(
        position.avg_price,
        round_trip_cost_ticks(position.avg_price, 0.0, cost_config),
        cost_config.min_price_increment,
    )
    pnl_pct -= costs_pct
    return BacktestTrade(
        side=position.side.value,
        entry_time=(position.opened_at.isoformat() if position.opened_at else ""),
        exit_time=candle.timestamp.isoformat(),
        entry_price=position.avg_price,
        exit_price=exit_price,
        lots=position.lots,
        pnl_pct=pnl_pct,
        reason=reason,
    )


def _execution_cost_config(config: RuntimeConfig) -> ExecutionCostConfig:
    return ExecutionCostConfig(
        slippage_bps_assumption=float(config.get("execution", "slippage_bps_assumption")),
        commission_per_lot_per_side=float(
            config.get("execution", "commission_per_lot_per_side", default=0.0)
        ),
        commission_round_trip_bps=float(
            config.get("execution", "commission_round_trip_bps", default=0.0)
        ),
        min_expected_net_ticks=float(config.get("execution", "min_expected_net_ticks", default=0.0)),
        min_price_increment=float(config.get("instrument", "min_price_increment")),
        money_value_per_price_step=float(config.get("instrument", "money_value_per_price_step")),
    )


def _metrics(trades: list[BacktestTrade], final_equity_pct: float, max_drawdown_pct: float) -> BacktestMetrics:
    if not trades:
        return BacktestMetrics(0, 0.0, 0.0, 0.0, 0.0, None, max_drawdown_pct, final_equity_pct)
    wins = [trade.pnl_pct for trade in trades if trade.pnl_pct > 0]
    losses = [trade.pnl_pct for trade in trades if trade.pnl_pct <= 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = gross_profit / gross_loss if gross_loss else None
    return BacktestMetrics(
        trades=len(trades),
        win_rate_pct=len(wins) / len(trades) * 100,
        avg_trade_pct=sum(trade.pnl_pct for trade in trades) / len(trades),
        gross_profit_pct=gross_profit,
        gross_loss_pct=gross_loss,
        profit_factor=profit_factor,
        max_drawdown_pct=max_drawdown_pct,
        final_equity_pct=final_equity_pct,
    )


class _NullLogger:
    def info(self, *args, **kwargs):
        return None

    def debug(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def exception(self, *args, **kwargs):
        return None


def _quotation_to_float(value) -> float:
    return float(value.units) + float(value.nano) / 1_000_000_000
