from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from ngn6_bot.legacy_signal import generate_legacy_signal
from ngn6_bot.models import OrderBookFeatures, Side, Signal, TradeFlowFeatures


@dataclass(frozen=True)
class SignalConfig:
    rsi_overbought: float
    rsi_oversold: float
    min_bollinger_width_pct: float
    volume_multiplier: float
    hold_above_ema_bars: int
    support_resistance_lookback: int
    max_level_distance_pct: float
    min_confidence: float
    min_imbalance: float
    trade_flow_min_buy_ratio: float
    trade_flow_min_sell_ratio: float
    news_halt: bool
    allow_rsi_extreme_with_strong_trend: bool
    require_15m_direction: bool = True
    engine: str = "python"
    legacy_bridge_path: str = "reference/ngn6_signal_source/bridge/compute_signal.js"
    legacy_node_command: str = "node"
    legacy_timeout_seconds: float = 3.0
    legacy_timeframe: str = "intraday"
    legacy_min_candles: int = 80
    legacy_max_candles: int = 320
    legacy_min_probability: float = 52.0
    legacy_news_bias: str = "auto"
    legacy_retest: str = "auto"
    legacy_structure: str = "auto"
    legacy_event_risk: str = "none"
    legacy_impulse_enabled: bool = True
    legacy_impulse_move_pct: float = 1.2
    legacy_impulse_breakout_buffer_pct: float = 0.03
    legacy_impulse_min_trend: float = 0.28
    legacy_impulse_min_momentum: float = 0.24
    legacy_impulse_min_candles: int = 2
    legacy_impulse_min_probability: float = 55.5
    legacy_impulse_max_probability: float = 63.5
    microstructure_enabled: bool = True
    require_microstructure: bool = False
    max_orderbook_age_seconds: float = 8.0
    min_book_pressure: float = 0.12
    min_trade_pressure: float = 0.12
    min_trade_flow_volume: float = 0.0
    require_trade_flow: bool = False
    max_adverse_mid_move_bps: float = 8.0
    max_spread_widening_bps: float = 8.0
    ema_adx_macd_warmup_candles: int = 80
    ema_adx_macd_min_adx: float = 20.0
    ema_adx_macd_require_adx_rising: bool = False
    ema_adx_macd_long_rsi_min: float = 50.0
    ema_adx_macd_long_rsi_max: float = 75.0
    ema_adx_macd_short_rsi_min: float = 30.0
    ema_adx_macd_short_rsi_max: float = 50.0
    ema_adx_macd_min_trend_strength: float = 0.002
    ema_adx_macd_min_signal_strength: float = 0.30
    ema_adx_macd_signal_strength_trend_scale: float = 0.0125
    ema_adx_macd_stop_atr_multiple: float = 1.5
    ema_adx_macd_take_profit_r_multiple: float = 2.5
    ema_adx_macd_orderbook_required: bool = True
    ema_adx_macd_always_trade: bool = False


def generate_signal(
    execution_df: pd.DataFrame,
    confirmation_df: pd.DataFrame,
    context_df: pd.DataFrame,
    orderbook: OrderBookFeatures,
    trade_flow: TradeFlowFeatures,
    config: SignalConfig,
    now: datetime,
) -> Signal:
    if config.news_halt:
        return _flat(now, "manual_news_halt_enabled")
    if config.engine == "legacy_ngn6":
        signal = generate_legacy_signal(
            execution_df=execution_df,
            context_df=context_df,
            orderbook=orderbook,
            config=config,
            now=now,
        )
        if signal.side == Side.FLAT:
            return signal
        micro_ok, micro_reason = _microstructure_allows(signal.side, orderbook, trade_flow, config)
        if not micro_ok:
            return _flat(now, micro_reason)
        return signal
    if config.engine == "ema_adx_macd":
        return _generate_ema_adx_macd_signal(context_df, orderbook, config, now)
    if execution_df.empty or len(execution_df) < max(25, config.support_resistance_lookback):
        return _flat(now, "not_enough_execution_candles")
    if context_df.empty or len(context_df) < 10:
        return _flat(now, "not_enough_15m_context")

    row = execution_df.iloc[-1]
    price = float(row["close"])

    bb_width = row.get("bb_width_pct")
    if pd.isna(bb_width) or float(bb_width) < config.min_bollinger_width_pct:
        return _flat(now, "bollinger_width_too_low")

    context = _context_state(context_df, config.support_resistance_lookback)
    confirmation = _confirmation_alignment(confirmation_df)
    held_above = _held_relative_to_emas(execution_df, config.hold_above_ema_bars, above=True)
    held_below = _held_relative_to_emas(execution_df, config.hold_above_ema_bars, above=False)
    volume_ok = _volume_confirmed(row, config.volume_multiplier)

    long_orderbook_ok = (
        orderbook.bid_ask_imbalance >= config.min_imbalance or orderbook.ask_wall_absorbed
    )
    short_orderbook_ok = (
        orderbook.bid_ask_imbalance <= 1 - config.min_imbalance or orderbook.bid_wall_absorbed
    )
    long_trade_flow_ok = trade_flow.buy_ratio >= config.trade_flow_min_buy_ratio
    short_trade_flow_ok = (1 - trade_flow.buy_ratio) >= config.trade_flow_min_sell_ratio

    entry_support, entry_resistance = _support_resistance(
        execution_df, config.support_resistance_lookback
    )
    support = context["support"] if context["support"] is not None else entry_support
    resistance = context["resistance"] if context["resistance"] is not None else entry_resistance
    support_retest = _near_level(price, support, config.max_level_distance_pct) and _bullish_candle(
        row
    )
    resistance_retest = _near_level(
        price, resistance, config.max_level_distance_pct
    ) and _bearish_candle(row)

    long_rsi_ok = _rsi_ok_for_long(row, context["bias"], config)
    short_rsi_ok = _rsi_ok_for_short(row, context["bias"], config)
    long_micro_ok, long_micro_reason = _microstructure_allows(
        Side.LONG, orderbook, trade_flow, config
    )
    short_micro_ok, short_micro_reason = _microstructure_allows(
        Side.SHORT, orderbook, trade_flow, config
    )

    long_context_ok = context["bias"] == "up" and price >= float(context["ema_slow"])
    short_context_ok = context["bias"] == "down" and price <= float(context["ema_slow"])
    if config.require_15m_direction and not long_context_ok and not short_context_ok:
        return _flat(now, f"15m_context_not_aligned:{context['bias']}")

    long_score = _score(
        [
            long_context_ok,
            held_above,
            confirmation in {"up", "neutral"},
            volume_ok,
            long_orderbook_ok,
            long_trade_flow_ok or trade_flow.total_volume == 0,
            long_micro_ok,
            support_retest or price > float(row["bb_mid"]),
            long_rsi_ok,
        ]
    )
    short_score = _score(
        [
            short_context_ok,
            held_below,
            confirmation in {"down", "neutral"},
            volume_ok,
            short_orderbook_ok,
            short_trade_flow_ok or trade_flow.total_volume == 0,
            short_micro_ok,
            resistance_retest or price < float(row["bb_mid"]),
            short_rsi_ok,
        ]
    )

    if long_context_ok and long_score >= config.min_confidence and long_score > short_score:
        if not long_micro_ok:
            return _flat(now, long_micro_reason)
        stop = min(float(row["low"]), support) if support else float(row["low"])
        return Signal(
            Side.LONG,
            long_score,
            "15m_context_up_1m_5m_entry_long",
            price,
            stop,
            now,
        )
    if short_context_ok and short_score >= config.min_confidence and short_score > long_score:
        if not short_micro_ok:
            return _flat(now, short_micro_reason)
        stop = max(float(row["high"]), resistance) if resistance else float(row["high"])
        return Signal(
            Side.SHORT,
            short_score,
            "15m_context_down_1m_5m_entry_short",
            price,
            stop,
            now,
        )
    return _flat(now, "no_signal")


def _flat(now: datetime, reason: str) -> Signal:
    return Signal(Side.FLAT, 0.0, reason, 0.0, None, now)


def microstructure_allows_entry(
    side: Side,
    orderbook: OrderBookFeatures,
    trade_flow: TradeFlowFeatures,
    config: SignalConfig,
) -> tuple[bool, str]:
    return _microstructure_allows(side, orderbook, trade_flow, config)


def _generate_ema_adx_macd_signal(
    context_df: pd.DataFrame,
    orderbook: OrderBookFeatures,
    config: SignalConfig,
    now: datetime,
) -> Signal:
    if context_df.empty or len(context_df) < config.ema_adx_macd_warmup_candles:
        return _flat(now, f"ema_adx_macd_not_enough_15m_candles:{len(context_df)}")

    row = context_df.iloc[-1]
    required_columns = ["ema_fast", "ema_slow", "atr", "adx", "rsi", "macd_hist"]
    if any(column not in row or pd.isna(row[column]) for column in required_columns):
        return _flat(now, "ema_adx_macd_missing_indicators")

    price = float(row["close"])
    ema_fast = float(row["ema_fast"])
    ema_slow = float(row["ema_slow"])
    atr_value = float(row["atr"])
    adx_value = float(row["adx"])
    previous_adx = float(context_df.iloc[-2]["adx"]) if len(context_df) >= 2 else adx_value
    rsi_value = float(row["rsi"])
    macd_hist = float(row["macd_hist"])
    if price <= 0 or ema_slow <= 0:
        return _flat(now, "ema_adx_macd_invalid_indicator_values")
    if atr_value <= 0:
        candle_range = abs(float(row.get("high", price)) - float(row.get("low", price)))
        atr_value = max(candle_range, price * 0.002)

    trend_strength = abs(ema_fast - ema_slow) / ema_slow
    signal_strength = min(
        1.0,
        trend_strength / max(config.ema_adx_macd_signal_strength_trend_scale, 1e-12),
    )
    metadata = {
        "strategy": "ema_adx_macd",
        "timeframe": "15min",
        "ema_fast": ema_fast,
        "ema_slow": ema_slow,
        "atr": atr_value,
        "adx": adx_value,
        "previous_adx": previous_adx,
        "rsi": rsi_value,
        "macd_hist": macd_hist,
        "trend_strength": trend_strength,
        "signal_strength": signal_strength,
    }

    warnings: list[str] = []
    if trend_strength < config.ema_adx_macd_min_trend_strength:
        warnings.append("trend_strength_below_min")
    if signal_strength < config.ema_adx_macd_min_signal_strength:
        warnings.append("signal_strength_below_min")
    if adx_value < config.ema_adx_macd_min_adx:
        warnings.append("adx_below_min")
    if config.ema_adx_macd_require_adx_rising and adx_value <= previous_adx:
        warnings.append("adx_not_rising")
    metadata["warnings"] = warnings
    if warnings and not config.ema_adx_macd_always_trade:
        return Signal(
            Side.FLAT,
            signal_strength,
            _reason("ema_adx_macd_filters_not_met", warnings),
            price,
            None,
            now,
            metadata=metadata,
        )

    long_setup = (
        ema_fast > ema_slow
        and macd_hist > 0
        and config.ema_adx_macd_long_rsi_min <= rsi_value <= config.ema_adx_macd_long_rsi_max
        and price >= ema_fast
    )
    short_setup = (
        ema_fast < ema_slow
        and macd_hist < 0
        and config.ema_adx_macd_short_rsi_min <= rsi_value <= config.ema_adx_macd_short_rsi_max
        and price <= ema_fast
    )

    if long_setup:
        orderbook_ok, orderbook_reason = _ema_adx_macd_orderbook_allows(
            Side.LONG,
            orderbook,
            config,
        )
        if not orderbook_ok:
            return Signal(
                Side.FLAT,
                signal_strength,
                orderbook_reason,
                price,
                None,
                now,
                metadata=metadata,
            )
        risk_distance = atr_value * config.ema_adx_macd_stop_atr_multiple
        return Signal(
            Side.LONG,
            signal_strength,
            _reason("ema_adx_macd_long", warnings),
            price,
            price - risk_distance,
            now,
            take_profit2=price + risk_distance * config.ema_adx_macd_take_profit_r_multiple,
            metadata=metadata,
        )

    if short_setup:
        orderbook_ok, orderbook_reason = _ema_adx_macd_orderbook_allows(
            Side.SHORT,
            orderbook,
            config,
        )
        if not orderbook_ok:
            return Signal(
                Side.FLAT,
                signal_strength,
                orderbook_reason,
                price,
                None,
                now,
                metadata=metadata,
            )
        risk_distance = atr_value * config.ema_adx_macd_stop_atr_multiple
        return Signal(
            Side.SHORT,
            signal_strength,
            _reason("ema_adx_macd_short", warnings),
            price,
            price + risk_distance,
            now,
            take_profit2=price - risk_distance * config.ema_adx_macd_take_profit_r_multiple,
            metadata=metadata,
        )

    if config.ema_adx_macd_always_trade:
        side, directional_confidence, directional_reason = _ema_adx_macd_fallback_side(
            context_df,
            ema_fast=ema_fast,
            ema_slow=ema_slow,
            rsi_value=rsi_value,
            macd_hist=macd_hist,
            orderbook=orderbook,
        )
        orderbook_ok, orderbook_reason = _ema_adx_macd_orderbook_allows(side, orderbook, config)
        metadata["fallback_reason"] = directional_reason
        metadata["orderbook"] = orderbook_reason
        if not orderbook_ok and config.ema_adx_macd_orderbook_required:
            return Signal(
                Side.FLAT,
                directional_confidence,
                orderbook_reason,
                price,
                None,
                now,
                metadata=metadata,
            )
        risk_distance = atr_value * config.ema_adx_macd_stop_atr_multiple
        if side == Side.LONG:
            return Signal(
                Side.LONG,
                max(signal_strength, directional_confidence),
                _reason("ema_adx_macd_fallback_long", warnings),
                price,
                price - risk_distance,
                now,
                take_profit2=price + risk_distance * config.ema_adx_macd_take_profit_r_multiple,
                metadata=metadata,
            )
        return Signal(
            Side.SHORT,
            max(signal_strength, directional_confidence),
            _reason("ema_adx_macd_fallback_short", warnings),
            price,
            price + risk_distance,
            now,
            take_profit2=price - risk_distance * config.ema_adx_macd_take_profit_r_multiple,
            metadata=metadata,
        )

    return Signal(
        Side.FLAT,
        signal_strength,
        "ema_adx_macd_no_entry",
        price,
        None,
        now,
        metadata=metadata,
    )


def _ema_adx_macd_orderbook_allows(
    side: Side,
    orderbook: OrderBookFeatures,
    config: SignalConfig,
) -> tuple[bool, str]:
    if not config.ema_adx_macd_orderbook_required:
        return True, "ema_adx_macd_orderbook_not_required"
    if _missing_orderbook(orderbook):
        return False, "ema_adx_macd_orderbook_required"
    if config.require_microstructure and orderbook.source in {"missing", "neutral"}:
        return False, f"ema_adx_macd_orderbook_untrusted:{orderbook.source}"
    if (
        orderbook.age_seconds is not None
        and orderbook.age_seconds > config.max_orderbook_age_seconds
    ):
        return False, f"ema_adx_macd_orderbook_stale:{orderbook.age_seconds:.1f}s"

    book_pressure = _book_pressure(orderbook)
    if side == Side.LONG and book_pressure < config.min_book_pressure:
        return False, f"ema_adx_macd_book_imbalance_too_weak:long:{book_pressure:.2f}"
    if side == Side.SHORT and book_pressure > -config.min_book_pressure:
        return False, f"ema_adx_macd_book_imbalance_too_weak:short:{book_pressure:.2f}"
    return True, "ema_adx_macd_orderbook_confirmed"


def _ema_adx_macd_fallback_side(
    context_df: pd.DataFrame,
    *,
    ema_fast: float,
    ema_slow: float,
    rsi_value: float,
    macd_hist: float,
    orderbook: OrderBookFeatures,
) -> tuple[Side, float, str]:
    row = context_df.iloc[-1]
    price = float(row["close"])
    previous_close = float(context_df.iloc[-2]["close"]) if len(context_df) >= 2 else price
    votes = [
        _sign(ema_fast - ema_slow),
        _sign(price - ema_slow),
        _sign(macd_hist),
        1 if rsi_value >= 50 else -1,
        _sign(price - previous_close),
        _sign(_book_pressure(orderbook)),
    ]
    score = sum(votes)
    if score == 0:
        score = 1 if price >= ema_slow else -1
    confidence = min(1.0, 0.5 + abs(score) / (len(votes) * 2))
    side = Side.LONG if score > 0 else Side.SHORT
    return side, confidence, f"votes={score};price={price};prev_close={previous_close}"


def _sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _reason(base: str, warnings: list[str]) -> str:
    if not warnings:
        return base
    return f"{base}:soft={','.join(warnings)}"


def _held_relative_to_emas(df: pd.DataFrame, bars: int, *, above: bool) -> bool:
    recent = df.tail(bars)
    if recent.empty or len(recent) < bars:
        return False
    if above:
        return bool(((recent["close"] > recent["ema_fast"]) & (recent["close"] > recent["ema_slow"])).all())
    return bool(((recent["close"] < recent["ema_fast"]) & (recent["close"] < recent["ema_slow"])).all())


def _volume_confirmed(row: pd.Series, multiplier: float) -> bool:
    volume_ma = row.get("volume_ma")
    if pd.isna(volume_ma) or float(volume_ma) <= 0:
        return False
    return float(row["volume"]) >= float(volume_ma) * multiplier


def _context_state(context_df: pd.DataFrame, lookback: int) -> dict[str, float | str | None]:
    last = context_df.iloc[-1]
    prev = context_df.iloc[-3]
    support, resistance = _support_resistance(context_df, min(lookback, len(context_df)))
    bias = "neutral"
    if (
        last["close"] > last["ema_fast"] > last["ema_slow"]
        and last["ema_fast"] > prev["ema_fast"]
        and last["ema_slow"] >= prev["ema_slow"]
    ):
        bias = "up"
    elif (
        last["close"] < last["ema_fast"] < last["ema_slow"]
        and last["ema_fast"] < prev["ema_fast"]
        and last["ema_slow"] <= prev["ema_slow"]
    ):
        bias = "down"
    return {
        "bias": bias,
        "support": support,
        "resistance": resistance,
        "ema_fast": float(last["ema_fast"]),
        "ema_slow": float(last["ema_slow"]),
    }


def _confirmation_alignment(confirmation_df: pd.DataFrame) -> str:
    if confirmation_df.empty or len(confirmation_df) < 2:
        return "neutral"
    last = confirmation_df.iloc[-1]
    if last["close"] > last["ema_fast"] > last["ema_slow"]:
        return "up"
    if last["close"] < last["ema_fast"] < last["ema_slow"]:
        return "down"
    return "neutral"


def _support_resistance(df: pd.DataFrame, lookback: int) -> tuple[float | None, float | None]:
    recent = df.tail(lookback)
    if recent.empty:
        return None, None
    return float(recent["low"].min()), float(recent["high"].max())


def _near_level(price: float, level: float | None, max_distance_pct: float) -> bool:
    if level is None or level <= 0:
        return False
    return abs(price - level) / level * 100 <= max_distance_pct


def _bullish_candle(row: pd.Series) -> bool:
    return float(row["close"]) > float(row["open"]) and float(row["close"]) >= float(row["high"]) * 0.995


def _bearish_candle(row: pd.Series) -> bool:
    return float(row["close"]) < float(row["open"]) and float(row["close"]) <= float(row["low"]) * 1.005


def _rsi_ok_for_long(row: pd.Series, trend: str, config: SignalConfig) -> bool:
    rsi = float(row["rsi"])
    if rsi <= config.rsi_overbought:
        return True
    return config.allow_rsi_extreme_with_strong_trend and trend == "up"


def _rsi_ok_for_short(row: pd.Series, trend: str, config: SignalConfig) -> bool:
    rsi = float(row["rsi"])
    if rsi >= config.rsi_oversold:
        return True
    return config.allow_rsi_extreme_with_strong_trend and trend == "down"


def _microstructure_allows(
    side: Side,
    orderbook: OrderBookFeatures,
    trade_flow: TradeFlowFeatures,
    config: SignalConfig,
) -> tuple[bool, str]:
    if not config.microstructure_enabled:
        return True, "microstructure_disabled"

    if _missing_orderbook(orderbook):
        if config.require_microstructure:
            return False, "microstructure_missing_orderbook"
        return True, "microstructure_missing_allowed"
    if config.require_microstructure and orderbook.source in {"missing", "neutral"}:
        return False, f"microstructure_untrusted_orderbook:{orderbook.source}"
    if (
        orderbook.age_seconds is not None
        and orderbook.age_seconds > config.max_orderbook_age_seconds
    ):
        return False, f"microstructure_orderbook_stale:{orderbook.age_seconds:.1f}s"

    book_pressure = _book_pressure(orderbook)
    flow_pressure = _flow_pressure(trade_flow)
    if side == Side.LONG:
        book_ok = book_pressure >= config.min_book_pressure or orderbook.ask_wall_absorbed
        flow_ok = flow_pressure >= config.min_trade_pressure
        adverse_mid = (
            orderbook.mid_price_change_bps is not None
            and orderbook.mid_price_change_bps < -config.max_adverse_mid_move_bps
        )
    elif side == Side.SHORT:
        book_ok = book_pressure <= -config.min_book_pressure or orderbook.bid_wall_absorbed
        flow_ok = flow_pressure <= -config.min_trade_pressure
        adverse_mid = (
            orderbook.mid_price_change_bps is not None
            and orderbook.mid_price_change_bps > config.max_adverse_mid_move_bps
        )
    else:
        return True, "microstructure_flat"

    if not book_ok:
        return False, f"microstructure_book_not_confirmed:{side.value}:{book_pressure:.2f}"
    if (
        orderbook.spread_change_bps is not None
        and orderbook.spread_change_bps > config.max_spread_widening_bps
    ):
        return False, f"microstructure_spread_widening:{orderbook.spread_change_bps:.1f}bps"
    if adverse_mid:
        return False, (
            f"microstructure_adverse_mid_move:{side.value}:"
            f"{orderbook.mid_price_change_bps:.1f}bps"
        )
    if trade_flow.total_volume < config.min_trade_flow_volume:
        if config.require_trade_flow:
            return False, "microstructure_trade_flow_missing"
        return True, "microstructure_trade_flow_missing_allowed"
    if trade_flow.total_volume == 0 and not config.require_trade_flow:
        return True, "microstructure_trade_flow_empty_allowed"
    if not flow_ok:
        return False, f"microstructure_trade_not_confirmed:{side.value}:{flow_pressure:.2f}"
    return True, "microstructure_confirmed"


def _missing_orderbook(orderbook: OrderBookFeatures) -> bool:
    return (
        orderbook.best_bid is None
        or orderbook.best_ask is None
        or orderbook.mid_price is None
        or orderbook.spread_bps is None
    )


def _book_pressure(orderbook: OrderBookFeatures) -> float:
    if orderbook.depth_pressure:
        return float(orderbook.depth_pressure)
    return float((orderbook.bid_ask_imbalance - 0.5) * 2)


def _flow_pressure(trade_flow: TradeFlowFeatures) -> float:
    if trade_flow.directional_volume > 0:
        return float(trade_flow.buy_sell_imbalance)
    return float((trade_flow.buy_ratio - 0.5) * 2)


def _score(conditions: list[bool]) -> float:
    return sum(1 for condition in conditions if condition) / len(conditions)
