from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

from ngn6_bot.models import Position, Side, Signal


@dataclass(frozen=True)
class RiskConfig:
    deposit_value: float
    risk_per_trade_pct: float
    max_risk_per_trade_pct: float
    max_position_lots: int
    min_position_lots: int
    stop_buffer_ticks: int
    min_price_increment: float
    money_value_per_price_step: float
    partial_take_profit_pct: float
    partial_take_fraction: float
    trailing_stop_pct: float
    close_before_clearing_minutes: int
    clearings: list[str]
    timezone: str
    take_profit_r_multiple: float = 2.5
    breakeven_trigger_pct: float = 0.75
    trailing_profit_trigger_rub: float = 0.0
    trailing_profit_lock_ratio: float = 0.35
    notional_multiplier: float = 0.0
    max_gross_exposure_multiplier: float = 0.0
    max_position_exposure_pct: float = 0.0
    cash_reserve_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    daily_max_loss_pct: float = 0.0
    stop_after_consecutive_losses: int = 0
    stop_after_consecutive_hard_stops: int = 0


def calculate_position_lots(signal: Signal, config: RiskConfig) -> int:
    if signal.side == Side.FLAT or signal.stop_price is None:
        return 0

    risk_fraction = min(
        _fraction_from_percent_or_fraction(config.risk_per_trade_pct),
        _fraction_from_percent_or_fraction(config.max_risk_per_trade_pct),
    )
    risk_money = config.deposit_value * risk_fraction
    stop_distance = abs(signal.price - signal.stop_price)
    if stop_distance <= 0:
        stop_distance = config.min_price_increment * config.stop_buffer_ticks

    price_steps = stop_distance / config.min_price_increment
    risk_per_lot = price_steps * config.money_value_per_price_step
    if risk_per_lot <= 0:
        return 0

    lots = int(risk_money // risk_per_lot)
    lots = max(0, min(lots, config.max_position_lots))
    lots = min(lots, _exposure_cap_lots(signal.price, config))
    if lots < config.min_position_lots:
        return 0
    return lots


def stop_with_buffer(signal: Signal, config: RiskConfig) -> float | None:
    if signal.stop_price is None:
        return None
    buffer_value = config.stop_buffer_ticks * config.min_price_increment
    if signal.side == Side.LONG:
        return signal.stop_price - buffer_value
    if signal.side == Side.SHORT:
        return signal.stop_price + buffer_value
    return None


def should_take_partial(position: Position, last_price: float, config: RiskConfig) -> bool:
    if not position.is_open or position.partial_taken:
        return False
    if config.partial_take_fraction <= 0 or config.partial_take_profit_pct <= 0:
        return False
    pnl_pct = unrealized_pnl_pct(position, last_price)
    return pnl_pct >= config.partial_take_profit_pct


def update_trailing_stop(position: Position, last_price: float, config: RiskConfig) -> None:
    if not position.is_open:
        return
    if config.trailing_profit_trigger_rub > 0:
        profit_money = unrealized_pnl_money(position, last_price, config)
        if profit_money < config.trailing_profit_trigger_rub:
            return
        lock_ratio = config.trailing_profit_lock_ratio
        if lock_ratio > 1:
            lock_ratio /= 100
        lock_ratio = min(max(lock_ratio, 0.0), 1.0)
        quantity_units = _quantity_units(position, config)
        if quantity_units <= 0 or lock_ratio <= 0:
            return
        locked_price_move = profit_money * lock_ratio / quantity_units
        if position.side == Side.LONG and last_price > position.avg_price:
            new_stop = position.avg_price + locked_price_move
            position.stop_price = max(position.stop_price or new_stop, new_stop)
            position.trailing_stop = max(position.trailing_stop or new_stop, new_stop)
        elif position.side == Side.SHORT and last_price < position.avg_price:
            new_stop = position.avg_price - locked_price_move
            position.stop_price = min(position.stop_price or new_stop, new_stop)
            position.trailing_stop = min(position.trailing_stop or new_stop, new_stop)
        return

    if config.trailing_stop_pct <= 0:
        return
    distance = last_price * config.trailing_stop_pct / 100
    if position.side == Side.LONG:
        new_stop = last_price - distance
        position.stop_price = max(position.stop_price or new_stop, new_stop)
        position.trailing_stop = max(position.trailing_stop or new_stop, new_stop)
    elif position.side == Side.SHORT:
        new_stop = last_price + distance
        position.stop_price = min(position.stop_price or new_stop, new_stop)
        position.trailing_stop = min(position.trailing_stop or new_stop, new_stop)


def move_stop_to_breakeven(position: Position, last_price: float, config: RiskConfig) -> bool:
    if not position.is_open or position.stop_price is None or config.breakeven_trigger_pct <= 0:
        return False
    if unrealized_pnl_pct(position, last_price) < config.breakeven_trigger_pct:
        return False
    if position.side == Side.LONG:
        if position.stop_price >= position.avg_price:
            return False
        position.stop_price = position.avg_price
        position.trailing_stop = max(position.trailing_stop or position.avg_price, position.avg_price)
        return True
    if position.side == Side.SHORT:
        if position.stop_price <= position.avg_price:
            return False
        position.stop_price = position.avg_price
        position.trailing_stop = min(position.trailing_stop or position.avg_price, position.avg_price)
        return True
    return False


def trailing_stop_hit(position: Position, last_price: float) -> bool:
    if not position.is_open or position.trailing_stop is None:
        return False
    if position.side == Side.LONG:
        return last_price <= position.trailing_stop
    return last_price >= position.trailing_stop


def hard_stop_hit(position: Position, last_price: float) -> bool:
    if not position.is_open or position.stop_price is None:
        return False
    if position.side == Side.LONG:
        return last_price <= position.stop_price
    return last_price >= position.stop_price


def unrealized_pnl_pct(position: Position, last_price: float) -> float:
    if not position.is_open or position.avg_price <= 0:
        return 0.0
    if position.side == Side.LONG:
        return (last_price - position.avg_price) / position.avg_price * 100
    return (position.avg_price - last_price) / position.avg_price * 100


def unrealized_pnl_money(position: Position, last_price: float, config: RiskConfig) -> float:
    if not position.is_open or config.min_price_increment <= 0:
        return 0.0
    move = last_price - position.avg_price
    if position.side == Side.SHORT:
        move = -move
    return move * _quantity_units(position, config)


def must_flatten_before_clearing(now: datetime, config: RiskConfig) -> bool:
    if config.close_before_clearing_minutes <= 0 or not config.clearings:
        return False
    tz = ZoneInfo(config.timezone)
    local_now = now.astimezone(tz)
    today = local_now.date()
    window = timedelta(minutes=config.close_before_clearing_minutes)
    for clearing_value in config.clearings:
        clearing_time = time.fromisoformat(clearing_value)
        clearing_dt = datetime.combine(today, clearing_time, tzinfo=tz)
        if clearing_dt - window <= local_now <= clearing_dt:
            return True
    return False


def trading_session_block_reason(
    now: datetime,
    *,
    timezone: str,
    trading_start: str | None,
    trading_end: str | None,
    forced_flat_hours: list[object] | None = None,
    forced_flat_weekdays: list[object] | None = None,
) -> str | None:
    local_now = now.astimezone(ZoneInfo(timezone))
    if _weekday_is_forced_flat(local_now, forced_flat_weekdays or []):
        return "forced_flat_weekday"
    if _hour_is_forced_flat(local_now, forced_flat_hours or []):
        return "forced_flat_hour"
    if not trading_start or not trading_end:
        return None

    start = time.fromisoformat(str(trading_start))
    end = time.fromisoformat(str(trading_end))
    current = local_now.timetz().replace(tzinfo=None)
    if start == end:
        return None
    if start < end:
        allowed = start <= current < end
    else:
        allowed = current >= start or current < end
    return None if allowed else "outside_trading_session"


def drawdown_limit_hit(equity: float, initial_equity: float, config: RiskConfig) -> bool:
    if config.max_drawdown_pct <= 0 or initial_equity <= 0:
        return False
    drawdown_fraction = (initial_equity - equity) / initial_equity
    return drawdown_fraction >= _fraction_from_percent_or_fraction(config.max_drawdown_pct)


def liquidity_covers_lots(
    side: Side,
    lots: int,
    bid_depth: float,
    ask_depth: float,
    min_cover: float,
) -> bool:
    if min_cover <= 0 or lots <= 0:
        return True
    available = ask_depth if side == Side.LONG else bid_depth
    return available >= lots * min_cover


def _exposure_cap_lots(price: float, config: RiskConfig) -> int:
    if price <= 0 or config.notional_multiplier <= 0:
        return config.max_position_lots

    exposure_per_lot = abs(price * config.notional_multiplier)
    if exposure_per_lot <= 0:
        return config.max_position_lots

    caps = [config.max_position_lots]
    reserve_multiplier = max(0.0, 1 - _fraction_from_percent_or_fraction(config.cash_reserve_pct))
    if config.max_gross_exposure_multiplier > 0:
        gross_limit = config.deposit_value * config.max_gross_exposure_multiplier * reserve_multiplier
        caps.append(math.floor(gross_limit / exposure_per_lot))
    if config.max_position_exposure_pct > 0:
        position_limit = config.deposit_value * _fraction_from_percent_or_fraction(
            config.max_position_exposure_pct
        )
        caps.append(math.floor(position_limit / exposure_per_lot))
    return max(0, min(caps))


def _fraction_from_percent_or_fraction(value: float) -> float:
    parsed = max(0.0, float(value))
    if parsed > 1.0:
        return parsed / 100.0
    return parsed


def _weekday_is_forced_flat(local_now: datetime, values: list[object]) -> bool:
    names = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
    }
    for value in values:
        text = str(value).strip().lower()
        try:
            weekday = int(text)
        except ValueError:
            weekday = names.get(text, -1)
        if weekday == local_now.weekday():
            return True
    return False


def _hour_is_forced_flat(local_now: datetime, values: list[object]) -> bool:
    for value in values:
        text = str(value).strip()
        try:
            if ":" in text:
                if time.fromisoformat(text).hour == local_now.hour:
                    return True
            elif int(text) == local_now.hour:
                return True
        except ValueError:
            continue
    return False


def _quantity_units(position: Position, config: RiskConfig) -> float:
    if config.min_price_increment <= 0:
        return 0.0
    return position.lots * config.money_value_per_price_step / config.min_price_increment
