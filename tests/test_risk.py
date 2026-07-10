from datetime import datetime, timezone

from ngn6_bot.models import Position, Side, Signal
from ngn6_bot.risk import (
    RiskConfig,
    calculate_position_lots,
    move_stop_to_breakeven,
    must_flatten_before_clearing,
    trading_session_block_reason,
    update_trailing_stop,
)


def _risk_config(**overrides) -> RiskConfig:
    values = {
        "deposit_value": 100000,
        "risk_per_trade_pct": 1,
        "max_risk_per_trade_pct": 2,
        "max_position_lots": 10,
        "min_position_lots": 1,
        "stop_buffer_ticks": 2,
        "min_price_increment": 0.01,
        "money_value_per_price_step": 1,
        "partial_take_profit_pct": 1.2,
        "partial_take_fraction": 0.5,
        "trailing_stop_pct": 0.8,
        "close_before_clearing_minutes": 5,
        "clearings": ["14:00"],
        "timezone": "Europe/Moscow",
    }
    values.update(overrides)
    return RiskConfig(**values)


def test_position_size_is_capped():
    signal = Signal(Side.LONG, 0.8, "test", 100, 99, datetime.now(timezone.utc))
    assert calculate_position_lots(signal, _risk_config()) == 10


def test_pre_clearing_window_detected():
    now = datetime(2026, 1, 1, 10, 57, tzinfo=timezone.utc)
    assert must_flatten_before_clearing(now, _risk_config())


def test_trading_session_blocks_outside_moscow_window():
    before_open = datetime(2026, 1, 1, 6, 59, tzinfo=timezone.utc)
    after_open = datetime(2026, 1, 1, 7, 0, tzinfo=timezone.utc)

    assert trading_session_block_reason(
        before_open,
        timezone="Europe/Moscow",
        trading_start="10:00",
        trading_end="23:45",
    ) == "outside_trading_session"
    assert trading_session_block_reason(
        after_open,
        timezone="Europe/Moscow",
        trading_start="10:00",
        trading_end="23:45",
    ) is None


def test_trading_session_supports_overnight_and_forced_weekday():
    now = datetime(2026, 1, 2, 20, 0, tzinfo=timezone.utc)

    assert trading_session_block_reason(
        now,
        timezone="UTC",
        trading_start="19:00",
        trading_end="03:00",
    ) is None
    assert trading_session_block_reason(
        now,
        timezone="UTC",
        trading_start="19:00",
        trading_end="03:00",
        forced_flat_weekdays=["friday"],
    ) == "forced_flat_weekday"


def test_position_is_open_property():
    position = Position(side=Side.LONG, lots=1, avg_price=100)
    assert position.is_open


def test_breakeven_uses_open_profit_pct():
    position = Position(side=Side.LONG, lots=1, avg_price=100, stop_price=98, trailing_stop=98)
    moved = move_stop_to_breakeven(position, 100.75, _risk_config(breakeven_trigger_pct=0.75))

    assert moved
    assert position.stop_price == 100
    assert position.trailing_stop == 100


def test_rub_trailing_locks_configured_open_profit_share():
    position = Position(side=Side.LONG, lots=10, avg_price=100, stop_price=98, trailing_stop=98)
    config = _risk_config(
        trailing_profit_trigger_rub=1200,
        trailing_profit_lock_ratio=0.35,
        trailing_stop_pct=0,
    )

    update_trailing_stop(position, 102, config)

    assert round(position.stop_price, 6) == 100.7
    assert round(position.trailing_stop, 6) == 100.7
