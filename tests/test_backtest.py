from datetime import datetime, timedelta, timezone

from ngn6_bot.backtest import _blocked_by_reentry_cooldown, run_replay_backtest
from ngn6_bot.config import load_config
from ngn6_bot.models import Candle, Side


def test_replay_backtest_returns_report():
    config = load_config("config/ngn6.yaml")
    start = datetime(2026, 1, 1, 7, 0, tzinfo=timezone.utc)
    candles = [
        Candle(
            timestamp=start + timedelta(minutes=i),
            open=100 + i * 0.02,
            high=100.2 + i * 0.02,
            low=99.8 + i * 0.02,
            close=100.1 + i * 0.02,
            volume=100 + i,
            timeframe="1min",
        )
        for i in range(80)
    ]

    report = run_replay_backtest(config, candles, "TESTFIGI")

    assert report.candles == 80
    assert report.figi == "TESTFIGI"
    assert report.limitations


def test_take_profit_reentry_cooldown_blocks_same_side_only():
    now = datetime(2026, 1, 1, 8, 30, tzinfo=timezone.utc)
    last_exit = (Side.LONG, now - timedelta(minutes=10), "take_profit")

    assert _blocked_by_reentry_cooldown(Side.LONG, now, last_exit, 30)
    assert not _blocked_by_reentry_cooldown(Side.SHORT, now, last_exit, 30)
    assert not _blocked_by_reentry_cooldown(
        Side.LONG,
        now,
        (Side.LONG, now - timedelta(minutes=31), "take_profit"),
        30,
    )
    assert not _blocked_by_reentry_cooldown(
        Side.LONG,
        now,
        (Side.LONG, now - timedelta(minutes=10), "hard_stop"),
        30,
    )


def test_reentry_cooldown_after_any_exit_blocks_both_sides():
    now = datetime(2026, 1, 1, 8, 30, tzinfo=timezone.utc)
    last_exit = (Side.LONG, now - timedelta(minutes=5), "signal-flip", 0.2)

    assert _blocked_by_reentry_cooldown(
        Side.LONG,
        now,
        last_exit,
        0,
        exit_cooldown_minutes=10,
    )
    assert _blocked_by_reentry_cooldown(
        Side.SHORT,
        now,
        last_exit,
        0,
        exit_cooldown_minutes=10,
    )
    assert not _blocked_by_reentry_cooldown(
        Side.SHORT,
        now,
        (Side.LONG, now - timedelta(minutes=11), "signal-flip", 0.2),
        0,
        exit_cooldown_minutes=10,
    )


def test_reentry_cooldown_after_loss_blocks_even_without_exit_cooldown():
    now = datetime(2026, 1, 1, 8, 30, tzinfo=timezone.utc)
    last_loss = (Side.SHORT, now - timedelta(minutes=20), "hard_stop_hit", -0.15)

    assert _blocked_by_reentry_cooldown(
        Side.LONG,
        now,
        last_loss,
        0,
        loss_cooldown_minutes=45,
    )
    assert not _blocked_by_reentry_cooldown(
        Side.LONG,
        now,
        (Side.SHORT, now - timedelta(minutes=20), "hard_stop_hit", 0.15),
        0,
        loss_cooldown_minutes=45,
    )
