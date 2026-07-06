from datetime import datetime, timezone

import pandas as pd

from ngn6_bot.bot import TradingBot
from ngn6_bot.models import OrderBookFeatures, Side, Signal


class DummyConfig:
    def get(self, *keys, default=None):
        return default


def _bot():
    bot = TradingBot.__new__(TradingBot)
    bot.config = DummyConfig()
    return bot


def _signal():
    return Signal(Side.SHORT, 0.7, "test_short", 100.0, 101.0, datetime.now(timezone.utc))


def _orderbook(imbalance=0.5, pressure=0.0):
    return OrderBookFeatures(
        best_bid=99.99,
        best_ask=100.01,
        mid_price=100.0,
        spread_bps=2.0,
        bid_ask_imbalance=imbalance,
        depth_pressure=pressure,
    )


def _context_frame(closes, lows, rsi=35.0, macd_hist=-0.001):
    rows = [
        {"close": close, "low": low, "rsi": rsi, "macd_hist": macd_hist}
        for close, low in zip(closes, lows)
    ]
    return pd.DataFrame(rows)


def test_short_exhaustion_guard_blocks_bid_support_near_low():
    frame = _context_frame(
        closes=[101.0, 100.8, 100.5, 100.3, 100.2, 100.1, 100.0],
        lows=[100.8, 100.6, 100.4, 100.2, 100.1, 100.0, 99.95],
    )

    signal, allowed, reason = TradingBot._apply_short_exhaustion_guard(
        _bot(),
        _signal(),
        frame,
        _orderbook(imbalance=0.58, pressure=0.16),
    )

    assert not allowed
    assert reason == "blocked_by_short_exhaustion:bid_support"
    assert signal.metadata["short_exhaustion"]["state"] == "microstructure_bid_support"


def test_short_exhaustion_guard_blocks_candle_capitulation_low():
    frame = _context_frame(
        closes=[102.2, 101.9, 101.4, 101.0, 100.6, 100.3, 100.0],
        lows=[102.0, 101.7, 101.2, 100.8, 100.4, 100.1, 99.95],
        rsi=28.0,
        macd_hist=-0.006,
    )

    signal, allowed, reason = TradingBot._apply_short_exhaustion_guard(
        _bot(),
        _signal(),
        frame,
        _orderbook(),
    )

    assert not allowed
    assert reason == "blocked_by_short_exhaustion:capitulation_low"
    assert signal.metadata["short_exhaustion"]["state"] == "capitulation_low"


def test_short_exhaustion_guard_allows_early_short_continuation():
    frame = _context_frame(
        closes=[100.8, 100.7, 100.5, 100.4, 100.3, 100.1, 100.0],
        lows=[100.7, 100.6, 100.4, 100.3, 100.2, 100.0, 99.95],
        rsi=28.0,
        macd_hist=-0.003,
    )

    _, allowed, reason = TradingBot._apply_short_exhaustion_guard(
        _bot(),
        _signal(),
        frame,
        _orderbook(),
    )

    assert allowed
    assert reason == "short_exhaustion_guard:clear"
