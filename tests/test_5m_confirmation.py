from datetime import datetime, timezone
import pandas as pd

from ngn6_bot.bot import TradingBot
from ngn6_bot.models import Side, Signal


class DummyConfig:
    def get(self, *keys, default=None):
        if keys == ("signals", "confirm_5m_for_entry"):
            return True
        return default


def _bot():
    bot = TradingBot.__new__(TradingBot)
    bot.config = DummyConfig()
    return bot


def _signal(side):
    return Signal(side, 0.7, "test_entry", 100.0, 99.0, datetime.now(timezone.utc))


def test_5m_opposite_confirmation_blocks_entry():
    frame = pd.DataFrame(
        [
            {"close": 99.0, "ema_fast": 99.2, "ema_slow": 100.0, "macd_hist": -0.1, "rsi": 42.0},
            {"close": 98.8, "ema_fast": 99.0, "ema_slow": 100.0, "macd_hist": -0.2, "rsi": 40.0},
            {"close": 98.5, "ema_fast": 98.9, "ema_slow": 100.0, "macd_hist": -0.3, "rsi": 38.0},
        ]
    )

    signal, allowed, reason = TradingBot._apply_5m_entry_confirmation(_bot(), _signal(Side.LONG), frame)

    assert not allowed
    assert reason == "blocked_by_5m_confirmation:short"
    assert signal.metadata["confirmation_5m"]["state"] == "confirm_short"


def test_5m_neutral_confirmation_allows_entry():
    frame = pd.DataFrame(
        [
            {"close": 100.0, "ema_fast": 100.0, "ema_slow": 100.0, "macd_hist": 0.0, "rsi": 50.0},
            {"close": 100.0, "ema_fast": 100.0, "ema_slow": 100.0, "macd_hist": 0.0, "rsi": 50.0},
            {"close": 100.0, "ema_fast": 100.0, "ema_slow": 100.0, "macd_hist": 0.0, "rsi": 50.0},
        ]
    )

    signal, allowed, reason = TradingBot._apply_5m_entry_confirmation(_bot(), _signal(Side.SHORT), frame)

    assert allowed
    assert reason == "5m_confirmation:neutral"
    assert signal.metadata["confirmation_5m"]["state"] == "neutral"
