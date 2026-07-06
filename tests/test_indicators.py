from datetime import datetime, timedelta, timezone

from ngn6_bot.indicators import add_indicators, candles_to_frame
from ngn6_bot.models import Candle


def test_add_indicators_creates_expected_columns():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    candles = [
        Candle(start + timedelta(minutes=i), 100 + i, 101 + i, 99 + i, 100 + i, 10 + i, "1min")
        for i in range(30)
    ]

    df = add_indicators(candles_to_frame(candles))

    for column in [
        "ema_fast",
        "ema_slow",
        "rsi",
        "atr",
        "adx",
        "macd",
        "macd_signal",
        "macd_hist",
        "bb_upper",
        "bb_lower",
        "bb_width_pct",
    ]:
        assert column in df.columns
    assert df["ema_fast"].iloc[-1] > df["ema_slow"].iloc[-1]
    assert 0 <= df["rsi"].iloc[-1] <= 100
