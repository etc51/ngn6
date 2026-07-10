from datetime import datetime, timedelta, timezone

from ngn6_bot.models import Candle, MarketState


def test_market_state_aggregates_1m_to_5m_and_15m():
    state = MarketState()
    start = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)

    for i in range(5):
        price = 100 + i
        state.update_candle(
            Candle(
                timestamp=start + timedelta(minutes=i),
                open=price,
                high=price + 1,
                low=price - 1,
                close=price + 0.5,
                volume=10,
                timeframe="1min",
            )
        )

    assert len(state.candles_5m) == 1
    candle_5m = state.candles_5m[-1]
    assert candle_5m.open == 100
    assert candle_5m.high == 105
    assert candle_5m.low == 99
    assert candle_5m.close == 104.5
    assert candle_5m.volume == 50


def test_market_state_upserts_older_direct_candle_without_duplicates():
    state = MarketState()
    newer = Candle(
        timestamp=datetime(2026, 1, 1, 10, 5, tzinfo=timezone.utc),
        open=105,
        high=106,
        low=104,
        close=105.5,
        volume=10,
        timeframe="5min",
    )
    older = Candle(
        timestamp=datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc),
        open=100,
        high=101,
        low=99,
        close=100.5,
        volume=10,
        timeframe="5min",
    )
    replacement = Candle(
        timestamp=older.timestamp,
        open=100,
        high=102,
        low=98,
        close=101,
        volume=20,
        timeframe="5min",
    )

    state.update_candle(newer)
    state.update_candle(older)
    state.update_candle(replacement)

    assert [candle.timestamp for candle in state.candles_5m] == [older.timestamp, newer.timestamp]
    assert state.candles_5m[0].close == 101
