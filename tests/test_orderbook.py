from datetime import datetime, timedelta, timezone

from ngn6_bot.models import OrderBookLevel, OrderBookSnapshot
from ngn6_bot.orderbook import analyze_order_book, spread_is_acceptable


def test_orderbook_spread_and_imbalance():
    now = datetime.now(timezone.utc)
    book = OrderBookSnapshot(
        timestamp=now,
        bids=[OrderBookLevel(100, 10), OrderBookLevel(99, 10)],
        asks=[OrderBookLevel(101, 5), OrderBookLevel(102, 5)],
    )

    features = analyze_order_book(
        book,
        None,
        levels=2,
        wall_multiplier=3,
        absorption_drop_pct=35,
        min_wall_notional=1,
    )

    assert features.spread_bps is not None
    assert features.bid_ask_imbalance > 0.5
    assert not spread_is_acceptable(features, max_spread_bps=1)


def test_orderbook_tracks_age_and_depth_changes():
    now = datetime.now(timezone.utc)
    previous = OrderBookSnapshot(
        timestamp=now,
        bids=[OrderBookLevel(99, 4), OrderBookLevel(98, 4)],
        asks=[OrderBookLevel(101, 8), OrderBookLevel(102, 8)],
    )
    current = OrderBookSnapshot(
        timestamp=now,
        bids=[OrderBookLevel(100, 12), OrderBookLevel(99, 8)],
        asks=[OrderBookLevel(101, 5), OrderBookLevel(102, 5)],
    )

    features = analyze_order_book(
        current,
        previous,
        levels=2,
        wall_multiplier=2,
        absorption_drop_pct=35,
        min_wall_notional=1,
        now=now + timedelta(seconds=3),
    )

    assert features.depth_pressure > 0
    assert features.age_seconds == 3
    assert features.mid_price_change_bps is not None
    assert features.bid_depth_change_pct is not None
