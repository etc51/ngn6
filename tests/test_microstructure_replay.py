import json
from datetime import datetime, timedelta, timezone

from ngn6_bot.microstructure_replay import MicrostructureReplay, neutral_orderbook
from ngn6_bot.models import Side
from ngn6_bot.risk import liquidity_covers_lots


def test_microstructure_replay_returns_nearby_snapshot(tmp_path):
    timestamp = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    path = tmp_path / "market_structure.jsonl"
    path.write_text(
        json.dumps(
            {
                "timestamp": timestamp.isoformat(),
                "last_price": 100.0,
                "orderbook": {
                    "best_bid": 99.9,
                    "best_ask": 100.1,
                    "mid_price": 100.0,
                    "spread_bps": 20.0,
                    "bid_ask_imbalance": 0.7,
                    "depth_pressure": 0.4,
                },
                "trade_flow": {
                    "buy_volume": 30,
                    "sell_volume": 10,
                    "buy_ratio": 0.75,
                    "total_volume": 40,
                    "directional_volume": 40,
                    "buy_sell_imbalance": 0.5,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    replay = MicrostructureReplay.from_jsonl(path, max_age_seconds=10)
    snapshot = replay.at(timestamp + timedelta(seconds=5))

    assert snapshot is not None
    assert snapshot.orderbook.source == "replay"
    assert snapshot.orderbook.age_seconds == 5
    assert snapshot.trade_flow.buy_sell_imbalance == 0.5
    assert replay.at(timestamp + timedelta(seconds=30)) is None


def test_neutral_orderbook_does_not_block_one_lot_liquidity_check():
    book = neutral_orderbook(100.0)

    assert liquidity_covers_lots(
        Side.LONG,
        lots=1,
        bid_depth=book.bid_depth,
        ask_depth=book.ask_depth,
        min_cover=2.0,
    )
