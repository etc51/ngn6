from datetime import datetime, timedelta, timezone

from ngn6_bot.models import TradeTick
from ngn6_bot.tradeflow import analyze_trade_flow


def test_trade_flow_buy_ratio_uses_recent_trades():
    now = datetime.now(timezone.utc)
    trades = [
        TradeTick(now - timedelta(seconds=10), 100, 8, "buy"),
        TradeTick(now - timedelta(seconds=5), 101, 2, "sell"),
        TradeTick(now - timedelta(seconds=90), 99, 100, "sell"),
    ]

    flow = analyze_trade_flow(trades, now, lookback_seconds=30)

    assert flow.buy_ratio == 0.8
    assert flow.total_volume == 10
