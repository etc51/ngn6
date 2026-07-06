from __future__ import annotations

from datetime import datetime, timedelta

from ngn6_bot.models import TradeFlowFeatures, TradeTick


def analyze_trade_flow(trades: list[TradeTick], now: datetime, lookback_seconds: int) -> TradeFlowFeatures:
    cutoff = now - timedelta(seconds=lookback_seconds)
    recent = [trade for trade in trades if trade.timestamp >= cutoff]
    buy_volume = sum(trade.quantity for trade in recent if trade.side == "buy")
    sell_volume = sum(trade.quantity for trade in recent if trade.side == "sell")
    unknown_volume = sum(trade.quantity for trade in recent if trade.side == "unknown")
    total = buy_volume + sell_volume + unknown_volume
    directional_total = buy_volume + sell_volume
    buy_ratio = buy_volume / directional_total if directional_total else 0.5
    buy_count = sum(1 for trade in recent if trade.side == "buy")
    sell_count = sum(1 for trade in recent if trade.side == "sell")
    unknown_count = sum(1 for trade in recent if trade.side == "unknown")
    signed_volume = buy_volume - sell_volume
    traded_notional = sum(trade.price * trade.quantity for trade in recent)
    last_trade = max(recent, key=lambda trade: trade.timestamp) if recent else None
    return TradeFlowFeatures(
        buy_volume=buy_volume,
        sell_volume=sell_volume,
        unknown_volume=unknown_volume,
        buy_ratio=buy_ratio,
        total_volume=total,
        directional_volume=directional_total,
        signed_volume=signed_volume,
        buy_sell_imbalance=signed_volume / directional_total if directional_total else 0.0,
        trade_count=len(recent),
        buy_count=buy_count,
        sell_count=sell_count,
        unknown_count=unknown_count,
        average_trade_size=total / len(recent) if recent else 0.0,
        vwap=traded_notional / total if total else None,
        last_trade_price=last_trade.price if last_trade is not None else None,
        last_trade_side=last_trade.side if last_trade is not None else None,
        source="live",
    )
