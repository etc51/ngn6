from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Deque, Literal
from collections import deque


class Side(str, Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass(frozen=True)
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    timeframe: str


@dataclass(frozen=True)
class OrderBookLevel:
    price: float
    quantity: float


@dataclass(frozen=True)
class OrderBookSnapshot:
    timestamp: datetime
    bids: list[OrderBookLevel]
    asks: list[OrderBookLevel]


@dataclass(frozen=True)
class TradeTick:
    timestamp: datetime
    price: float
    quantity: float
    side: Literal["buy", "sell", "unknown"] = "unknown"


@dataclass(frozen=True)
class OrderBookFeatures:
    best_bid: float | None
    best_ask: float | None
    mid_price: float | None
    spread_bps: float | None
    bid_ask_imbalance: float
    bid_depth: float = 0.0
    ask_depth: float = 0.0
    depth_pressure: float = 0.0
    best_bid_qty: float = 0.0
    best_ask_qty: float = 0.0
    mid_price_change_bps: float | None = None
    spread_change_bps: float | None = None
    imbalance_change: float = 0.0
    bid_depth_change_pct: float | None = None
    ask_depth_change_pct: float | None = None
    bid_wall_price: float | None = None
    bid_wall_qty: float = 0.0
    bid_wall_notional: float = 0.0
    bid_wall_distance_bps: float | None = None
    ask_wall_price: float | None = None
    ask_wall_qty: float = 0.0
    ask_wall_notional: float = 0.0
    ask_wall_distance_bps: float | None = None
    bid_wall_absorbed: bool = False
    ask_wall_absorbed: bool = False
    age_seconds: float | None = None
    source: str = "unknown"


@dataclass(frozen=True)
class TradeFlowFeatures:
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    unknown_volume: float = 0.0
    buy_ratio: float = 0.5
    total_volume: float = 0.0
    directional_volume: float = 0.0
    signed_volume: float = 0.0
    buy_sell_imbalance: float = 0.0
    trade_count: int = 0
    buy_count: int = 0
    sell_count: int = 0
    unknown_count: int = 0
    average_trade_size: float = 0.0
    vwap: float | None = None
    last_trade_price: float | None = None
    last_trade_side: Literal["buy", "sell", "unknown"] | None = None
    source: str = "unknown"


@dataclass(frozen=True)
class Signal:
    side: Side
    confidence: float
    reason: str
    price: float
    stop_price: float | None
    timestamp: datetime
    take_profit1: float | None = None
    take_profit2: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    side: Side = Side.FLAT
    lots: int = 0
    avg_price: float = 0.0
    opened_at: datetime | None = None
    stop_price: float | None = None
    trailing_stop: float | None = None
    partial_taken: bool = False
    take_profit1: float | None = None
    take_profit2: float | None = None

    @property
    def is_open(self) -> bool:
        return self.side != Side.FLAT and self.lots > 0


@dataclass
class MarketState:
    candles_1m: Deque[Candle] = field(default_factory=lambda: deque(maxlen=300))
    candles_5m: Deque[Candle] = field(default_factory=lambda: deque(maxlen=300))
    candles_15m: Deque[Candle] = field(default_factory=lambda: deque(maxlen=300))
    order_book: OrderBookSnapshot | None = None
    previous_order_book: OrderBookSnapshot | None = None
    trades: Deque[TradeTick] = field(default_factory=lambda: deque(maxlen=1000))
    position: Position = field(default_factory=Position)
    last_stream_update: datetime | None = None

    def update_order_book(self, snapshot: OrderBookSnapshot) -> None:
        self.previous_order_book = self.order_book
        self.order_book = snapshot
        self.last_stream_update = snapshot.timestamp

    def update_candle(self, candle: Candle) -> None:
        target = {
            "1min": self.candles_1m,
            "5min": self.candles_5m,
            "15min": self.candles_15m,
        }.get(candle.timeframe)
        if target is None:
            return
        self._upsert_candle(target, candle)
        self.last_stream_update = candle.timestamp
        if candle.timeframe == "1min":
            self._aggregate_from_1m(candle, "5min", self.candles_5m, 5)
            self._aggregate_from_1m(candle, "15min", self.candles_15m, 15)

    def update_trade(self, trade: TradeTick) -> None:
        self.trades.append(trade)
        self.last_stream_update = trade.timestamp

    @staticmethod
    def _aggregate_from_1m(candle: Candle, timeframe: str, target: Deque[Candle], minutes: int) -> None:
        bucket_minute = candle.timestamp.minute - (candle.timestamp.minute % minutes)
        bucket_start = candle.timestamp.replace(minute=bucket_minute, second=0, microsecond=0)
        existing = next((item for item in target if item.timestamp == bucket_start), None)
        if existing is not None:
            MarketState._upsert_candle(
                target,
                Candle(
                timestamp=bucket_start,
                open=existing.open,
                high=max(existing.high, candle.high),
                low=min(existing.low, candle.low),
                close=candle.close,
                volume=existing.volume + candle.volume,
                timeframe=timeframe,
                ),
            )
            return
        MarketState._upsert_candle(
            target,
            Candle(
                timestamp=bucket_start,
                open=candle.open,
                high=candle.high,
                low=candle.low,
                close=candle.close,
                volume=candle.volume,
                timeframe=timeframe,
            ),
        )

    @staticmethod
    def _upsert_candle(target: Deque[Candle], candle: Candle) -> None:
        by_timestamp = {item.timestamp: item for item in target}
        by_timestamp[candle.timestamp] = candle
        ordered = [by_timestamp[key] for key in sorted(by_timestamp)]
        maxlen = target.maxlen
        target.clear()
        target.extend(ordered[-maxlen:] if maxlen is not None else ordered)
