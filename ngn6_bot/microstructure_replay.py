from __future__ import annotations

import json
from bisect import bisect_right
from dataclasses import dataclass, fields, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ngn6_bot.config import RuntimeConfig
from ngn6_bot.models import OrderBookFeatures, TradeFlowFeatures


@dataclass(frozen=True)
class MicrostructureSnapshot:
    timestamp: datetime
    last_price: float | None
    orderbook: OrderBookFeatures
    trade_flow: TradeFlowFeatures


class MicrostructureReplay:
    def __init__(self, snapshots: list[MicrostructureSnapshot], max_age_seconds: float):
        self.snapshots = sorted(snapshots, key=lambda item: item.timestamp)
        self.max_age_seconds = max_age_seconds
        self._timestamps = [item.timestamp for item in self.snapshots]

    @classmethod
    def empty(cls) -> MicrostructureReplay:
        return cls([], 0.0)

    @classmethod
    def from_config(cls, config: RuntimeConfig) -> MicrostructureReplay:
        path = _project_path(
            config,
            config.get("backtest", "microstructure_replay_file", default=None)
            or config.get("data_collection", "market_structure_file", default="data/market_structure.jsonl"),
        )
        max_age_seconds = float(
            config.get("backtest", "microstructure_replay_max_age_seconds", default=75.0)
        )
        return cls.from_jsonl(path, max_age_seconds=max_age_seconds)

    @classmethod
    def from_jsonl(cls, path: str | Path, *, max_age_seconds: float) -> MicrostructureReplay:
        source = Path(path)
        if not source.exists():
            return cls.empty()
        snapshots: list[MicrostructureSnapshot] = []
        with source.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                snapshot = _snapshot_from_payload(payload)
                if snapshot is not None:
                    snapshots.append(snapshot)
        return cls(snapshots, max_age_seconds=max_age_seconds)

    @property
    def available(self) -> bool:
        return bool(self.snapshots)

    def at(self, timestamp: datetime) -> MicrostructureSnapshot | None:
        if not self.snapshots:
            return None
        target = _utc(timestamp)
        index = bisect_right(self._timestamps, target) - 1
        if index < 0:
            return None
        snapshot = self.snapshots[index]
        age_seconds = (target - snapshot.timestamp).total_seconds()
        if age_seconds < 0 or age_seconds > self.max_age_seconds:
            return None
        return MicrostructureSnapshot(
            timestamp=snapshot.timestamp,
            last_price=snapshot.last_price,
            orderbook=replace(snapshot.orderbook, age_seconds=age_seconds, source="replay"),
            trade_flow=replace(snapshot.trade_flow, source="replay"),
        )


def neutral_orderbook(price: float, *, source: str = "neutral") -> OrderBookFeatures:
    return OrderBookFeatures(
        best_bid=price,
        best_ask=price,
        mid_price=price,
        spread_bps=0.0,
        bid_ask_imbalance=0.5,
        bid_depth=100.0,
        ask_depth=100.0,
        source=source,
    )


def neutral_trade_flow(*, source: str = "neutral") -> TradeFlowFeatures:
    return TradeFlowFeatures(source=source)


def _snapshot_from_payload(payload: dict[str, Any]) -> MicrostructureSnapshot | None:
    timestamp = _parse_timestamp(payload.get("timestamp"))
    if timestamp is None:
        return None
    orderbook_payload = payload.get("orderbook")
    trade_flow_payload = payload.get("trade_flow")
    if not isinstance(orderbook_payload, dict) or not isinstance(trade_flow_payload, dict):
        return None
    orderbook = _dataclass_from_payload(OrderBookFeatures, orderbook_payload, source="replay")
    trade_flow = _dataclass_from_payload(TradeFlowFeatures, trade_flow_payload, source="replay")
    if orderbook is None or trade_flow is None:
        return None
    return MicrostructureSnapshot(
        timestamp=timestamp,
        last_price=_safe_float(payload.get("last_price")),
        orderbook=orderbook,
        trade_flow=trade_flow,
    )


def _dataclass_from_payload(cls, payload: dict[str, Any], *, source: str):
    values: dict[str, Any] = {}
    for field in fields(cls):
        if field.name in payload and payload[field.name] is not None:
            values[field.name] = payload[field.name]
    if "source" in {field.name for field in fields(cls)}:
        values["source"] = source
    try:
        return cls(**values)
    except TypeError:
        return None


def _parse_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        timestamp = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return _utc(timestamp)


def _utc(timestamp: datetime) -> datetime:
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


def _safe_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _project_path(config: RuntimeConfig, value: str | Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    config_path = config.path.resolve()
    project_root = config_path.parent.parent if config_path.parent.name == "config" else config_path.parent
    return (project_root / path).resolve()
