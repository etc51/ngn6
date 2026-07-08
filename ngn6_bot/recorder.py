from __future__ import annotations

import json
import hashlib
import uuid
from collections import deque
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from ngn6_bot.config import RuntimeConfig
from ngn6_bot.learning.feedback_model import FEATURE_SCHEMA_VERSION
from ngn6_bot.models import Position, Signal
from ngn6_bot.runtime_metadata import add_commit_hash


class StrategyRecorder:
    def __init__(
        self,
        enabled: bool,
        market_path: Path,
        decisions_path: Path,
        *,
        record_raw_microstructure: bool = True,
        record_decision_context: bool = True,
    ):
        self.enabled = enabled
        self.market_path = market_path
        self.decisions_path = decisions_path
        self.record_raw_microstructure = record_raw_microstructure
        self.record_decision_context = record_decision_context
        if enabled:
            self.market_path.parent.mkdir(parents=True, exist_ok=True)
            self.decisions_path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_runtime_config(cls, config: RuntimeConfig) -> StrategyRecorder:
        return cls(
            enabled=bool(config.get("data_collection", "enabled", default=True)),
            market_path=Path(
                config.get("data_collection", "market_structure_file", default="data/market_structure.jsonl")
            ),
            decisions_path=Path(
                config.get("data_collection", "decisions_file", default="data/decisions.jsonl")
            ),
            record_raw_microstructure=bool(
                config.get("data_collection", "record_raw_microstructure", default=True)
            ),
            record_decision_context=bool(
                config.get("data_collection", "record_decision_context", default=True)
            ),
        )

    @classmethod
    def disabled(cls) -> StrategyRecorder:
        return cls(False, Path("data/market_structure.jsonl"), Path("data/decisions.jsonl"))

    def record_market(
        self,
        *,
        timestamp: datetime,
        ticker: str,
        last_price: float,
        orderbook: Any,
        trade_flow: Any,
        candle_counts: dict[str, int],
        position: Position,
        orderbook_snapshot: Any | None = None,
        recent_trades: list[Any] | None = None,
        market_data_trusted: bool | None = None,
        market_data_trust_reason: str | None = None,
        market_data_trust_details: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return
        payload = {
            "schema_version": 2,
            "timestamp": timestamp,
            "ticker": ticker,
            "last_price": last_price,
            "orderbook": _to_jsonable(orderbook),
            "trade_flow": _to_jsonable(trade_flow),
            "candle_counts": candle_counts,
            "position": _position_payload(position),
            "market_data_trusted": market_data_trusted,
            "market_data_trust_reason": market_data_trust_reason,
            "market_data_trust_details": _to_jsonable(market_data_trust_details or {}),
        }
        if self.record_raw_microstructure:
            payload["orderbook_snapshot"] = _to_jsonable(orderbook_snapshot)
            payload["recent_trades"] = _to_jsonable(recent_trades or [])
        append_jsonl(
            self.market_path,
            payload,
        )

    def record_decision(
        self,
        *,
        timestamp: datetime,
        action: str,
        reason: str,
        signal: Signal | None = None,
        details: dict[str, Any] | None = None,
        position: Position | None = None,
        market_context: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return
        payload: dict[str, Any] = {
            "schema_version": 2,
            "decision_id": f"{timestamp.strftime('%Y%m%dT%H%M%S%f')}-{uuid.uuid4().hex[:8]}",
            "timestamp": timestamp,
            "action": action,
            "reason": reason,
            "final_action": action,
            "reject_reason": reason if action in {"skip", "open_rejected", "close_rejected"} else None,
            "details": details or {},
            "label_matured": bool((details or {}).get("label_matured", False)),
        }
        if signal is not None:
            payload.update(
                {
                    "side": signal.side.value,
                    "confidence": signal.confidence,
                    "price": signal.price,
                    "stop_price": signal.stop_price,
                    "take_profit1": signal.take_profit1,
                    "take_profit2": signal.take_profit2,
                    "metadata": signal.metadata,
                    "signal_reason": signal.reason,
                }
            )
        if position is not None:
            payload["position"] = _position_payload(position)
            payload["position_state"] = position.side.value
        if self.record_decision_context and market_context is not None:
            payload["market_context"] = _to_jsonable(market_context)
        payload.update(_decision_flat_fields(signal, details or {}, position, market_context))
        append_jsonl(self.decisions_path, payload)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    add_commit_hash(payload)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False, default=_json_default) + "\n")


def read_jsonl_tail(path: str | Path, limit: int) -> list[dict[str, Any]]:
    target = Path(path)
    if limit <= 0 or not target.exists():
        return []
    rows: deque[dict[str, Any]] = deque(maxlen=limit)
    with target.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return list(rows)


def _position_payload(position: Position) -> dict[str, Any]:
    return {
        "side": position.side.value,
        "lots": position.lots,
        "avg_price": position.avg_price,
        "stop_price": position.stop_price,
        "trailing_stop": position.trailing_stop,
        "partial_taken": position.partial_taken,
        "take_profit1": position.take_profit1,
        "take_profit2": position.take_profit2,
    }


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    return value


def _decision_flat_fields(
    signal: Signal | None,
    details: dict[str, Any],
    position: Position | None,
    market_context: dict[str, Any] | None,
) -> dict[str, Any]:
    metadata = signal.metadata if signal is not None else details.get("metadata", {})
    feedback = metadata.get("feedback", {}) if isinstance(metadata, dict) else {}
    model_validation = metadata.get("model_validation", {}) if isinstance(metadata, dict) else {}
    orderbook = (market_context or {}).get("orderbook") if market_context else None
    features = None
    if isinstance(metadata, dict):
        features = metadata.get("features")
    if features is None and market_context:
        features = market_context.get("features")
    feature_complete = (
        bool(market_context.get("feature_complete"))
        if isinstance(market_context, dict)
        else isinstance(features, dict)
    )
    market_data_trusted = (
        bool(market_context.get("market_data_trusted"))
        if isinstance(market_context, dict)
        else None
    )
    flat = {
        "position_state": position.side.value if position is not None else None,
        "base_signal": signal.reason if signal is not None else None,
        "base_signal_strength": signal.confidence if signal is not None else None,
        "ml_model_version": model_validation.get("schema_version") if model_validation else None,
        "ml_status": model_validation.get("reason") if model_validation else None,
        "ml_expected_R": metadata.get("expected_R") if isinstance(metadata, dict) else None,
        "risk_gate_result": details.get("risk_gate_result"),
        "features_hash": _features_hash(features),
        "feature_complete": feature_complete,
        "feature_timestamp": market_context.get("feature_timestamp") if market_context else None,
        "missing_feature_fields": market_context.get("missing_feature_fields") if market_context else None,
        "feature_reject_reason": market_context.get("feature_reject_reason") if market_context else None,
        "market_data_trusted": market_data_trusted,
        "market_data_trust_reason": market_context.get("market_data_trust_reason")
        if market_context
        else None,
        "feature_schema_version": model_validation.get("feature_schema_version")
        if model_validation
        else FEATURE_SCHEMA_VERSION if feature_complete else None,
    }
    if isinstance(feedback, dict):
        probabilities = {item.get("target"): item.get("score") for item in feedback.get("alternatives", [])}
        flat.update(
            {
                "ml_proba_long": probabilities.get("long"),
                "ml_proba_short": probabilities.get("short"),
                "ml_proba_flat": probabilities.get("flat"),
            }
        )
    if orderbook is not None:
        flat.update(
            {
                "orderbook_age_sec": getattr(orderbook, "age_seconds", None),
                "spread_bps": getattr(orderbook, "spread_bps", None),
                "liquidity_cover": None,
                "book_pressure": getattr(orderbook, "depth_pressure", None),
            }
        )
    return flat


def _features_hash(features: Any) -> str | None:
    if not isinstance(features, dict):
        return None
    payload = json.dumps(_to_jsonable(features), sort_keys=True, default=_json_default)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _json_default(value: Any) -> Any:
    return _to_jsonable(value)
