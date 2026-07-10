from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ngn6_bot.config import RuntimeConfig
from ngn6_bot.runtime_metadata import with_commit_hash


@dataclass(frozen=True)
class PaperFeedbackSyncReport:
    completed_trades: int
    matched_entries: int
    labels_added: int
    labels_total: int


def sync_paper_trade_feedback(config: RuntimeConfig) -> PaperFeedbackSyncReport:
    events_path = _project_path(
        config,
        config.get("paper", "events_file", default="data/paper_events.jsonl"),
    )
    decisions_path = _project_path(
        config,
        config.get("data_collection", "decisions_file", default="data/decisions.jsonl"),
    )
    labels_path = _project_path(
        config,
        config.get("learning", "labels_file", default="data/labels/feedback_labels.jsonl"),
    )
    started_at = _parse_datetime(
        config.get("learning", "paper_trade_feedback_started_at", default=None)
    )
    min_net_pct = float(
        config.get("learning", "paper_trade_feedback_min_net_pct", default=0.0)
    )
    notional_multiplier = float(
        config.get("paper", "notional_multiplier", default=0.0)
    )
    if notional_multiplier <= 0:
        min_increment = float(config.get("instrument", "min_price_increment", default=0.0))
        step_value = float(
            config.get("instrument", "money_value_per_price_step", default=0.0)
        )
        notional_multiplier = step_value / min_increment if min_increment > 0 else 0.0

    trades = _completed_trades(events_path, started_at, notional_multiplier)
    decisions = (
        _entry_decisions(decisions_path, started_at)
        if any(not _trade_features(trade) for trade in trades)
        else []
    )
    existing = _read_jsonl(labels_path)
    existing_ids = {
        str(row.get("paper_trade_id"))
        for row in existing
        if row.get("paper_trade_id") is not None
    }
    added: list[dict[str, Any]] = []
    matched = 0
    for trade in trades:
        feedback_context = trade.get("feedback_context") or {}
        features = _trade_features(trade)
        decision = None
        if not features:
            decision = _nearest_entry_decision(trade, decisions)
            if decision is None:
                continue
            features = _decision_features(decision)
        matched += 1
        trade_id = str(trade["paper_trade_id"])
        if trade_id in existing_ids:
            continue
        if not features:
            continue
        side = str(trade["side"])
        net_pct = float(trade["net_pct"])
        target = side if net_pct > min_net_pct else "flat"
        label = (
            f"PAPER_TRADE_WIN_{side.upper()}"
            if target == side
            else f"PAPER_TRADE_AVOID_{side.upper()}"
        )
        added.append(
            with_commit_hash(
                {
                    "schema_version": 2,
                    "paper_trade_id": trade_id,
                    "timestamp": trade["opened_at"].isoformat(),
                    "event_end": trade["closed_at"].isoformat(),
                    "label": label,
                    "target": target,
                    "task": "entry",
                    "features": features,
                    "pnl_pct": abs(net_pct),
                    "net_pnl_pct": net_pct,
                    "net_pnl_rub": trade["net_pnl"],
                    "outcomes": {side: net_pct, "flat": 0.0},
                    "feature_complete": bool(
                        feedback_context.get(
                            "feature_complete",
                            decision.get("feature_complete", True) if decision else True,
                        )
                    ),
                    "label_matured": True,
                    "market_data_trusted": bool(
                        feedback_context.get(
                            "market_data_trusted",
                            decision.get("market_data_trusted", True) if decision else True,
                        )
                    ),
                    "source": "paper_trade_outcome",
                    "entry_decision_id": decision.get("decision_id") if decision else None,
                    "entry_reason": trade.get("entry_reason"),
                    "exit_reason": trade.get("exit_reason"),
                }
            )
        )
        existing_ids.add(trade_id)

    if added:
        labels_path.parent.mkdir(parents=True, exist_ok=True)
        with labels_path.open("a", encoding="utf-8") as file:
            for row in added:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
    return PaperFeedbackSyncReport(
        completed_trades=len(trades),
        matched_entries=matched,
        labels_added=len(added),
        labels_total=len(existing) + len(added),
    )


def _entry_decisions(path: Path, started_at: datetime | None) -> list[dict[str, Any]]:
    result = []
    for row in _read_jsonl(path):
        timestamp = _parse_datetime(row.get("timestamp"))
        if timestamp is None or started_at is not None and timestamp < started_at:
            continue
        if str(row.get("action")) != "open_accepted":
            continue
        if not bool(row.get("feature_complete", False)):
            continue
        if not bool(row.get("market_data_trusted", False)):
            continue
        row = dict(row)
        row["_timestamp"] = timestamp
        result.append(row)
    return result


def _completed_trades(
    path: Path,
    started_at: datetime | None,
    notional_multiplier: float,
) -> list[dict[str, Any]]:
    trades = []
    active: dict[str, Any] | None = None
    for row in _read_jsonl(path):
        timestamp = _parse_datetime(row.get("timestamp"))
        details = row.get("details") or {}
        if timestamp is None:
            continue
        if row.get("event") == "paper_open":
            if started_at is not None and timestamp < started_at:
                active = None
                continue
            lots = max(1, int(details.get("lots", 1) or 1))
            price = float(details.get("price", 0.0) or 0.0)
            active = {
                "opened_at": timestamp,
                "side": str(details.get("side") or "flat"),
                "lots": lots,
                "entry_price": price,
                "entry_reason": str(details.get("reason") or "unknown"),
                "feedback_context": details.get("feedback_context") or {},
                "net_pnl": -float(details.get("commission", 0.0) or 0.0),
            }
            continue
        if row.get("event") != "paper_close" or active is None:
            continue
        active["net_pnl"] += float(details.get("realized_pnl", 0.0) or 0.0)
        if int(details.get("remaining_lots", 0) or 0) > 0:
            continue
        exposure = (
            abs(active["entry_price"] * active["lots"] * notional_multiplier)
            if notional_multiplier > 0
            else 0.0
        )
        net_pct = active["net_pnl"] / exposure * 100 if exposure > 0 else 0.0
        trades.append(
            {
                **active,
                "closed_at": timestamp,
                "exit_reason": str(details.get("reason") or "unknown"),
                "net_pct": net_pct,
                "paper_trade_id": (
                    f"{active['opened_at'].isoformat()}:{timestamp.isoformat()}:"
                    f"{active['side']}"
                ),
            }
        )
        active = None
    return trades


def _nearest_entry_decision(
    trade: dict[str, Any], decisions: list[dict[str, Any]]
) -> dict[str, Any] | None:
    side = str(trade["side"])
    opened_at = trade["opened_at"]
    candidates = [
        row
        for row in decisions
        if str(row.get("side")) == side
        and abs((row["_timestamp"] - opened_at).total_seconds()) <= 15
    ]
    return min(
        candidates,
        key=lambda row: abs((row["_timestamp"] - opened_at).total_seconds()),
        default=None,
    )


def _decision_features(row: dict[str, Any]) -> dict[str, float]:
    candidates = [
        (row.get("metadata") or {}).get("features"),
        (row.get("market_context") or {}).get("features"),
    ]
    for candidate in candidates:
        if isinstance(candidate, dict) and candidate:
            return {
                str(key): float(value)
                for key, value in candidate.items()
                if isinstance(value, (int, float))
            }
    return {}


def _trade_features(trade: dict[str, Any]) -> dict[str, float]:
    candidate = (trade.get("feedback_context") or {}).get("features")
    if not isinstance(candidate, dict):
        return {}
    return {
        str(key): float(value)
        for key, value in candidate.items()
        if isinstance(value, (int, float))
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    rows = []
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _project_path(config: RuntimeConfig, value: str | Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    config_path = config.path.resolve()
    project_root = (
        config_path.parent.parent if config_path.parent.name == "config" else config_path.parent
    )
    return (project_root / path).resolve()


def report_payload(report: PaperFeedbackSyncReport) -> dict[str, int]:
    return asdict(report)
