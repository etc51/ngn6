from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from ngn6_bot.config import RuntimeConfig
from ngn6_bot.learning.feedback_model import FEATURE_KEYS
from ngn6_bot.runtime_metadata import add_commit_hash


def audit_feature_completeness(
    config: RuntimeConfig,
    *,
    decisions_path: str | Path | None = None,
    market_path: str | Path | None = None,
) -> dict[str, Any]:
    decision_file = _project_path(
        config,
        decisions_path
        or config.get("data_collection", "decisions_file", default="data/decisions.jsonl"),
    )
    market_file = _project_path(
        config,
        market_path
        or config.get("data_collection", "market_structure_file", default="data/market_structure.jsonl"),
    )
    decision_rows = _read_jsonl(decision_file)
    market_rows = _read_jsonl(market_file)

    missing_fields: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    complete = 0
    trusted = 0
    trainable = 0
    label_matured = 0

    for row in decision_rows:
        features = _features(row)
        row_missing = [key for key in FEATURE_KEYS if key not in features]
        feature_complete = bool(row.get("feature_complete", False)) and not row_missing
        market_data_trusted = bool(row.get("market_data_trusted", False))
        matured = bool(row.get("label_matured", False))
        if feature_complete:
            complete += 1
        else:
            missing_fields.update(row_missing or FEATURE_KEYS)
            reasons[_missing_reason(row, features, row_missing)] += 1
        if market_data_trusted:
            trusted += 1
        if matured:
            label_matured += 1
        if feature_complete and matured and market_data_trusted:
            trainable += 1

    market_trusted = sum(1 for row in market_rows if bool(row.get("market_data_trusted", False)))
    report = {
        "schema_version": 1,
        "decisions_path": str(decision_file),
        "market_path": str(market_file),
        "decisions": len(decision_rows),
        "feature_complete_records": complete,
        "market_data_trusted_records": trusted,
        "label_matured_records": label_matured,
        "trainable_decision_records": trainable,
        "missing_feature_records": len(decision_rows) - complete,
        "missing_reasons": dict(reasons.most_common()),
        "top_missing_fields": dict(missing_fields.most_common(25)),
        "market_structure_records": len(market_rows),
        "market_structure_trusted_records": market_trusted,
    }
    add_commit_hash(report)
    return report


def save_feature_completeness_report(report: dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def _features(row: dict[str, Any]) -> dict[str, Any]:
    for candidate in [
        row.get("metadata", {}).get("features") if isinstance(row.get("metadata"), dict) else None,
        row.get("details", {}).get("metadata", {}).get("features")
        if isinstance(row.get("details"), dict)
        and isinstance(row.get("details", {}).get("metadata"), dict)
        else None,
        row.get("market_context", {}).get("features")
        if isinstance(row.get("market_context"), dict)
        else None,
    ]:
        if isinstance(candidate, dict):
            return candidate
    return {}


def _missing_reason(row: dict[str, Any], features: dict[str, Any], missing: list[str]) -> str:
    reason = str(row.get("reason") or row.get("reject_reason") or "").lower()
    feature_reason = str(row.get("feature_reject_reason") or "").lower()
    market_reason = str(row.get("market_data_trust_reason") or "").lower()
    market_context = row.get("market_context") if isinstance(row.get("market_context"), dict) else {}
    if int(row.get("schema_version") or 0) < 2:
        return "old_schema"
    if "warmup" in reason or "not_enough" in reason:
        return "warmup"
    if "stale" in reason or "stale" in market_reason:
        return "stale_market_data"
    if not market_context:
        return "no_market_context"
    if not isinstance(market_context.get("orderbook"), dict):
        return "no_orderbook"
    if feature_reason == "no_closed_candles":
        return "no_closed_candles"
    if feature_reason.startswith("feature_builder_error"):
        return "feature_builder_error"
    if features and missing:
        return "partial_feature_schema"
    if "features" in market_context and not isinstance(market_context.get("features"), dict):
        return "serialization_bug"
    return "missing_features"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _project_path(config: RuntimeConfig, value: str | Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    config_path = config.path.resolve()
    project_root = config_path.parent.parent if config_path.parent.name == "config" else config_path.parent
    return (project_root / path).resolve()
