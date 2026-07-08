from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

from ngn6_bot.config import RuntimeConfig
from ngn6_bot.learning.ensemble import FeedbackEnsemble
from ngn6_bot.learning.feature_audit import audit_feature_completeness
from ngn6_bot.learning.regime_report import (
    _label_for_timestamp,
    _load_oracle_labels,
    _project_path,
    _round_trip_cost_pct,
    generate_regime_report,
)
from ngn6_bot.runtime_metadata import add_commit_hash


DEFAULT_THRESHOLDS = (0.62, 0.66, 0.68, 0.72)


def generate_model_diagnostics(
    config: RuntimeConfig,
    *,
    model_path: str | Path | None = None,
    decisions_path: str | Path | None = None,
    labels_dir: str | Path | None = None,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
    top_features: int = 20,
) -> dict[str, Any]:
    decision_file = _project_path(
        config,
        decisions_path
        or config.get("data_collection", "decisions_file", default="data/decisions.jsonl"),
    )
    label_root = _project_path(
        config,
        labels_dir or config.get("daily_oracle", "output_dir", default="reports/daily_oracle"),
    )
    model_file = _project_path(
        config,
        model_path
        or config.get("learning", "candidate_model_path", default="data/models/feedback_ensemble.candidate.joblib"),
    )

    decisions = _read_jsonl(decision_file)
    predictions = _candidate_predictions(decisions)
    labels = _load_oracle_labels(label_root, ZoneInfo(config.timezone))
    as_of = datetime.now(timezone.utc)
    matured_labels = [label for label in labels if label.end <= as_of]
    feature_audit = audit_feature_completeness(
        config,
        decisions_path=decision_file,
        market_path=config.get("data_collection", "market_structure_file", default="data/market_structure.jsonl"),
    )
    model = _load_model(model_file)
    regime_report = generate_regime_report(
        config,
        decisions_path=decision_file,
        labels_dir=label_root,
    )

    latest = predictions[-1] if predictions else None
    report = {
        "schema_version": 1,
        "generated_at": as_of.isoformat(),
        "model_path": str(model_file),
        "decisions_path": str(decision_file),
        "labels_dir": str(label_root),
        "model": _model_summary(model),
        "label_distribution": _label_distribution(model),
        "labels": {
            "oracle_intervals": len(labels),
            "mature_oracle_intervals": len(matured_labels),
            "mature_prediction_rows": _mature_prediction_rows(predictions, labels, as_of),
        },
        "runtime_rows": {
            "decisions": len(decisions),
            "candidate_predictions": len(predictions),
        },
        "rejected": {
            "missing_features": feature_audit["missing_feature_records"],
            "stale_orderbook": _stale_orderbook_rows(decisions),
            "missing_reasons": feature_audit["missing_reasons"],
        },
        "threshold_replay": _threshold_replay(
            predictions,
            labels,
            thresholds,
            as_of,
            _round_trip_cost_pct(config),
        ),
        "current_candidate": _current_candidate(latest, thresholds),
        "top_flat_features": _top_flat_features(model, latest, predictions, top_features),
        "regime_expectancy": _regime_expectancy(regime_report),
    }
    add_commit_hash(report)
    return report


def save_model_diagnostics(report: dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def _load_model(path: Path) -> FeedbackEnsemble | None:
    if not path.exists():
        return None
    try:
        return FeedbackEnsemble.load(path)
    except Exception:
        return None


def _model_summary(model: FeedbackEnsemble | None) -> dict[str, Any]:
    if model is None:
        return {"loaded": False}
    return {
        "loaded": True,
        "schema_version": model.schema_version,
        "feature_schema_version": model.feature_schema_version,
        "examples_total_heads": model.examples,
        "entry_examples": int((model.heads.get("entry") or {}).get("examples", 0)),
        "exit_examples": int((model.heads.get("exit") or {}).get("examples", 0)),
        "trained_at": model.trained_at,
        "model_status": model.model_status,
        "promotion_status": model.promotion_status,
        "promotion_score": model.promotion_score,
        "heads": {
            name: {
                "examples": int(head.get("examples", 0)),
                "classes": list(head.get("classes") or []),
                "class_counts": _head_class_counts(model, name, head),
            }
            for name, head in sorted(model.heads.items())
        },
    }


def _label_distribution(model: FeedbackEnsemble | None) -> dict[str, Any]:
    if model is None:
        return {}
    entry_head = model.heads.get("entry") or {}
    exit_head = model.heads.get("exit") or {}
    entry = _head_class_counts(model, "entry", entry_head)
    exit_counts = _head_class_counts(model, "exit", exit_head)
    return {
        "long": int(entry.get("long", 0)),
        "short": int(entry.get("short", 0)),
        "flat": int(entry.get("flat", 0)),
        "exit": int(exit_counts.get("exit", 0)),
        "hold": int(exit_counts.get("hold", 0)),
        "by_head": {
            name: _head_class_counts(model, name, head)
            for name, head in sorted(model.heads.items())
        },
    }


def _head_class_counts(
    model: FeedbackEnsemble,
    name: str,
    head: dict[str, Any],
) -> dict[str, int]:
    counts = head.get("class_counts") or {}
    if not counts:
        counts = (model.task_reports.get(name) or {}).get("class_counts") or {}
    return {str(key): int(value) for key, value in dict(counts).items()}


def _candidate_predictions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for row in rows:
        shadow = _candidate_shadow(row)
        timestamp = _parse_timestamp(row.get("timestamp"))
        if timestamp is None or not isinstance(shadow, dict):
            continue
        probabilities = _probabilities(shadow)
        predictions.append(
            {
                "timestamp": timestamp,
                "target": str(shadow.get("target") or "unknown").lower(),
                "score": _safe_float(shadow.get("score")),
                "probabilities": probabilities,
                "shadow": shadow,
                "row": row,
                "features": _features(row),
            }
        )
    return predictions


def _candidate_shadow(row: dict[str, Any]) -> dict[str, Any] | None:
    for candidate in (
        row.get("metadata", {}).get("candidate_shadow")
        if isinstance(row.get("metadata"), dict)
        else None,
        row.get("details", {}).get("metadata", {}).get("candidate_shadow")
        if isinstance(row.get("details"), dict)
        and isinstance(row.get("details", {}).get("metadata"), dict)
        else None,
    ):
        if isinstance(candidate, dict):
            return candidate
    return None


def _probabilities(shadow: dict[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for item in shadow.get("alternatives") or []:
        if isinstance(item, dict):
            target = str(item.get("target") or "").lower()
            if target:
                values[target] = _safe_float(item.get("score"))
    target = str(shadow.get("target") or "").lower()
    if target and target not in values:
        values[target] = _safe_float(shadow.get("score"))
    return values


def _features(row: dict[str, Any]) -> dict[str, float]:
    for candidate in (
        row.get("market_context", {}).get("features")
        if isinstance(row.get("market_context"), dict)
        else None,
        row.get("details", {}).get("metadata", {}).get("features")
        if isinstance(row.get("details"), dict)
        and isinstance(row.get("details", {}).get("metadata"), dict)
        else None,
        row.get("metadata", {}).get("features") if isinstance(row.get("metadata"), dict) else None,
    ):
        if isinstance(candidate, dict):
            return {str(key): _safe_float(value) for key, value in candidate.items()}
    return {}


def _mature_prediction_rows(
    predictions: list[dict[str, Any]],
    labels: list[Any],
    as_of: datetime,
) -> int:
    total = 0
    for prediction in predictions:
        label = _label_for_timestamp(prediction["timestamp"], labels)
        if label is not None and label.end <= as_of:
            total += 1
    return total


def _stale_orderbook_rows(rows: list[dict[str, Any]]) -> int:
    total = 0
    for row in rows:
        reason = " ".join(
            str(value or "")
            for value in (
                row.get("reason"),
                row.get("reject_reason"),
                row.get("market_data_trust_reason"),
            )
        ).lower()
        orderbook = (
            row.get("market_context", {}).get("orderbook", {})
            if isinstance(row.get("market_context"), dict)
            else {}
        )
        age = _safe_float(orderbook.get("age_seconds"), math.nan)
        source = str(orderbook.get("source") or "")
        trusted = row.get("market_data_trusted")
        if (
            "stale" in reason
            or trusted is False
            or source not in {"", "live"}
            or (math.isfinite(age) and age > 10.0)
        ):
            total += 1
    return total


def _threshold_replay(
    predictions: list[dict[str, Any]],
    labels: list[Any],
    thresholds: tuple[float, ...],
    as_of: datetime,
    cost_pct: float,
) -> dict[str, Any]:
    report = {}
    for threshold in thresholds:
        trades = []
        for prediction in predictions:
            side, score = _directional_signal(prediction["probabilities"])
            if side is None or score < threshold:
                continue
            label = _label_for_timestamp(prediction["timestamp"], labels)
            matured = label is not None and label.end <= as_of
            outcome = _outcome_pct(side, label, cost_pct) if matured else None
            trades.append({"side": side, "score": score, "outcome_pct": outcome})
        matured_outcomes = [
            float(item["outcome_pct"])
            for item in trades
            if item["outcome_pct"] is not None
        ]
        report[f"{threshold:.2f}"] = {
            "potential_trades": len(trades),
            "matured_trades": len(matured_outcomes),
            "shadow_profit_factor": _profit_factor(matured_outcomes),
            "shadow_expectancy_pct": (
                float(np.mean(matured_outcomes)) if matured_outcomes else None
            ),
        }
    return report


def _directional_signal(probabilities: dict[str, float]) -> tuple[str | None, float]:
    long_score = float(probabilities.get("long", 0.0))
    short_score = float(probabilities.get("short", 0.0))
    if long_score <= 0.0 and short_score <= 0.0:
        return None, 0.0
    return ("long", long_score) if long_score >= short_score else ("short", short_score)


def _outcome_pct(side: str, label: Any, cost_pct: float) -> float:
    if label is None:
        return 0.0
    gross = max(abs(float(label.score_pct)), 0.01)
    signed = gross if side == label.target else -gross
    return signed - cost_pct


def _current_candidate(
    latest: dict[str, Any] | None,
    thresholds: tuple[float, ...],
) -> dict[str, Any]:
    if latest is None:
        return {"available": False}
    probabilities = latest["probabilities"]
    side, side_score = _directional_signal(probabilities)
    flat_score = float(probabilities.get("flat", 0.0))
    threshold_checks = {
        f"{threshold:.2f}": bool(side is not None and side_score >= threshold)
        for threshold in thresholds
    }
    shadow = latest["shadow"]
    row = latest["row"]
    validation = (
        row.get("details", {}).get("metadata", {}).get("model_validation")
        if isinstance(row.get("details"), dict)
        and isinstance(row.get("details", {}).get("metadata"), dict)
        else {}
    )
    return {
        "available": True,
        "timestamp": latest["timestamp"].isoformat(),
        "target": latest["target"],
        "score": latest["score"],
        "probabilities": probabilities,
        "best_direction": side,
        "best_direction_score": side_score,
        "flat_score": flat_score,
        "why_flat": [
            f"flat_score={flat_score:.4f}",
            f"best_direction={side or 'none'}:{side_score:.4f}",
            f"direction_below_thresholds={not any(threshold_checks.values())}",
            f"candidate_can_trade={bool(shadow.get('can_trade', False))}",
            f"active_model_gate={validation.get('reason') or row.get('ml_status')}",
        ],
        "threshold_checks": threshold_checks,
        "model_status": shadow.get("model_status"),
        "promotion_status": shadow.get("promotion_status"),
        "can_trade": bool(shadow.get("can_trade", False)),
        "active_gate_reason": validation.get("reason") or row.get("ml_status"),
    }


def _top_flat_features(
    model: FeedbackEnsemble | None,
    latest: dict[str, Any] | None,
    predictions: list[dict[str, Any]],
    top_features: int,
) -> list[dict[str, Any]]:
    if model is None or latest is None or not latest["features"]:
        return []
    features = dict(latest["features"])
    base = model.predict(features)
    base_flat = float(base.probabilities.get("flat", 0.0))
    medians = _feature_medians(predictions[-2000:])
    impacts = []
    for key in model.feature_keys:
        current = float(features.get(key, 0.0))
        replacement = float(medians.get(key, 0.0))
        changed = dict(features)
        changed[key] = replacement
        prediction = model.predict(changed)
        changed_flat = float(prediction.probabilities.get("flat", 0.0))
        delta = base_flat - changed_flat
        impacts.append(
            {
                "feature": key,
                "current": current,
                "baseline": replacement,
                "flat_delta_vs_baseline": delta,
                "abs_delta": abs(delta),
                "direction": "supports_flat" if delta > 0 else "reduces_flat",
            }
        )
    return sorted(impacts, key=lambda item: item["abs_delta"], reverse=True)[:top_features]


def _feature_medians(predictions: list[dict[str, Any]]) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for prediction in predictions:
        for key, value in prediction["features"].items():
            if math.isfinite(value):
                values.setdefault(key, []).append(float(value))
    return {key: float(median(items)) for key, items in values.items() if items}


def _regime_expectancy(regime_report: dict[str, Any]) -> dict[str, Any]:
    regimes = {}
    positive = []
    for name, item in regime_report.get("by_regime", {}).items():
        summary = {
            "trades": item.get("trades"),
            "expectancy_pct": item.get("expectancy_pct"),
            "profit_factor": item.get("profit_factor"),
            "avg_net_ticks": item.get("avg_net_ticks"),
            "gate_status": item.get("promotion_gate", {}).get("status"),
        }
        regimes[name] = summary
        expectancy = item.get("expectancy_pct")
        if expectancy is not None and float(expectancy) > 0:
            positive.append(name)
    return {
        "matured_feature_rows": regime_report.get("matured_feature_rows"),
        "positive_expectancy_regimes": positive,
        "by_regime": regimes,
    }


def _profit_factor(values: list[float]) -> float | None:
    if not values:
        return None
    wins = [value for value in values if value > 0]
    losses = [value for value in values if value <= 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    if gross_loss:
        return gross_profit / gross_loss
    return float("inf") if gross_profit > 0 else None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _parse_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(parsed) or math.isinf(parsed):
        return default
    return parsed
