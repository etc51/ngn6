from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

from ngn6_bot.config import RuntimeConfig
from ngn6_bot.learning.feedback_model import FeedbackConfig, label_to_target


@dataclass(frozen=True)
class ShadowLabel:
    start: datetime
    end: datetime
    target: str
    score_pct: float


@dataclass(frozen=True)
class ShadowPrediction:
    timestamp: datetime
    target: str
    score: float
    model_status: str
    promotion_status: str
    can_trade: bool


@dataclass(frozen=True)
class ShadowEvaluationReport:
    candidate_model_path: str
    decisions_path: str
    labels_path: str | None
    predictions: int
    shadow_trade_signals: int
    matured_labels: int
    matched_trade_labels: int
    shadow_days: int
    shadow_profit_factor: float | None
    shadow_avg_net_pct: float
    shadow_drawdown_pct: float
    passed: bool
    reason: str
    gate_details: dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


def evaluate_shadow_predictions(
    config: RuntimeConfig,
    *,
    decisions_path: str | Path | None = None,
    labels_path: str | Path | None = None,
) -> ShadowEvaluationReport:
    feedback_config = FeedbackConfig.from_runtime_config(config)
    decision_file = _project_path(
        config,
        decisions_path
        or config.get("data_collection", "decisions_file", default="data/decisions.jsonl"),
    )
    label_file = _project_path(
        config,
        labels_path
        or config.get(
            "learning",
            "oracle_labels_csv",
            default="reports/daily_oracle/latest_oracle_labels.csv",
        ),
    )
    predictions = _load_shadow_predictions(decision_file)
    labels = _load_labels(label_file, ZoneInfo(config.timezone)) if label_file.exists() else []
    trade_predictions = [item for item in predictions if item.target in {"long", "short"}]
    trade_days = {
        item.timestamp.astimezone(ZoneInfo(config.timezone)).date().isoformat()
        for item in trade_predictions
    }

    outcomes: list[float] = []
    matured = 0
    matched_trade_labels = 0
    for prediction in predictions:
        label = _label_for_timestamp(prediction.timestamp, labels)
        if label is None:
            continue
        matured += 1
        if prediction.target not in {"long", "short"}:
            continue
        matched_trade_labels += 1
        outcomes.append(_prediction_outcome_pct(prediction, label))

    profit_factor = _profit_factor(outcomes)
    drawdown = _max_drawdown_pct(outcomes)
    avg_net = float(np.mean(outcomes)) if outcomes else 0.0
    passed, reason, gate_details = _shadow_gates(
        feedback_config,
        shadow_days=len(trade_days),
        trade_signals=len(trade_predictions),
        matched_trade_labels=matched_trade_labels,
        profit_factor=profit_factor,
        drawdown_pct=drawdown,
    )

    return ShadowEvaluationReport(
        candidate_model_path=str(feedback_config.candidate_model_path or ""),
        decisions_path=str(decision_file),
        labels_path=str(label_file) if label_file.exists() else None,
        predictions=len(predictions),
        shadow_trade_signals=len(trade_predictions),
        matured_labels=matured,
        matched_trade_labels=matched_trade_labels,
        shadow_days=len(trade_days),
        shadow_profit_factor=profit_factor,
        shadow_avg_net_pct=avg_net,
        shadow_drawdown_pct=drawdown,
        passed=passed,
        reason=reason,
        gate_details=gate_details,
    )


def save_shadow_report(report: ShadowEvaluationReport, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(report.to_json(), encoding="utf-8")
    return target


def _load_shadow_predictions(path: Path) -> list[ShadowPrediction]:
    if not path.exists():
        return []
    predictions: list[ShadowPrediction] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            timestamp = _parse_timestamp(payload.get("timestamp"))
            shadow = _candidate_shadow_from_decision(payload)
            if timestamp is None or not isinstance(shadow, dict):
                continue
            predictions.append(
                ShadowPrediction(
                    timestamp=timestamp,
                    target=str(shadow.get("target") or "unknown").lower(),
                    score=_safe_float(shadow.get("score")),
                    model_status=str(shadow.get("model_status") or "candidate"),
                    promotion_status=str(shadow.get("promotion_status") or "candidate"),
                    can_trade=bool(shadow.get("can_trade", False)),
                )
            )
    return predictions


def _candidate_shadow_from_decision(payload: dict[str, Any]) -> dict[str, Any] | None:
    for candidate in [
        payload.get("metadata", {}).get("candidate_shadow"),
        payload.get("details", {}).get("metadata", {}).get("candidate_shadow"),
    ]:
        if isinstance(candidate, dict):
            return candidate
    return None


def _load_labels(path: Path, tz: ZoneInfo) -> list[ShadowLabel]:
    if path.suffix.lower() == ".csv":
        return _load_oracle_csv_labels(path, tz)
    labels: list[ShadowLabel] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            timestamp = _parse_timestamp(payload.get("timestamp"))
            target = str(payload.get("target") or label_to_target(str(payload.get("label") or "")))
            if timestamp is None or target == "unknown":
                continue
            labels.append(
                ShadowLabel(
                    start=timestamp,
                    end=timestamp + timedelta(minutes=15),
                    target=target,
                    score_pct=_safe_float(payload.get("pnl_pct") or payload.get("score")),
                )
            )
    return labels


def _load_oracle_csv_labels(path: Path, tz: ZoneInfo) -> list[ShadowLabel]:
    labels: list[ShadowLabel] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            target = label_to_target(str(row.get("label") or ""))
            if target not in {"long", "short", "flat"}:
                continue
            interval = _csv_interval(row, tz)
            if interval is None:
                continue
            labels.append(
                ShadowLabel(
                    start=interval[0],
                    end=interval[1],
                    target=target,
                    score_pct=abs(_safe_float(row.get("score") or row.get("mfe_pct"))),
                )
            )
    return labels


def _csv_interval(row: dict[str, str], tz: ZoneInfo) -> tuple[datetime, datetime] | None:
    try:
        trading_date = datetime.fromisoformat(str(row.get("date"))).date()
        start_time = datetime.strptime(str(row.get("start_time")), "%H:%M:%S").time()
        end_time = datetime.strptime(str(row.get("end_time")), "%H:%M:%S").time()
    except (TypeError, ValueError):
        return None
    start = datetime.combine(trading_date, start_time, tzinfo=tz)
    end = datetime.combine(trading_date, end_time, tzinfo=tz)
    if end <= start:
        end += timedelta(days=1)
    return start.astimezone(ZoneInfo("UTC")), end.astimezone(ZoneInfo("UTC"))


def _label_for_timestamp(timestamp: datetime, labels: list[ShadowLabel]) -> ShadowLabel | None:
    target = timestamp.astimezone(ZoneInfo("UTC"))
    for label in labels:
        if label.start <= target <= label.end:
            return label
    return None


def _prediction_outcome_pct(prediction: ShadowPrediction, label: ShadowLabel) -> float:
    value = max(abs(label.score_pct), 0.01)
    if prediction.target == label.target:
        return value
    return -value


def _shadow_gates(
    config: FeedbackConfig,
    *,
    shadow_days: int,
    trade_signals: int,
    matched_trade_labels: int,
    profit_factor: float | None,
    drawdown_pct: float,
) -> tuple[bool, str, dict[str, Any]]:
    details = {
        "min_days": config.shadow_min_days,
        "min_trade_signals": config.shadow_min_trade_signals,
        "min_profit_factor": config.shadow_min_profit_factor,
        "max_drawdown_pct": config.shadow_max_drawdown_pct,
        "shadow_days": shadow_days,
        "trade_signals": trade_signals,
        "matched_trade_labels": matched_trade_labels,
        "profit_factor": profit_factor,
        "drawdown_pct": drawdown_pct,
    }
    if matched_trade_labels <= 0:
        return False, "insufficient_matured_labels", details
    if shadow_days < int(details["min_days"]):
        return False, "shadow_days_below_min", details
    if trade_signals < int(details["min_trade_signals"]):
        return False, "shadow_trade_signals_below_min", details
    if profit_factor is None or profit_factor < float(details["min_profit_factor"]):
        return False, "shadow_profit_factor_below_min", details
    max_dd = float(details["max_drawdown_pct"])
    max_dd_pct = max_dd * 100 if max_dd <= 1 else max_dd
    if abs(drawdown_pct) > max_dd_pct:
        return False, "shadow_drawdown_above_max", details
    return True, "accepted", details


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


def _max_drawdown_pct(values: list[float]) -> float:
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for value in values:
        equity += value
        peak = max(peak, equity)
        max_dd = min(max_dd, equity - peak)
    return max_dd


def _parse_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=ZoneInfo("UTC"))
    return parsed.astimezone(ZoneInfo("UTC"))


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _project_path(config: RuntimeConfig, value: str | Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    config_path = config.path.resolve()
    project_root = config_path.parent.parent if config_path.parent.name == "config" else config_path.parent
    return (project_root / path).resolve()
