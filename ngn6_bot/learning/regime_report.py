from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

from ngn6_bot.config import RuntimeConfig
from ngn6_bot.runtime_metadata import add_commit_hash


@dataclass(frozen=True)
class OracleLabel:
    start: datetime
    end: datetime
    target: str
    score_pct: float


REGIME_DEFINITIONS = {
    "morning_long_07_10": "MSK time in [07:00, 10:00), side=long",
    "bb_compression_long": "feature bb_width in [0.10, 0.20], side=long",
    "book_pressure_long": "orderbook depth_pressure <= -0.15, side=long",
    "low_spread_long": "spread_bps <= 4, side=long",
    "evening_short_19_23": "MSK time in [19:00, 23:00), side=short",
}


def generate_regime_report(
    config: RuntimeConfig,
    *,
    decisions_path: str | Path | None = None,
    labels_dir: str | Path | None = None,
    folds: int = 8,
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
    timezone_name = config.timezone
    tz = ZoneInfo(timezone_name)
    labels = _load_oracle_labels(label_root, tz)
    cost_pct = _round_trip_cost_pct(config)
    rows = _decision_rows(decision_file)
    matured_rows = []
    leakage_violations = 0
    for row in rows:
        ts = _parse_timestamp(row.get("timestamp"))
        if ts is None:
            continue
        label = _label_for_timestamp(ts, labels)
        if label is None:
            continue
        feature_ts = _parse_timestamp(row.get("feature_timestamp"))
        if feature_ts is not None and feature_ts > ts:
            leakage_violations += 1
        matured_rows.append((row, ts, label))

    by_regime = {}
    for name in REGIME_DEFINITIONS:
        trades = _regime_trades(name, matured_rows, tz, cost_pct)
        metrics = _metrics(trades)
        by_regime[name] = {
            "definition": REGIME_DEFINITIONS[name],
            **metrics,
            "walk_forward": _walk_forward(trades, max(1, folds)),
            "promotion_gate": _regime_gate(config, metrics, _walk_forward(trades, max(1, folds))),
        }

    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ticker": config.get("instrument", "ticker"),
        "decisions_path": str(decision_file),
        "labels_dir": str(label_root),
        "time_basis": timezone_name,
        "regime_definitions": REGIME_DEFINITIONS,
        "labels": {
            "intervals": len(labels),
            "target_counts": _target_counts(labels),
        },
        "matured_feature_rows": len(matured_rows),
        "checks": {
            "matured_labels_only": True,
            "feature_timestamp_lte_decision_timestamp": leakage_violations == 0,
            "feature_timestamp_violations": leakage_violations,
            "costs_slippage_included": True,
            "round_trip_cost_pct": cost_pct,
            "lookahead_note": "Oracle future path is used only as label; regime predicates use decision-time fields.",
        },
        "by_regime": by_regime,
    }
    add_commit_hash(report)
    return report


def save_regime_report(report: dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def _regime_trades(
    name: str,
    rows: list[tuple[dict[str, Any], datetime, OracleLabel]],
    tz: ZoneInfo,
    cost_pct: float,
) -> list[dict[str, Any]]:
    trades = []
    for row, timestamp, label in rows:
        side = _regime_side(name)
        if side is None or not _matches_regime(name, row, timestamp, tz):
            continue
        gross = label.score_pct if label.target == side else -abs(label.score_pct)
        net = gross - cost_pct
        trades.append(
            {
                "timestamp": timestamp,
                "day": timestamp.astimezone(tz).date().isoformat(),
                "side": side,
                "gross_pct": gross,
                "net_pct": net,
                "net_ticks": _pct_to_ticks(row, net),
                "hard_stop": net < 0,
            }
        )
    return trades


def _matches_regime(name: str, row: dict[str, Any], timestamp: datetime, tz: ZoneInfo) -> bool:
    local = timestamp.astimezone(tz)
    features = _features(row)
    orderbook = row.get("market_context", {}).get("orderbook", {}) if isinstance(row.get("market_context"), dict) else {}
    if name == "morning_long_07_10":
        return 7 <= local.hour < 10
    if name == "bb_compression_long":
        bb_width = _safe_float(features.get("bb_width"), math.nan)
        return math.isfinite(bb_width) and 0.10 <= bb_width <= 0.20
    if name == "book_pressure_long":
        pressure = _safe_float(orderbook.get("depth_pressure"), math.nan)
        return math.isfinite(pressure) and pressure <= -0.15
    if name == "low_spread_long":
        spread = _safe_float(orderbook.get("spread_bps"), math.nan)
        return math.isfinite(spread) and spread <= 4.0
    if name == "evening_short_19_23":
        return 19 <= local.hour < 23
    return False


def _regime_side(name: str) -> str | None:
    if name.endswith("_long"):
        return "long"
    if name.endswith("_short"):
        return "short"
    return None


def _metrics(trades: list[dict[str, Any]]) -> dict[str, Any]:
    values = [float(item["net_pct"]) for item in trades]
    ticks = [float(item["net_ticks"]) for item in trades if item.get("net_ticks") is not None]
    day_values: dict[str, float] = defaultdict(float)
    for item in trades:
        day_values[str(item["day"])] += float(item["net_pct"])
    return {
        "trades": len(values),
        "profit_factor": _profit_factor(values),
        "expectancy_pct": float(np.mean(values)) if values else None,
        "avg_net_ticks": float(np.mean(ticks)) if ticks else None,
        "max_drawdown_pct": _max_drawdown(values),
        "win_loss_ratio": _win_loss_ratio(values),
        "hard_stop_share": _share([item["hard_stop"] for item in trades]),
        "days_with_trades": len(day_values),
        "positive_day_share": _share([value > 0 for value in day_values.values()]),
        "day_pnl": dict(sorted(day_values.items())),
    }


def _walk_forward(trades: list[dict[str, Any]], folds: int) -> dict[str, Any]:
    if not trades:
        return {
            "folds": folds,
            "fold_profit_factor": [],
            "fold_expectancy_pct": [],
            "fold_trades": [],
            "positive_folds_share": 0.0,
            "median_fold_pf": None,
        }
    sorted_trades = sorted(trades, key=lambda item: item["timestamp"])
    chunks = np.array_split(sorted_trades, min(folds, len(sorted_trades)))
    fold_values = [[float(item["net_pct"]) for item in chunk] for chunk in chunks]
    fold_pf = [_profit_factor(values) for values in fold_values]
    fold_expectancy = [float(np.mean(values)) if values else None for values in fold_values]
    numeric_pf = [value for value in fold_pf if value is not None and math.isfinite(value)]
    positive = [sum(values) > 0 for values in fold_values if values]
    return {
        "folds": len(fold_values),
        "fold_profit_factor": fold_pf,
        "fold_expectancy_pct": fold_expectancy,
        "fold_trades": [len(values) for values in fold_values],
        "positive_folds_share": _share(positive),
        "median_fold_pf": float(np.median(numeric_pf)) if numeric_pf else None,
    }


def _regime_gate(config: RuntimeConfig, metrics: dict[str, Any], wf: dict[str, Any]) -> dict[str, Any]:
    min_trades = int(config.get("promotion", "regime_min_oos_trades_total", default=50))
    min_pf = float(config.get("promotion", "regime_min_profit_factor_oos", default=1.25))
    min_median_pf = float(config.get("promotion", "regime_min_profit_factor_median_fold", default=1.15))
    min_positive_folds = float(config.get("promotion", "regime_min_positive_folds_share", default=0.70))
    max_dd = float(config.get("promotion", "regime_max_total_oos_drawdown_pct", default=0.08))
    min_ticks = float(config.get("promotion", "regime_min_avg_net_ticks", default=8.0))
    max_hard_stop = float(config.get("promotion", "regime_max_hard_stop_share", default=0.35))
    checks = {
        "min_oos_trades_total": metrics["trades"] >= min_trades,
        "profit_factor_oos": (metrics["profit_factor"] or 0.0) >= min_pf,
        "median_fold_pf": (wf["median_fold_pf"] or 0.0) >= min_median_pf,
        "positive_folds_share": wf["positive_folds_share"] >= min_positive_folds,
        "max_drawdown": abs(metrics["max_drawdown_pct"] or 0.0) <= max_dd,
        "avg_net_ticks": (metrics["avg_net_ticks"] or 0.0) >= min_ticks,
        "hard_stop_share": (metrics["hard_stop_share"] or 0.0) <= max_hard_stop,
    }
    passed = all(checks.values())
    return {
        "status": "shadow" if passed else "rejected",
        "passed": passed,
        "checks": checks,
    }


def _load_oracle_labels(root: Path, tz: ZoneInfo) -> list[OracleLabel]:
    labels: list[OracleLabel] = []
    for path in sorted(root.glob("*_oracle_labels.csv")):
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                target = _target_from_label(str(row.get("label") or ""))
                if target not in {"long", "short", "flat"}:
                    continue
                interval = _csv_interval(row, tz)
                if interval is None:
                    continue
                score = max(abs(_safe_float(row.get("score") or row.get("mfe_pct"))), 0.01)
                labels.append(OracleLabel(interval[0], interval[1], target, score))
    return labels


def _decision_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and row.get("market_context"):
                rows.append(row)
    return rows


def _label_for_timestamp(timestamp: datetime, labels: list[OracleLabel]) -> OracleLabel | None:
    for label in labels:
        if label.start <= timestamp <= label.end:
            return label
    return None


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
        end = end + timedelta(days=1)
    return start.astimezone(timezone.utc), end.astimezone(timezone.utc)


def _target_from_label(label: str) -> str:
    normalized = label.upper()
    if "LONG" in normalized:
        return "long"
    if "SHORT" in normalized:
        return "short"
    if normalized in {"SIDEWAYS", "NO_TRADE", "FLAT", "RANGE", "WEAK_BOUNCE"}:
        return "flat"
    return "unknown"


def _features(row: dict[str, Any]) -> dict[str, Any]:
    context = row.get("market_context") if isinstance(row.get("market_context"), dict) else {}
    features = context.get("features")
    return features if isinstance(features, dict) else {}


def _round_trip_cost_pct(config: RuntimeConfig) -> float:
    slippage_bps = float(config.get("execution", "slippage_bps_assumption", default=0.0))
    commission_bps = float(config.get("execution", "commission_round_trip_bps", default=0.0))
    return (slippage_bps * 2 + commission_bps) / 100.0


def _pct_to_ticks(row: dict[str, Any], value_pct: float) -> float | None:
    context = row.get("market_context") if isinstance(row.get("market_context"), dict) else {}
    orderbook = context.get("orderbook") if isinstance(context.get("orderbook"), dict) else {}
    price = _safe_float(
        row.get("price")
        or context.get("last_price")
        or orderbook.get("mid_price")
        or orderbook.get("best_bid"),
        math.nan,
    )
    if not math.isfinite(price) or price <= 0:
        return None
    return price * value_pct / 100.0 / 0.001


def _profit_factor(values: list[float]) -> float | None:
    wins = [value for value in values if value > 0]
    losses = [value for value in values if value <= 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    if gross_loss:
        return gross_profit / gross_loss
    return float("inf") if gross_profit > 0 else None


def _max_drawdown(values: list[float]) -> float:
    equity = 0.0
    peak = 0.0
    drawdown = 0.0
    for value in values:
        equity += value
        peak = max(peak, equity)
        drawdown = min(drawdown, equity - peak)
    return drawdown


def _win_loss_ratio(values: list[float]) -> float | None:
    wins = sum(1 for value in values if value > 0)
    losses = sum(1 for value in values if value <= 0)
    if losses:
        return wins / losses
    return float("inf") if wins else None


def _share(values: list[bool]) -> float:
    if not values:
        return 0.0
    return sum(1 for value in values if value) / len(values)


def _target_counts(labels: list[OracleLabel]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label in labels:
        counts[label.target] = counts.get(label.target, 0) + 1
    return counts


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


def _project_path(config: RuntimeConfig, value: str | Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    config_path = config.path.resolve()
    project_root = config_path.parent.parent if config_path.parent.name == "config" else config_path.parent
    return (project_root / path).resolve()
