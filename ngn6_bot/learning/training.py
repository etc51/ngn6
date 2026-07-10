from __future__ import annotations

import copy
import csv
import json
import shutil
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from ngn6_bot.backtest import fetch_1m_history, run_walk_forward
from ngn6_bot.config import RuntimeConfig
from ngn6_bot.indicators import add_indicators, candles_to_frame
from ngn6_bot.learning.ensemble import (
    FeedbackEnsemble,
    FeedbackEnsembleReport,
    train_feedback_ensemble,
)
from ngn6_bot.learning.feedback_model import (
    FEATURE_KEYS,
    FeedbackConfig,
    FeedbackExample,
    build_feature_snapshot,
    label_to_target,
    load_feedback_examples,
)
from ngn6_bot.learning.paper_feedback import sync_paper_trade_feedback
from ngn6_bot.microstructure_replay import MicrostructureReplay, neutral_orderbook, neutral_trade_flow
from ngn6_bot.models import Candle
from ngn6_bot.runtime_metadata import add_commit_hash


@dataclass(frozen=True)
class FeedbackTrainingResult:
    figi: str
    candles: int
    generated_examples: int
    total_examples: int
    report: FeedbackEnsembleReport


def train_feedback_from_api(
    config: RuntimeConfig,
    logger,
    *,
    minutes: int,
    output_path: str | Path | None = None,
    min_examples: int | None = None,
) -> FeedbackTrainingResult:
    figi, candles = fetch_1m_history(config, logger, minutes)
    feedback_config = FeedbackConfig.from_runtime_config(config)
    examples, generated = build_training_examples(config, candles, feedback_config)
    examples = _limit_examples(examples, feedback_config.max_examples)
    target_path = output_path or feedback_config.ensemble_model_path
    target_path = Path(target_path)
    minimum = min_examples if min_examples is not None else feedback_config.ensemble_min_examples
    if feedback_config.promote_enabled:
        candidate_path = feedback_config.candidate_model_path or _candidate_path(target_path)
        report = train_feedback_ensemble(
            examples,
            output_path=candidate_path,
            min_examples=minimum,
            class_balance=feedback_config.class_balance,
        )
        promotion = _promotion_decision(
            report=report,
            target_path=target_path,
            candidate_path=candidate_path,
            feedback_config=feedback_config,
            runtime_config=config,
            candles=candles,
            figi=figi,
        )
        report = replace(
            report,
            promotion_score=promotion.get("candidate_score"),
            task_reports={**report.task_reports, "promotion": promotion},
            promotion_status=_promotion_status_for_decision(promotion),
            model_status="candidate",
            promotion_metrics=_promotion_metrics_from_decision(promotion),
        )
        _set_model_promotion_metadata(
            candidate_path,
            status=_promotion_status_for_decision(promotion),
            model_status="candidate",
            promotion_metrics=_promotion_metrics_from_decision(promotion),
        )
        promoted = bool(promotion["promoted"])
        if promoted:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(candidate_path, target_path)
            report = replace(report, path=target_path, promoted=True)
        else:
            report = replace(report, promoted=False)
        _write_training_report(report, _training_report_path(target_path))
    else:
        report = train_feedback_ensemble(
            examples,
            output_path=target_path,
            min_examples=minimum,
            class_balance=feedback_config.class_balance,
        )
        _write_training_report(report, _training_report_path(target_path))
    return FeedbackTrainingResult(
        figi=figi,
        candles=len(candles),
        generated_examples=generated,
        total_examples=len(examples),
        report=report,
    )


def build_training_examples(
    config: RuntimeConfig,
    candles_1m: list[Candle],
    feedback_config: FeedbackConfig | None = None,
) -> tuple[list[FeedbackExample], int]:
    active_config = feedback_config or FeedbackConfig.from_runtime_config(config)
    if bool(config.get("learning", "paper_trade_feedback_enabled", default=False)):
        sync_paper_trade_feedback(config)
    examples = load_feedback_examples(active_config)
    generated = 0
    if active_config.generate_pnl_examples:
        pnl_examples = _pnl_examples(config, candles_1m, active_config)
        examples.extend(pnl_examples)
        generated += len(pnl_examples)
    if active_config.decision_examples_enabled:
        decision_examples = _decision_examples(config)
        examples.extend(decision_examples)
        generated += len(decision_examples)
    for csv_path in (active_config.legacy_labels_csv, active_config.oracle_labels_csv):
        if csv_path is None:
            continue
        interval_examples = _legacy_interval_examples(config, candles_1m, csv_path)
        examples.extend(interval_examples)
        generated += len(interval_examples)
    return _dedupe_examples(examples), generated


def _legacy_interval_examples(
    config: RuntimeConfig,
    candles_1m: list[Candle],
    csv_path: Path,
) -> list[FeedbackExample]:
    if not csv_path.exists() or not candles_1m:
        return []

    tz = ZoneInfo(config.timezone)
    frames = _indicator_frames(config, candles_1m)
    microstructure = MicrostructureReplay.from_config(config)
    candles_by_timeframe = {
        "1min": candles_1m,
        "5min": _aggregate_candles(candles_1m, "5min"),
        "15min": _aggregate_candles(candles_1m, "15min"),
    }
    examples: list[FeedbackExample] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            label = str(row.get("label") or "").strip().upper()
            target = label_to_target(label)
            if target == "unknown":
                continue
            timeframe = str(row.get("timeframe") or "15min").strip() or "15min"
            if timeframe not in candles_by_timeframe:
                continue
            interval = _parse_interval(row, tz)
            if interval is None:
                continue
            start_utc, end_utc = interval
            sampled = [
                candle
                for candle in candles_by_timeframe[timeframe]
                if start_utc <= _utc(candle.timestamp) <= end_utc
            ]
            for candle in sampled:
                feature_time = _utc(candle.timestamp)
                snapshot = _snapshot_at(config, frames, feature_time, microstructure)
                if snapshot is None:
                    continue
                task = "exit" if target == "exit" else "entry"
                examples.append(
                    FeedbackExample(
                        label=label,
                        target=target,
                        features=snapshot,
                        source=str(csv_path),
                        timestamp=feature_time.isoformat(),
                        task=task,
                    )
                )
    return examples


def _pnl_examples(
    config: RuntimeConfig,
    candles_1m: list[Candle],
    feedback_config: FeedbackConfig,
) -> list[FeedbackExample]:
    if not candles_1m:
        return []
    frames = _indicator_frames(config, candles_1m)
    microstructure = MicrostructureReplay.from_config(config)
    candles_by_timeframe = {
        "1min": candles_1m,
        "5min": _aggregate_candles(candles_1m, "5min"),
        "15min": _aggregate_candles(candles_1m, "15min"),
    }
    source_candles = candles_by_timeframe.get(feedback_config.pnl_timeframe, candles_1m)
    entry_horizon = max(1, feedback_config.entry_label_horizon_bars)
    exit_horizon = max(1, feedback_config.exit_label_horizon_bars)
    max_horizon = max(entry_horizon, exit_horizon)
    costs_pct = _round_trip_cost_pct(config)
    examples: list[FeedbackExample] = []
    for index in range(0, max(0, len(source_candles) - max_horizon)):
        candle = source_candles[index]
        feature_time = _utc(candle.timestamp)
        snapshot = _snapshot_at(config, frames, feature_time, microstructure)
        if snapshot is None:
            continue
        price = float(candle.close)
        entry_future = float(source_candles[index + entry_horizon].close)
        long_net = (entry_future - price) / price * 100 - costs_pct
        short_net = (price - entry_future) / price * 100 - costs_pct
        entry_target = _entry_target(long_net, short_net, feedback_config)
        examples.append(
            FeedbackExample(
                label=f"PNL_{entry_target.upper()}",
                target=entry_target,
                features=snapshot,
                source="pnl_replay",
                timestamp=feature_time.isoformat(),
                task="entry",
                pnl_pct=max(long_net, short_net, 0.0),
                outcomes={"long": long_net, "short": short_net, "flat": 0.0},
            )
        )

        exit_future = float(source_candles[index + exit_horizon].close)
        long_hold = (exit_future - price) / price * 100
        short_hold = (price - exit_future) / price * 100
        for side_name, side_value, hold_value in [
            ("LONG", 1.0, long_hold),
            ("SHORT", -1.0, short_hold),
        ]:
            exit_features = {**snapshot, "position_side": side_value}
            exit_target = "hold" if hold_value >= feedback_config.hold_min_net_pct else "exit"
            examples.append(
                FeedbackExample(
                    label=f"PNL_{side_name}_{exit_target.upper()}",
                    target=exit_target,
                    features=exit_features,
                    source="pnl_replay",
                    timestamp=feature_time.isoformat(),
                    task="exit",
                    pnl_pct=hold_value,
                    outcomes={"hold": hold_value, "exit": 0.0},
                )
            )
    return examples


def _entry_target(long_net: float, short_net: float, config: FeedbackConfig) -> str:
    if (
        long_net >= config.min_entry_net_pct
        and long_net - short_net >= config.min_direction_edge_pct
    ):
        return "long"
    if (
        short_net >= config.min_entry_net_pct
        and short_net - long_net >= config.min_direction_edge_pct
    ):
        return "short"
    return "flat"


def _decision_examples(config: RuntimeConfig) -> list[FeedbackExample]:
    path = Path(config.get("data_collection", "decisions_file", default="data/decisions.jsonl"))
    if not path.exists():
        return []
    examples: list[FeedbackExample] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            features = _decision_features(item)
            quality = _decision_quality(item, features)
            if not quality["feature_complete"]:
                continue
            if not quality["label_matured"] or not quality["market_data_trusted"]:
                continue
            action = str(item.get("action") or "").lower()
            side = str(item.get("side") or "").lower()
            target = _decision_target(action, side)
            if target is None:
                continue
            task = "exit" if target in {"hold", "exit"} else "entry"
            examples.append(
                FeedbackExample(
                    label=f"DECISION_{action.upper()}",
                    target=target,
                    features=features,
                    source=str(path),
                    timestamp=str(item.get("timestamp") or "") or None,
                    task=task,
                    feature_complete=quality["feature_complete"],
                    label_matured=quality["label_matured"],
                    market_data_trusted=quality["market_data_trusted"],
                    reject_reason=quality["reject_reason"],
                )
            )
    return examples


def _decision_features(item: dict) -> dict[str, float]:
    for candidate in [
        item.get("metadata", {}).get("features"),
        item.get("details", {}).get("metadata", {}).get("features"),
        item.get("market_context", {}).get("features"),
    ]:
        if isinstance(candidate, dict):
            features = {}
            for key, value in candidate.items():
                try:
                    features[str(key)] = float(value)
                except (TypeError, ValueError):
                    continue
            return features
    return {}


def _decision_quality(item: dict, features: dict[str, float]) -> dict[str, bool | str | None]:
    reason = str(item.get("reason") or item.get("reject_reason") or "")
    feature_complete = bool(item.get("feature_complete", False)) and bool(features)
    missing_fields = [
        key
        for key in FEATURE_KEYS
        if key not in features
    ]
    if missing_fields:
        feature_complete = False
    market_data_trusted = bool(item.get("market_data_trusted", False))
    if not market_data_trusted and isinstance(item.get("market_context"), dict):
        orderbook = item["market_context"].get("orderbook") or {}
        source = str(orderbook.get("source") or "")
        age = orderbook.get("age_seconds")
        try:
            age_ok = age is not None and float(age) <= 10.0
        except (TypeError, ValueError):
            age_ok = False
        market_data_trusted = source == "live" and age_ok
    if "stale" in reason.lower() or "untrusted" in reason.lower():
        market_data_trusted = False
    label_matured = bool(item.get("label_matured", False))
    reject_reason = None
    if not feature_complete:
        reject_reason = "missing_features"
    elif not label_matured:
        reject_reason = "label_not_matured"
    elif not market_data_trusted:
        reject_reason = "market_data_not_trusted"
    return {
        "feature_complete": feature_complete,
        "label_matured": label_matured,
        "market_data_trusted": market_data_trusted,
        "reject_reason": reject_reason,
    }


def _decision_target(action: str, side: str) -> str | None:
    if action == "open_accepted":
        if side in {"long", "short"}:
            return side
        return None
    if action in {"skip", "open_rejected"}:
        return "flat"
    if action == "close_accepted":
        return "exit"
    if action == "hold":
        return "hold"
    return None


def _indicator_frames(config: RuntimeConfig, candles_1m: list[Candle]):
    candles_by_timeframe = {
        "1min": candles_1m,
        "5min": _aggregate_candles(candles_1m, "5min"),
        "15min": _aggregate_candles(candles_1m, "15min"),
    }
    return {
        timeframe: add_indicators(
            candles_to_frame(candles),
            ema_fast=int(config.get("indicators", "ema_fast")),
            ema_slow=int(config.get("indicators", "ema_slow")),
            rsi_period=int(config.get("indicators", "rsi_period")),
            atr_period=int(config.get("indicators", "atr_period", default=14)),
            adx_period=int(config.get("indicators", "adx_period", default=14)),
            macd_fast=int(config.get("indicators", "macd_fast", default=12)),
            macd_slow=int(config.get("indicators", "macd_slow", default=26)),
            macd_signal=int(config.get("indicators", "macd_signal", default=9)),
            bollinger_period=int(config.get("indicators", "bollinger_period")),
            bollinger_std=float(config.get("indicators", "bollinger_std")),
            volume_ma_period=int(config.get("indicators", "volume_ma_period")),
        )
        for timeframe, candles in candles_by_timeframe.items()
    }


def _snapshot_at(
    config: RuntimeConfig,
    frames: dict[str, object],
    feature_time: datetime,
    microstructure: MicrostructureReplay,
):
    execution_df = frames["1min"].loc[:feature_time].tail(320)
    confirmation_df = frames["5min"].loc[:feature_time].tail(300)
    context_df = frames["15min"].loc[:feature_time].tail(300)
    if len(execution_df) < int(config.get("signals", "legacy_min_candles", default=40)):
        return None
    price = float(execution_df["close"].iloc[-1])
    micro_snapshot = microstructure.at(feature_time)
    if micro_snapshot is None:
        orderbook = neutral_orderbook(price)
        trade_flow = neutral_trade_flow()
    else:
        orderbook = micro_snapshot.orderbook
        trade_flow = micro_snapshot.trade_flow
    return build_feature_snapshot(
        execution_df=execution_df,
        confirmation_df=confirmation_df,
        context_df=context_df,
        orderbook=orderbook,
        trade_flow=trade_flow,
        now=feature_time,
    )


def _aggregate_candles(candles: list[Candle], timeframe: str) -> list[Candle]:
    minutes = {"5min": 5, "15min": 15}[timeframe]
    aggregated: dict[datetime, Candle] = {}
    for candle in candles:
        timestamp = _utc(candle.timestamp)
        bucket_minute = timestamp.minute - (timestamp.minute % minutes)
        bucket_start = timestamp.replace(minute=bucket_minute, second=0, microsecond=0)
        bucket_close = bucket_start + timedelta(minutes=minutes)
        existing = aggregated.get(bucket_start)
        if existing is None:
            aggregated[bucket_start] = Candle(
                timestamp=bucket_close,
                open=candle.open,
                high=candle.high,
                low=candle.low,
                close=candle.close,
                volume=candle.volume,
                timeframe=timeframe,
            )
            continue
        aggregated[bucket_start] = Candle(
            timestamp=bucket_close,
            open=existing.open,
            high=max(existing.high, candle.high),
            low=min(existing.low, candle.low),
            close=candle.close,
            volume=existing.volume + candle.volume,
            timeframe=timeframe,
        )
    return [aggregated[key] for key in sorted(aggregated)]


def _parse_interval(row: dict[str, str], tz: ZoneInfo) -> tuple[datetime, datetime] | None:
    try:
        trading_date = date.fromisoformat(str(row.get("date") or ""))
        start_time = time.fromisoformat(str(row.get("start_time") or ""))
        end_time = time.fromisoformat(str(row.get("end_time") or ""))
    except ValueError:
        return None
    start_local = datetime.combine(trading_date, start_time, tzinfo=tz)
    end_local = datetime.combine(trading_date, end_time, tzinfo=tz)
    if end_local < start_local:
        end_local += timedelta(days=1)
    return _utc(start_local), _utc(end_local)


def _dedupe_examples(examples: list[FeedbackExample]) -> list[FeedbackExample]:
    seen: set[tuple[str, str, str, str | None]] = set()
    result: list[FeedbackExample] = []
    for item in examples:
        key = (item.task, item.target, item.label, item.timestamp)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _limit_examples(examples: list[FeedbackExample], max_examples: int) -> list[FeedbackExample]:
    if max_examples <= 0 or len(examples) <= max_examples:
        return examples
    indexed = list(enumerate(examples))
    indexed.sort(key=lambda item: (_timestamp_key(item[1].timestamp), item[0]))
    return [item for _, item in indexed[-max_examples:]]


def _timestamp_key(value: str | None) -> str:
    return value or ""


def _round_trip_cost_pct(config: RuntimeConfig) -> float:
    slippage_bps = float(config.get("execution", "slippage_bps_assumption", default=0.0))
    commission_bps = float(config.get("execution", "commission_round_trip_bps", default=0.0))
    return (slippage_bps * 2 + commission_bps) / 100.0


def _candidate_path(target_path: Path) -> Path:
    suffix = target_path.suffix or ".joblib"
    return target_path.with_name(f"{target_path.stem}.candidate{suffix}")


def _training_report_path(target_path: Path) -> Path:
    return target_path.with_name(f"{target_path.stem}.training_report.json")


def _write_training_report(report: FeedbackEnsembleReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(report)
    payload["path"] = str(report.path)
    add_commit_hash(payload)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _promotion_decision(
    *,
    report: FeedbackEnsembleReport,
    target_path: Path,
    candidate_path: Path,
    feedback_config: FeedbackConfig,
    runtime_config: RuntimeConfig,
    candles: list[Candle],
    figi: str,
) -> dict:
    candidate_score = report.promotion_score
    current_score = None
    candidate_backtest = None
    current_backtest = None
    reason = "ml_score_gate"

    if feedback_config.promotion_backtest_enabled:
        eval_candles = _promotion_candles(candles, feedback_config.promotion_backtest_max_candles)
        candidate_eval = _evaluate_model_path(
            runtime_config,
            candidate_path,
            eval_candles,
            figi,
            folds=feedback_config.promotion_backtest_folds,
        )
        candidate_backtest = candidate_eval
        candidate_score = candidate_eval["score"]
        reason = "backtest_gate"
        if target_path.exists():
            current_eval = _evaluate_model_path(
                runtime_config,
                target_path,
                eval_candles,
                figi,
                folds=feedback_config.promotion_backtest_folds,
            )
            current_backtest = current_eval
            current_score = current_eval["score"]
    elif target_path.exists():
        try:
            current = FeedbackEnsemble.load(target_path)
            current_score = current.promotion_score
        except Exception:
            current_score = None

    accepted, reject_reason = _promotion_requirements_met(
        candidate_score,
        candidate_backtest,
        feedback_config,
        report=report,
    )
    if not accepted:
        return {
            "promoted": False,
            "reason": reject_reason,
            "candidate_score": candidate_score,
            "current_score": current_score,
            "candidate_backtest": candidate_backtest,
            "current_backtest": current_backtest,
        }

    shadow_ok, shadow_reason, shadow_details = _shadow_gate_passed(feedback_config, runtime_config)
    if feedback_config.shadow_required_before_control and not shadow_ok:
        return {
            "promoted": False,
            "reason": shadow_reason,
            "candidate_score": candidate_score,
            "current_score": current_score,
            "candidate_backtest": candidate_backtest,
            "current_backtest": current_backtest,
            "shadow_gate": shadow_details,
        }

    if current_score is None:
        promoted = True
    elif feedback_config.promotion_backtest_enabled:
        promoted = (
            candidate_score
            >= current_score + feedback_config.promotion_min_backtest_improvement_pct
        )
    else:
        promoted = candidate_score >= current_score + feedback_config.promotion_min_improvement

    return {
        "promoted": promoted,
        "reason": reason if promoted else "candidate_not_better_than_current",
        "candidate_score": candidate_score,
        "current_score": current_score,
        "candidate_backtest": candidate_backtest,
        "current_backtest": current_backtest,
    }


def _evaluate_model_path(
    config: RuntimeConfig,
    model_path: Path,
    candles: list[Candle],
    figi: str,
    *,
    folds: int,
) -> dict:
    eval_config = _config_with_model_path(config, model_path)
    report = run_walk_forward(eval_config, candles, figi, max(1, folds))
    fold_metrics = [fold.metrics for fold in report.folds]
    trades = sum(metric.trades for metric in fold_metrics)
    final_equity = sum(metric.final_equity_pct for metric in fold_metrics)
    max_drawdown = min((metric.max_drawdown_pct for metric in fold_metrics), default=0.0)
    profit_factors = [
        metric.profit_factor
        for metric in fold_metrics
        if metric.profit_factor is not None
    ]
    profit_factor = (
        sum(profit_factors) / len(profit_factors)
        if profit_factors
        else None
    )
    score = _backtest_score(final_equity, max_drawdown, profit_factor, trades)
    return {
        "folds": len(fold_metrics),
        "trades": trades,
        "final_equity_pct": final_equity,
        "max_drawdown_pct": max_drawdown,
        "profit_factor": profit_factor,
        "score": score,
        "fold_final_equity_pct": [metric.final_equity_pct for metric in fold_metrics],
        "fold_trades": [metric.trades for metric in fold_metrics],
        "fold_profit_factor": [metric.profit_factor for metric in fold_metrics],
        "fold_max_drawdown_pct": [metric.max_drawdown_pct for metric in fold_metrics],
    }


def _promotion_candles(candles: list[Candle], max_candles: int) -> list[Candle]:
    if max_candles <= 0 or len(candles) <= max_candles:
        return candles
    return candles[-max_candles:]


def _config_with_model_path(config: RuntimeConfig, model_path: Path) -> RuntimeConfig:
    raw = copy.deepcopy(config.raw)
    raw.setdefault("learning", {})["ensemble_model_path"] = str(model_path)
    raw["learning"]["ensemble_enabled"] = True
    raw["learning"]["mode"] = "control"
    raw["learning"]["control_require_promoted_model"] = False
    raw["learning"]["active_can_trade_only_if_promoted"] = False
    raw["learning"]["min_entry_examples"] = 0
    raw["learning"]["min_exit_examples"] = 0
    raw["learning"]["min_examples_per_class"] = 0
    return RuntimeConfig(raw=raw, path=config.path)


def _backtest_score(
    final_equity_pct: float,
    max_drawdown_pct: float,
    profit_factor: float | None,
    trades: int,
) -> float:
    profit_factor_bonus = 0.0 if profit_factor is None else min(profit_factor, 3.0) - 1.0
    trade_penalty = 0.0 if trades > 0 else 10.0
    return final_equity_pct + profit_factor_bonus - abs(max_drawdown_pct) * 0.25 - trade_penalty


def _promotion_requirements_met(
    candidate_score: float | None,
    candidate_backtest: dict | None,
    config: FeedbackConfig,
    *,
    report: FeedbackEnsembleReport | None = None,
) -> tuple[bool, str]:
    if report is not None:
        entry_report = report.task_reports.get("entry") or {}
        exit_report = report.task_reports.get("exit") or {}
        if config.min_entry_examples > 0 and int(entry_report.get("examples", 0)) < config.min_entry_examples:
            return False, "candidate_entry_examples_below_min"
        if config.min_exit_examples > 0 and int(exit_report.get("examples", 0)) < config.min_exit_examples:
            return False, "candidate_exit_examples_below_min"
        class_counts = {
            str(key): int(value)
            for key, value in (entry_report.get("class_counts") or {}).items()
        }
        if config.min_examples_per_class > 0:
            if not class_counts:
                return False, "candidate_entry_class_counts_missing"
            low_classes = {
                target: class_counts.get(target, 0)
                for target in config.control_required_entry_targets
                if class_counts.get(target, 0) < config.min_examples_per_class
            }
            if low_classes:
                return False, "candidate_entry_class_examples_below_min"
        if class_counts and config.max_class_share < 1.0:
            total = sum(class_counts.values()) or 1
            max_share = max(class_counts.values()) / total
            if max_share > config.max_class_share:
                return False, "candidate_class_share_above_max"
    if candidate_score is None:
        return False, "candidate_score_missing"
    if candidate_score < config.promotion_min_score:
        return False, "candidate_score_below_min"
    if candidate_backtest is None:
        return True, "accepted"
    if int(candidate_backtest["trades"]) < config.promotion_min_trades:
        return False, "candidate_backtest_too_few_trades"
    fold_trades = [
        int(value)
        for value in candidate_backtest.get("fold_trades", [])
        if value is not None
    ]
    if config.promotion_min_oos_trades_per_fold > 0 and any(
        value < config.promotion_min_oos_trades_per_fold for value in fold_trades
    ):
        return False, "candidate_fold_trades_below_min"
    profit_factor = candidate_backtest.get("profit_factor")
    if profit_factor is None or float(profit_factor) < config.promotion_min_profit_factor:
        return False, "candidate_backtest_profit_factor_below_min"
    fold_profit_factors = [
        float(value)
        for value in candidate_backtest.get("fold_profit_factor", [])
        if value is not None
    ]
    if (
        config.promotion_min_profit_factor_median_fold > 0
        and fold_profit_factors
        and sorted(fold_profit_factors)[len(fold_profit_factors) // 2]
        < config.promotion_min_profit_factor_median_fold
    ):
        return False, "candidate_median_fold_profit_factor_below_min"
    if config.promotion_min_positive_folds_share > 0:
        fold_equity = [
            float(value)
            for value in candidate_backtest.get("fold_final_equity_pct", [])
            if value is not None
        ]
        if fold_equity:
            positive_share = sum(1 for value in fold_equity if value > 0) / len(fold_equity)
            if positive_share < config.promotion_min_positive_folds_share:
                return False, "candidate_positive_folds_share_below_min"
    if float(candidate_backtest["final_equity_pct"]) < config.promotion_min_final_equity_pct:
        return False, "candidate_backtest_final_equity_below_min"
    if config.promotion_max_single_fold_drawdown_pct > 0:
        fold_drawdowns = [
            abs(float(value))
            for value in candidate_backtest.get("fold_max_drawdown_pct", [])
            if value is not None
        ]
        if any(
            _fraction_from_percent_or_fraction(value)
            > _fraction_from_percent_or_fraction(config.promotion_max_single_fold_drawdown_pct)
            for value in fold_drawdowns
        ):
            return False, "candidate_single_fold_drawdown_above_max"
    if _fraction_from_percent_or_fraction(
        abs(float(candidate_backtest["max_drawdown_pct"]))
    ) > _fraction_from_percent_or_fraction(config.promotion_max_drawdown_abs_pct):
        return False, "candidate_backtest_drawdown_above_max"
    return True, "accepted"


def _shadow_gate_passed(
    feedback_config: FeedbackConfig,
    runtime_config: RuntimeConfig,
) -> tuple[bool, str, dict]:
    if not feedback_config.shadow_required_before_control:
        return True, "shadow_gate_not_required", {}
    report_path = _project_path(runtime_config, feedback_config.shadow_report_path)
    if not report_path.exists():
        return False, "shadow_gate_required", {"report": str(report_path), "exists": False}
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False, "shadow_report_invalid", {"report": str(report_path)}
    details = {"report": str(report_path), **report}
    if not bool(report.get("passed", False)):
        return False, str(report.get("reason") or "shadow_gate_failed"), details
    candidate_path = str(feedback_config.candidate_model_path or "")
    report_candidate = str(report.get("candidate_model_path") or "")
    if candidate_path and report_candidate and Path(candidate_path).name != Path(report_candidate).name:
        return False, "shadow_report_candidate_mismatch", details
    return True, "shadow_gate_passed", details


def _promotion_status_for_decision(decision: dict) -> str:
    if bool(decision.get("promoted")):
        return "approved"
    reason = str(decision.get("reason") or "")
    if reason.startswith("shadow_") or reason in {
        "insufficient_matured_labels",
        "shadow_days_below_min",
        "shadow_trade_signals_below_min",
        "shadow_profit_factor_below_min",
        "shadow_drawdown_above_max",
    }:
        return "candidate"
    return "rejected"


def _promotion_metrics_from_decision(decision: dict) -> dict:
    backtest = decision.get("candidate_backtest") or {}
    metrics = {
        "candidate_score": decision.get("candidate_score"),
        "current_score": decision.get("current_score"),
        "reason": decision.get("reason"),
        "oos_trades_total": backtest.get("trades"),
        "oos_profit_factor": backtest.get("profit_factor"),
        "max_total_oos_drawdown_pct": backtest.get("max_drawdown_pct"),
        "final_equity_pct": backtest.get("final_equity_pct"),
        "folds": backtest.get("folds"),
        "fold_trades": backtest.get("fold_trades"),
        "fold_final_equity_pct": backtest.get("fold_final_equity_pct"),
        "shadow_gate": decision.get("shadow_gate"),
    }
    return {key: value for key, value in metrics.items() if value is not None}


def _set_model_promotion_metadata(
    path: Path,
    *,
    status: str,
    model_status: str,
    promotion_metrics: dict,
) -> None:
    try:
        ensemble = FeedbackEnsemble.load(path)
    except Exception:
        return
    ensemble.promotion_status = status
    ensemble.model_status = model_status
    ensemble.promotion_metrics = dict(promotion_metrics)
    ensemble.metadata.update(
        {
            "promotion_status": status,
            "model_status": model_status,
            "promotion_metrics": dict(promotion_metrics),
        }
    )
    ensemble.save(path)


def _fraction_from_percent_or_fraction(value: float) -> float:
    parsed = abs(float(value))
    if parsed > 1.0:
        return parsed / 100.0
    return parsed


def _utc(timestamp: datetime) -> datetime:
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=ZoneInfo("UTC"))
    return timestamp.astimezone(ZoneInfo("UTC"))


def _project_path(config: RuntimeConfig, value: str | Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    config_path = config.path.resolve()
    project_root = config_path.parent.parent if config_path.parent.name == "config" else config_path.parent
    return (project_root / path).resolve()
