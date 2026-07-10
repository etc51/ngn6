from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ngn6_bot.models import OrderBookFeatures, Side, Signal, TradeFlowFeatures


LABEL_TARGETS = {
    "LONG": "long",
    "STRONG_LONG": "long",
    "VERY_STRONG_LONG": "long",
    "WEAK_LONG": "flat",
    "LONG_CONTINUATION": "long",
    "SHORT": "short",
    "GOOD_SHORT": "short",
    "FAST_SHORT": "short",
    "SHORT_CONTINUATION": "short",
    "LIGHT_DOWNTREND_SHORT": "short",
    "LIGHT_DOWNTREND_SHORT_ZONE": "short",
    "SHORT_FROM_PEAK": "short",
    "SHORT_FROM_PEAK_HOLD_TO_REVERSAL": "short",
    "SIDEWAYS": "flat",
    "NO_TRADE": "flat",
    "FLAT": "flat",
    "RANGE": "flat",
    "WEAK_BOUNCE": "flat",
    "EXIT": "exit",
    "EXIT_BY_ORDERBOOK": "exit",
    "EXIT_ZONE": "exit",
    "TAKE_PROFIT": "exit",
    "CLOSE": "exit",
}

ENTRY_TARGETS = {"long", "short", "flat"}
EXIT_CONTROL_TARGETS = {"hold", "exit"}
LONG_TARGETS = {
    "long",
    "strong_long",
    "very_strong_long",
    "weak_long",
    "long_continuation",
}
SHORT_TARGETS = {
    "short",
    "good_short",
    "fast_short",
    "short_continuation",
    "light_downtrend_short",
    "light_downtrend_short_zone",
    "short_from_peak",
    "short_from_peak_hold_to_reversal",
}
SIDEWAYS_TARGETS = {"sideways", "no_trade", "flat", "range", "weak_bounce"}
EXIT_TARGETS = {"exit", "exit_by_orderbook", "exit_zone", "take_profit", "close"}
WEAK_ENTRY_TARGETS = {"weak_long", "weak_bounce", "sideways", "no_trade", "flat", "range"}
TRAINABLE_TARGETS = ENTRY_TARGETS | EXIT_CONTROL_TARGETS
OPPORTUNITY_TARGETS = {"trade", "no_trade"}
DIRECTION_TARGETS = {"long", "short"}
CONTROL_MODES = {"control", "shadow_then_control"}
APPROVED_PROMOTION_STATUS = "approved"
FEATURE_SCHEMA_VERSION = 2


FEATURE_KEYS = [
    "return_1",
    "return_3",
    "return_8",
    "body_pct",
    "ema_fast_distance",
    "ema_slow_distance",
    "ema_spread",
    "ema_fast",
    "ema_slow",
    "ema_fast_slope_3",
    "adx",
    "adx_slope",
    "plus_di",
    "minus_di",
    "atr",
    "atr_pct",
    "macd",
    "macd_signal",
    "macd_hist",
    "trend_strength",
    "bb_width",
    "bb_position",
    "rsi_centered",
    "volume_ratio",
    "range_position",
    "confirmation_bias",
    "context_bias",
    "orderbook_pressure",
    "orderbook_depth_pressure",
    "spread_bps",
    "mid_price_change_bps",
    "spread_change_bps",
    "imbalance_change",
    "bid_depth_change",
    "ask_depth_change",
    "bid_wall_closeness",
    "ask_wall_closeness",
    "bid_wall_strength",
    "ask_wall_strength",
    "trade_pressure",
    "trade_flow_imbalance",
    "trade_activity",
    "average_trade_size",
    "vwap_distance",
    "last_trade_distance",
    "last_trade_side",
    "position_side",
    "hour_sin",
    "hour_cos",
]


@dataclass(frozen=True)
class ClassBalanceConfig:
    enabled: bool = False
    train_two_stage: bool = False
    flat_downsample_ratio: float = 4.0
    use_class_weights: bool = True
    min_directional_examples: int = 100
    max_flat_share_after_balance: float = 0.70


@dataclass(frozen=True)
class FeedbackConfig:
    enabled: bool = False
    mode: str = "shadow"
    labels_path: Path = Path("data/labels/feedback_labels.jsonl")
    legacy_labels_csv: Path | None = Path("reports/labels.csv")
    oracle_labels_csv: Path | None = Path("reports/daily_oracle/latest_oracle_labels.csv")
    ensemble_enabled: bool = True
    ensemble_model_path: Path = Path("data/models/feedback_ensemble.joblib")
    ensemble_min_examples: int = 20
    min_examples: int = 8
    neighbors: int = 7
    block_similarity_threshold: float = 0.76
    opposite_similarity_threshold: float = 0.78
    confirmation_similarity_threshold: float = 0.82
    control_entry_threshold: float = 0.42
    control_exit_threshold: float = 0.50
    control_max_neutral_probability: float = 1.0
    control_min_directional_edge: float = 0.0
    control_block_neutral_runner_up: bool = True
    control_require_ensemble_model: bool = False
    control_require_schema_v2: bool = True
    control_require_promoted_model: bool = False
    control_min_promotion_score: float = 0.0
    control_required_entry_targets: tuple[str, ...] = ("long", "short", "flat")
    active_required_schema: int = 2
    required_heads: tuple[str, ...] = ("entry", "exit")
    candidate_can_trade: bool = False
    candidate_execution_entry_threshold: float = 0.25
    candidate_execution_max_neutral_probability: float = 0.45
    candidate_execution_min_directional_edge: float = 0.05
    active_can_trade_only_if_promoted: bool = True
    min_entry_examples: int = 0
    min_exit_examples: int = 0
    min_examples_per_class: int = 0
    max_class_share: float = 1.0
    promotion_min_oos_trades_total: int = 0
    promotion_min_oos_trades_per_fold: int = 0
    promotion_min_profit_factor_oos: float = 0.0
    promotion_min_profit_factor_median_fold: float = 0.0
    promotion_min_positive_folds_share: float = 0.0
    promotion_max_single_fold_drawdown_pct: float = 0.0
    promotion_max_total_oos_drawdown_pct: float = 0.0
    shadow_required_before_control: bool = False
    shadow_report_path: Path = Path("reports/shadow/shadow_evaluation.json")
    shadow_min_days: int = 10
    shadow_min_trade_signals: int = 50
    shadow_min_profit_factor: float = 1.15
    shadow_max_drawdown_pct: float = 0.03
    control_stop_atr_multiple: float = 1.5
    control_take_profit_r_multiple: float = 2.5
    exploration_enabled: bool = False
    exploration_entry_threshold: float = 0.20
    exploration_runner_up_enabled: bool = True
    exploration_runner_up_threshold: float = 0.20
    exploration_max_lots: int = 1
    generate_pnl_examples: bool = True
    pnl_timeframe: str = "1min"
    entry_label_horizon_bars: int = 12
    exit_label_horizon_bars: int = 6
    min_entry_net_pct: float = 0.10
    min_direction_edge_pct: float = 0.04
    hold_min_net_pct: float = 0.03
    decision_examples_enabled: bool = True
    promote_enabled: bool = True
    candidate_model_path: Path | None = None
    promotion_min_score: float = 0.0
    promotion_min_improvement: float = 0.0
    promotion_backtest_enabled: bool = True
    promotion_backtest_folds: int = 4
    promotion_backtest_max_candles: int = 1500
    promotion_min_trades: int = 3
    promotion_min_profit_factor: float = 1.0
    promotion_min_final_equity_pct: float = 0.0
    promotion_max_drawdown_abs_pct: float = 8.0
    promotion_min_backtest_improvement_pct: float = 0.25
    max_examples: int = 5000
    class_balance: ClassBalanceConfig = ClassBalanceConfig()

    @classmethod
    def from_runtime_config(cls, config) -> FeedbackConfig:
        raw = config.get("learning", default={}) or {}
        promotion = config.get("promotion", default={}) or {}
        training = config.get("training", default={}) or {}
        shadow = config.get("shadow", default={}) or {}
        class_balance = raw.get("class_balance", {}) or {}
        legacy_csv = raw.get("legacy_labels_csv", "reports/labels.csv")
        oracle_csv = raw.get("oracle_labels_csv", "reports/daily_oracle/latest_oracle_labels.csv")
        return cls(
            enabled=bool(raw.get("enabled", False)),
            mode=str(raw.get("mode", "shadow")),
            labels_path=Path(raw.get("labels_file", "data/labels/feedback_labels.jsonl")),
            legacy_labels_csv=Path(legacy_csv) if legacy_csv else None,
            oracle_labels_csv=Path(oracle_csv) if oracle_csv else None,
            ensemble_enabled=bool(raw.get("ensemble_enabled", True)),
            ensemble_model_path=Path(
                raw.get("ensemble_model_path", "data/models/feedback_ensemble.joblib")
            ),
            ensemble_min_examples=int(raw.get("ensemble_min_examples", 20)),
            min_examples=int(raw.get("min_examples", 8)),
            neighbors=int(raw.get("neighbors", 7)),
            block_similarity_threshold=float(raw.get("block_similarity_threshold", 0.76)),
            opposite_similarity_threshold=float(raw.get("opposite_similarity_threshold", 0.78)),
            confirmation_similarity_threshold=float(
                raw.get("confirmation_similarity_threshold", 0.82)
            ),
            control_entry_threshold=float(
                raw.get("entry_threshold", raw.get("control_entry_threshold", 0.42))
            ),
            control_exit_threshold=float(
                raw.get("exit_threshold", raw.get("control_exit_threshold", 0.50))
            ),
            control_max_neutral_probability=float(
                raw.get(
                    "max_neutral_probability",
                    raw.get("control_max_neutral_probability", 1.0),
                )
            ),
            control_min_directional_edge=float(
                raw.get(
                    "min_directional_edge",
                    raw.get("control_min_directional_edge", 0.0),
                )
            ),
            control_block_neutral_runner_up=bool(
                raw.get("control_block_neutral_runner_up", True)
            ),
            control_require_ensemble_model=bool(
                raw.get("control_require_ensemble_model", False)
            ),
            control_require_schema_v2=bool(raw.get("control_require_schema_v2", True)),
            control_require_promoted_model=bool(
                raw.get("control_require_promoted_model", False)
            ),
            control_min_promotion_score=float(raw.get("control_min_promotion_score", 0.0)),
            control_required_entry_targets=tuple(
                str(target)
                for target in raw.get(
                    "control_required_entry_targets",
                    promotion.get("required_entry_classes", ["long", "short", "flat"]),
                )
            ),
            active_required_schema=int(
                raw.get(
                    "active_required_schema",
                    promotion.get("required_schema_version", 2),
                )
            ),
            required_heads=tuple(
                str(head)
                for head in raw.get(
                    "required_heads",
                    promotion.get("required_heads", ["entry", "exit"]),
                )
            ),
            candidate_can_trade=bool(raw.get("candidate_can_trade", False)),
            candidate_execution_entry_threshold=float(
                raw.get("candidate_execution_entry_threshold", 0.25)
            ),
            candidate_execution_max_neutral_probability=float(
                raw.get("candidate_execution_max_neutral_probability", 0.45)
            ),
            candidate_execution_min_directional_edge=float(
                raw.get("candidate_execution_min_directional_edge", 0.05)
            ),
            active_can_trade_only_if_promoted=bool(
                raw.get("active_can_trade_only_if_promoted", True)
            ),
            min_entry_examples=int(
                raw.get(
                    "min_entry_examples",
                    promotion.get(
                        "min_entry_examples_total",
                        training.get("min_entry_examples", 0),
                    ),
                )
            ),
            min_exit_examples=int(
                raw.get(
                    "min_exit_examples",
                    promotion.get(
                        "min_exit_examples_total",
                        training.get("min_exit_examples", 0),
                    ),
                )
            ),
            min_examples_per_class=int(
                raw.get(
                    "min_examples_per_class",
                    promotion.get(
                        "min_examples_per_entry_class",
                        training.get("min_examples_per_class", 0),
                    ),
                )
            ),
            max_class_share=float(promotion.get("max_class_share", 1.0)),
            promotion_min_oos_trades_total=int(
                promotion.get("min_oos_trades_total", raw.get("promotion_min_trades", 0))
            ),
            promotion_min_oos_trades_per_fold=int(
                promotion.get("min_oos_trades_per_fold", 0)
            ),
            promotion_min_profit_factor_oos=float(
                promotion.get("min_profit_factor_oos", raw.get("promotion_min_profit_factor", 0.0))
            ),
            promotion_min_profit_factor_median_fold=float(
                promotion.get("min_profit_factor_median_fold", 0.0)
            ),
            promotion_min_positive_folds_share=float(
                promotion.get("min_positive_folds_share", 0.0)
            ),
            promotion_max_single_fold_drawdown_pct=float(
                promotion.get("max_single_fold_drawdown_pct", 0.0)
            ),
            promotion_max_total_oos_drawdown_pct=float(
                promotion.get(
                    "max_total_oos_drawdown_pct",
                    raw.get("promotion_max_drawdown_abs_pct", 0.0),
                )
            ),
            shadow_required_before_control=bool(shadow.get("required_before_control", False)),
            shadow_report_path=Path(
                shadow.get("report_file", "reports/shadow/shadow_evaluation.json")
            ),
            shadow_min_days=int(shadow.get("min_days", 10)),
            shadow_min_trade_signals=int(shadow.get("min_trade_signals", 50)),
            shadow_min_profit_factor=float(shadow.get("min_profit_factor", 1.15)),
            shadow_max_drawdown_pct=float(shadow.get("max_drawdown_pct", 0.03)),
            control_stop_atr_multiple=float(raw.get("control_stop_atr_multiple", 1.5)),
            control_take_profit_r_multiple=float(
                raw.get("control_take_profit_r_multiple", 2.5)
            ),
            exploration_enabled=bool(raw.get("exploration_enabled", False)),
            exploration_entry_threshold=float(raw.get("exploration_entry_threshold", 0.20)),
            exploration_runner_up_enabled=bool(raw.get("exploration_runner_up_enabled", True)),
            exploration_runner_up_threshold=float(raw.get("exploration_runner_up_threshold", 0.20)),
            exploration_max_lots=int(raw.get("exploration_max_lots", 1)),
            generate_pnl_examples=bool(raw.get("generate_pnl_examples", True)),
            pnl_timeframe=str(raw.get("pnl_timeframe", "1min")),
            entry_label_horizon_bars=int(raw.get("entry_label_horizon_bars", 12)),
            exit_label_horizon_bars=int(raw.get("exit_label_horizon_bars", 6)),
            min_entry_net_pct=float(raw.get("min_entry_net_pct", 0.10)),
            min_direction_edge_pct=float(raw.get("min_direction_edge_pct", 0.04)),
            hold_min_net_pct=float(raw.get("hold_min_net_pct", 0.03)),
            decision_examples_enabled=bool(raw.get("decision_examples_enabled", True)),
            promote_enabled=bool(raw.get("promote_enabled", True)),
            candidate_model_path=(
                Path(raw["candidate_model_path"]) if raw.get("candidate_model_path") else None
            ),
            promotion_min_score=float(raw.get("promotion_min_score", 0.0)),
            promotion_min_improvement=float(raw.get("promotion_min_improvement", 0.0)),
            promotion_backtest_enabled=bool(raw.get("promotion_backtest_enabled", True)),
            promotion_backtest_folds=int(raw.get("promotion_backtest_folds", 4)),
            promotion_backtest_max_candles=int(raw.get("promotion_backtest_max_candles", 1500)),
            promotion_min_trades=int(raw.get("promotion_min_trades", 3)),
            promotion_min_profit_factor=float(raw.get("promotion_min_profit_factor", 1.0)),
            promotion_min_final_equity_pct=float(
                raw.get("promotion_min_final_equity_pct", 0.0)
            ),
            promotion_max_drawdown_abs_pct=float(
                raw.get("promotion_max_drawdown_abs_pct", 8.0)
            ),
            promotion_min_backtest_improvement_pct=float(
                raw.get("promotion_min_backtest_improvement_pct", 0.25)
            ),
            max_examples=int(raw.get("max_examples", 5000)),
            class_balance=ClassBalanceConfig(
                enabled=bool(class_balance.get("enabled", False)),
                train_two_stage=bool(class_balance.get("train_two_stage", False)),
                flat_downsample_ratio=float(class_balance.get("flat_downsample_ratio", 4.0)),
                use_class_weights=bool(class_balance.get("use_class_weights", True)),
                min_directional_examples=int(class_balance.get("min_directional_examples", 100)),
                max_flat_share_after_balance=float(
                    class_balance.get("max_flat_share_after_balance", 0.70)
                ),
            ),
        )


@dataclass(frozen=True)
class FeedbackExample:
    label: str
    target: str
    features: dict[str, float]
    source: str
    timestamp: str | None = None
    task: str = "entry"
    pnl_pct: float = 0.0
    outcomes: dict[str, float] | None = None
    feature_complete: bool = True
    label_matured: bool = True
    market_data_trusted: bool = True
    reject_reason: str | None = None


@dataclass(frozen=True)
class FeedbackPrediction:
    target: str
    score: float
    examples: int
    nearest: list[dict[str, Any]]
    reason: str
    alternatives: list[dict[str, Any]] | None = None

    def as_metadata(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "score": round(self.score, 4),
            "examples": self.examples,
            "nearest": self.nearest,
            "reason": self.reason,
            "alternatives": self.alternatives or [],
        }


class FeedbackModel:
    def __init__(self, config: FeedbackConfig, examples: list[FeedbackExample]):
        self.config = config
        self.examples = examples[: config.max_examples]
        self.ensemble = _load_ensemble(config.ensemble_model_path) if config.ensemble_enabled else None
        self.candidate_ensemble = (
            _load_ensemble(config.candidate_model_path)
            if config.ensemble_enabled
            and config.candidate_model_path is not None
            and config.candidate_model_path != config.ensemble_model_path
            else None
        )

    @classmethod
    def from_runtime_config(cls, config) -> FeedbackModel:
        feedback_config = FeedbackConfig.from_runtime_config(config)
        examples = load_feedback_examples(feedback_config)
        return cls(feedback_config, examples)

    @property
    def active(self) -> bool:
        return self.config.enabled and self.config.mode != "off"

    def control_model_validation(self) -> tuple[bool, str, dict[str, Any]]:
        if self.config.mode not in CONTROL_MODES:
            return True, "not_control_mode", {}
        if not self.config.control_require_ensemble_model:
            return True, "strict_ensemble_not_required", {}
        if self.ensemble is None:
            return False, "ensemble_model_missing_or_invalid", {}

        schema_version = int(getattr(self.ensemble, "schema_version", 1))
        heads = getattr(self.ensemble, "heads", {}) or {}
        entry_head = heads.get("entry")
        metadata = getattr(self.ensemble, "metadata", {}) or {}
        promotion_metrics = getattr(self.ensemble, "promotion_metrics", {}) or {}
        promotion_status = str(
            getattr(self.ensemble, "promotion_status", None)
            or metadata.get("promotion_status")
            or ""
        ).lower()
        model_status = str(
            getattr(self.ensemble, "model_status", None)
            or metadata.get("model_status")
            or "unknown"
        ).lower()
        details = {
            "schema_version": schema_version,
            "heads": sorted(heads.keys()),
            "promotion_score": getattr(self.ensemble, "promotion_score", None),
            "promotion_status": promotion_status or None,
            "model_status": model_status,
            "trained_at": getattr(self.ensemble, "trained_at", None),
            "feature_schema_version": getattr(self.ensemble, "feature_schema_version", None),
            "promotion_metrics": promotion_metrics,
        }
        required_schema = max(
            int(self.config.active_required_schema),
            2 if self.config.control_require_schema_v2 else 1,
        )
        if schema_version < required_schema:
            return False, f"ensemble_schema_below_v{required_schema}", details

        missing_heads = [
            head
            for head in self.config.required_heads
            if not isinstance(heads.get(head), dict)
        ]
        if missing_heads:
            details["missing_heads"] = missing_heads
            return False, "missing_required_heads", details
        if not isinstance(entry_head, dict):
            return False, "entry_head_missing", details

        entry_classes = {str(item) for item in entry_head.get("classes") or []}
        details["entry_classes"] = sorted(entry_classes)
        missing_classes = [
            target
            for target in self.config.control_required_entry_targets
            if target not in entry_classes
        ]
        if missing_classes:
            details["missing_entry_classes"] = missing_classes
            return False, "entry_head_missing_required_classes", details

        entry_examples = int(entry_head.get("examples", 0))
        details["entry_examples"] = entry_examples
        min_entry_examples = max(self.config.ensemble_min_examples, self.config.min_entry_examples)
        if entry_examples < min_entry_examples:
            details["min_entry_examples"] = min_entry_examples
            return False, "entry_head_not_enough_examples", details

        if self.config.min_examples_per_class > 0:
            class_counts = _class_counts(entry_head)
            details["entry_class_counts"] = class_counts
            missing_class_examples = {
                target: class_counts.get(target, 0)
                for target in self.config.control_required_entry_targets
                if class_counts.get(target, 0) < self.config.min_examples_per_class
            }
            if missing_class_examples:
                details["missing_class_examples"] = missing_class_examples
                details["min_examples_per_class"] = self.config.min_examples_per_class
                return False, "entry_head_not_enough_class_examples", details

        if "exit" in self.config.required_heads:
            exit_head = heads.get("exit") or {}
            exit_examples = int(exit_head.get("examples", 0))
            details["exit_examples"] = exit_examples
            min_exit_examples = max(self.config.ensemble_min_examples, self.config.min_exit_examples)
            if exit_examples < min_exit_examples:
                details["min_exit_examples"] = min_exit_examples
                return False, "exit_head_not_enough_examples", details

        if self.config.control_require_promoted_model or self.config.active_can_trade_only_if_promoted:
            if promotion_status != APPROVED_PROMOTION_STATUS:
                return False, "model_not_promoted", details
            missing_metadata = _missing_model_metadata(self.ensemble)
            if missing_metadata:
                details["missing_metadata"] = missing_metadata
                return False, "model_metadata_incomplete", details
            promotion_score = getattr(self.ensemble, "promotion_score", None)
            if promotion_score is None:
                return False, "promotion_score_missing", details
            if float(promotion_score) < self.config.control_min_promotion_score:
                return False, "promotion_score_below_min", details
            metrics_ok, metrics_reason = _promotion_metrics_eligible(
                promotion_metrics,
                self.config,
            )
            if not metrics_ok:
                return False, metrics_reason, details

        return True, "control_model_ready", details

    def apply(
        self,
        signal: Signal,
        *,
        execution_df: pd.DataFrame,
        confirmation_df: pd.DataFrame,
        context_df: pd.DataFrame,
        orderbook: OrderBookFeatures,
        trade_flow: TradeFlowFeatures,
        now: datetime,
    ) -> Signal:
        if not self.active:
            return signal

        features = build_feature_snapshot(
            execution_df=execution_df,
            confirmation_df=confirmation_df,
            context_df=context_df,
            orderbook=orderbook,
            trade_flow=trade_flow,
            now=now,
        )
        prediction = self.predict(features, task="entry")
        metadata = {
            **signal.metadata,
            "feedback": prediction.as_metadata(),
        }

        if self.config.mode == "shadow" or prediction.target == "unknown":
            return _with_metadata(signal, metadata)

        if signal.side == Side.FLAT:
            return _with_metadata(signal, metadata)

        if _is_sideways_target(prediction.target) and (
            prediction.score >= self.config.block_similarity_threshold
        ):
            return Signal(
                side=Side.FLAT,
                confidence=0.0,
                reason=f"feedback_blocked_{prediction.target}:{prediction.score:.2f}",
                price=signal.price,
                stop_price=None,
                timestamp=signal.timestamp,
                metadata=metadata,
            )

        if _is_exit_target(prediction.target) and (
            prediction.score >= self.config.block_similarity_threshold
        ):
            return Signal(
                side=Side.FLAT,
                confidence=0.0,
                reason=f"feedback_blocked_{prediction.target}:{prediction.score:.2f}",
                price=signal.price,
                stop_price=None,
                timestamp=signal.timestamp,
                metadata=metadata,
            )

        if _opposes_signal(prediction.target, signal.side) and (
            prediction.score >= self.config.opposite_similarity_threshold
        ):
            return Signal(
                side=Side.FLAT,
                confidence=0.0,
                reason=f"feedback_blocked_opposite:{prediction.target}:{prediction.score:.2f}",
                price=signal.price,
                stop_price=None,
                timestamp=signal.timestamp,
                metadata=metadata,
            )

        if _matches_signal(prediction.target, signal.side) and (
            prediction.score >= self.config.confirmation_similarity_threshold
        ):
            return Signal(
                side=signal.side,
                confidence=min(1.0, signal.confidence + min(0.08, prediction.score * 0.05)),
                reason=f"{signal.reason}|feedback_confirmed:{prediction.target}:{prediction.score:.2f}",
                price=signal.price,
                stop_price=signal.stop_price,
                timestamp=signal.timestamp,
                take_profit1=signal.take_profit1,
                take_profit2=signal.take_profit2,
                metadata=metadata,
            )

        return _with_metadata(signal, metadata)

    def predict(self, features: dict[str, float], *, task: str = "entry") -> FeedbackPrediction:
        if self.config.mode in CONTROL_MODES and self.config.control_require_ensemble_model:
            ready, reason, details = self.control_model_validation()
            if not ready:
                return FeedbackPrediction(
                    target="unknown",
                    score=0.0,
                    examples=int(details.get("entry_examples", 0) or 0),
                    nearest=[],
                    reason=f"control_model_not_ready:{reason}",
                )

        if self.ensemble is not None:
            prediction = self.ensemble.predict(features, task=task)
            if prediction.target != "unknown":
                return FeedbackPrediction(
                    target=prediction.target,
                    score=prediction.score,
                    examples=prediction.examples,
                    nearest=[
                        {"model": name, "score": score}
                        for name, score in prediction.model_scores.items()
                    ],
                    reason=prediction.reason,
                    alternatives=_alternatives_from_probabilities(prediction.probabilities),
                )

        task_examples = [example for example in self.examples if example.task == task]
        if len(task_examples) < self.config.min_examples:
            return FeedbackPrediction(
                target="unknown",
                score=0.0,
                examples=len(task_examples),
                nearest=[],
                reason="not_enough_feedback_examples",
            )

        scored = []
        for example in task_examples:
            similarity = feature_similarity(features, example.features)
            scored.append((similarity, example))

        scored.sort(key=lambda item: item[0], reverse=True)
        nearest = scored[: max(1, self.config.neighbors)]
        totals: dict[str, float] = {}
        for similarity, example in nearest:
            totals[example.target] = totals.get(example.target, 0.0) + similarity

        if not totals:
            return FeedbackPrediction("unknown", 0.0, len(task_examples), [], "no_feature_overlap")

        target, weighted_score = max(totals.items(), key=lambda item: item[1])
        total_weight = sum(totals.values()) or 1.0
        confidence = weighted_score / total_weight
        strongest_similarity = nearest[0][0] if nearest else 0.0
        score = confidence * strongest_similarity
        probabilities = {
            target: weight / total_weight
            for target, weight in totals.items()
        }
        return FeedbackPrediction(
            target=target,
            score=score,
            examples=len(task_examples),
            nearest=[
                {
                    "label": example.label,
                    "target": example.target,
                    "similarity": round(similarity, 4),
                    "source": example.source,
                    "timestamp": example.timestamp,
                }
                for similarity, example in nearest[:3]
            ],
            reason="nearest_feedback_examples",
            alternatives=_alternatives_from_probabilities(probabilities),
        )

    def candidate_shadow_prediction(
        self,
        features: dict[str, float],
        *,
        task: str = "entry",
    ) -> dict[str, Any] | None:
        if self.candidate_ensemble is None:
            return None
        prediction = self.candidate_ensemble.predict(features, task=task)
        return {
            "target": prediction.target,
            "score": round(float(prediction.score), 4),
            "examples": int(prediction.examples),
            "reason": prediction.reason,
            "alternatives": _alternatives_from_probabilities(prediction.probabilities),
            "model_status": str(
                getattr(self.candidate_ensemble, "model_status", None) or "candidate"
            ),
            "promotion_status": str(
                getattr(self.candidate_ensemble, "promotion_status", None) or "candidate"
            ),
            "schema_version": int(getattr(self.candidate_ensemble, "schema_version", 1)),
            "can_trade": bool(self.config.candidate_can_trade),
        }

    def signal_from_prediction(
        self,
        *,
        execution_df: pd.DataFrame,
        confirmation_df: pd.DataFrame,
        context_df: pd.DataFrame,
        orderbook: OrderBookFeatures,
        trade_flow: TradeFlowFeatures,
        now: datetime,
        price: float,
        allow_candidate: bool = False,
    ) -> Signal:
        features = build_feature_snapshot(
            execution_df=execution_df,
            confirmation_df=confirmation_df,
            context_df=context_df,
            orderbook=orderbook,
            trade_flow=trade_flow,
            now=now,
        )
        shadow_prediction = self.candidate_shadow_prediction(features, task="entry")
        model_ready, model_reason, model_details = self.control_model_validation()
        candidate_execution = bool(
            allow_candidate
            and self.config.candidate_can_trade
            and self.candidate_ensemble is not None
        )
        if not model_ready and not candidate_execution:
            metadata = {
                "strategy": "ml_control",
                "model_validation": {
                    "ready": False,
                    "reason": model_reason,
                    **model_details,
                },
                "candidate_shadow": shadow_prediction,
                "features": {key: round(value, 6) for key, value in features.items()},
            }
            return Signal(
                Side.FLAT,
                0.0,
                f"ml_control_model_not_ready:{model_reason}",
                price,
                None,
                now,
                metadata=metadata,
            )

        prediction = (
            self._candidate_prediction(features, task="entry")
            if candidate_execution
            else self.predict(features, task="entry")
        )
        metadata = {
            "strategy": "ml_candidate_paper" if candidate_execution else "ml_control",
            "candidate_execution": candidate_execution,
            "model_validation": {
                "ready": model_ready,
                "reason": model_reason,
                **model_details,
            },
            "feedback": prediction.as_metadata(),
            "candidate_shadow": shadow_prediction,
            "features": {key: round(value, 6) for key, value in features.items()},
        }
        side = _entry_target_side(prediction.target)
        exploration_target = prediction.target
        exploration_score = prediction.score
        if prediction.target in WEAK_ENTRY_TARGETS and self.config.control_block_neutral_runner_up:
            return Signal(
                Side.FLAT,
                prediction.score,
                f"ml_no_entry_weak_or_neutral:{prediction.target}:{prediction.score:.2f}",
                price,
                None,
                now,
                metadata=metadata,
            )
        if side is None and self.config.exploration_runner_up_enabled:
            runner_up = _runner_up_directional_target(prediction.alternatives or [])
            if runner_up is not None and runner_up["score"] >= self.config.exploration_runner_up_threshold:
                exploration_target = str(runner_up["target"])
                exploration_score = float(runner_up["score"])
                side = _entry_target_side(exploration_target)
        if side is None or (
            prediction.target in WEAK_ENTRY_TARGETS
            and exploration_target == prediction.target
        ):
            return Signal(
                Side.FLAT,
                prediction.score,
                f"ml_no_entry:{prediction.target}:{prediction.score:.2f}",
                price,
                None,
                now,
                metadata=metadata,
            )

        neutral_score = _max_alternative_score(
            prediction.alternatives or [],
            SIDEWAYS_TARGETS | EXIT_TARGETS | WEAK_ENTRY_TARGETS,
        )
        opposite_score = _max_alternative_score(
            prediction.alternatives or [],
            {"short"} if side == Side.LONG else {"long"},
        )
        directional_edge = prediction.score - max(neutral_score, opposite_score)
        metadata = {
            **metadata,
            "neutral_score": round(neutral_score, 4),
            "opposite_score": round(opposite_score, 4),
            "directional_edge": round(directional_edge, 4),
        }
        max_neutral_probability = (
            self.config.candidate_execution_max_neutral_probability
            if candidate_execution
            else self.config.control_max_neutral_probability
        )
        min_directional_edge = (
            self.config.candidate_execution_min_directional_edge
            if candidate_execution
            else self.config.control_min_directional_edge
        )
        entry_threshold = (
            self.config.candidate_execution_entry_threshold
            if candidate_execution
            else self.config.control_entry_threshold
        )
        if neutral_score > max_neutral_probability:
            return Signal(
                Side.FLAT,
                prediction.score,
                (
                    f"ml_no_entry_neutral_risk:{prediction.target}:"
                    f"{prediction.score:.2f}:neutral={neutral_score:.2f}"
                ),
                price,
                None,
                now,
                metadata=metadata,
            )
        if directional_edge < min_directional_edge:
            return Signal(
                Side.FLAT,
                prediction.score,
                (
                    f"ml_no_entry_directional_edge:{prediction.target}:"
                    f"{prediction.score:.2f}:edge={directional_edge:.2f}"
                ),
                price,
                None,
                now,
                metadata=metadata,
            )
        is_primary = (
            prediction.target == exploration_target
            and prediction.score >= entry_threshold
        )
        is_exploration = (
            not is_primary
            and self.config.exploration_enabled
            and exploration_score >= self.config.exploration_entry_threshold
        )
        if not is_primary and not is_exploration:
            return Signal(
                Side.FLAT,
                prediction.score,
                f"ml_no_entry:{prediction.target}:{prediction.score:.2f}",
                price,
                None,
                now,
                metadata=metadata,
            )

        if is_exploration:
            metadata = {
                **metadata,
                "exploration": True,
                "exploration_target": exploration_target,
                "exploration_score": round(exploration_score, 4),
                "primary_entry_threshold": entry_threshold,
                "exploration_entry_threshold": self.config.exploration_entry_threshold,
            }

        risk_distance = self._risk_distance(execution_df, context_df, price)
        if side == Side.LONG:
            stop_price = price - risk_distance
            take_profit2 = price + risk_distance * self.config.control_take_profit_r_multiple
        else:
            stop_price = price + risk_distance
            take_profit2 = price - risk_distance * self.config.control_take_profit_r_multiple

        return Signal(
            side,
            prediction.score,
            (
                f"ml_explore_entry:{exploration_target}:{exploration_score:.2f}"
                if is_exploration
                else f"ml_entry:{prediction.target}:{prediction.score:.2f}"
            ),
            price,
            stop_price,
            now,
            take_profit2=take_profit2,
            metadata=metadata,
        )

    def _candidate_prediction(
        self, features: dict[str, float], *, task: str
    ) -> FeedbackPrediction:
        if self.candidate_ensemble is None:
            return FeedbackPrediction("unknown", 0.0, 0, [], "candidate_model_missing")
        prediction = self.candidate_ensemble.predict(features, task=task)
        return FeedbackPrediction(
            target=prediction.target,
            score=prediction.score,
            examples=prediction.examples,
            nearest=[
                {"model": name, "score": score}
                for name, score in prediction.model_scores.items()
            ],
            reason=prediction.reason,
            alternatives=_alternatives_from_probabilities(prediction.probabilities),
        )

    def exit_reason_from_prediction(
        self,
        *,
        position_side: Side,
        execution_df: pd.DataFrame,
        confirmation_df: pd.DataFrame,
        context_df: pd.DataFrame,
        orderbook: OrderBookFeatures,
        trade_flow: TradeFlowFeatures,
        now: datetime,
        allow_candidate: bool = False,
    ) -> tuple[str | None, FeedbackPrediction]:
        features = build_feature_snapshot(
            execution_df=execution_df,
            confirmation_df=confirmation_df,
            context_df=context_df,
            orderbook=orderbook,
            trade_flow=trade_flow,
            now=now,
        )
        features["position_side"] = _position_side_value(position_side)
        candidate_execution = bool(
            allow_candidate
            and self.config.candidate_can_trade
            and self.candidate_ensemble is not None
        )
        prediction = (
            self._candidate_prediction(features, task="exit")
            if candidate_execution
            else self.predict(features, task="exit")
        )
        if prediction.target == "unknown" and prediction.reason == "no_exit_head":
            prediction = (
                self._candidate_prediction(features, task="entry")
                if candidate_execution
                else self.predict(features, task="entry")
            )
        if prediction.target == "unknown" or prediction.score < self.config.control_exit_threshold:
            return None, prediction
        if prediction.target == "exit":
            return f"ml_exit:{prediction.target}:{prediction.score:.2f}", prediction
        if prediction.target == "hold":
            return None, prediction
        if _is_exit_target(prediction.target) or _is_sideways_target(prediction.target):
            return f"ml_exit:{prediction.target}:{prediction.score:.2f}", prediction
        if _opposes_signal(prediction.target, position_side):
            return f"ml_exit_opposite:{prediction.target}:{prediction.score:.2f}", prediction
        return None, prediction

    def _risk_distance(
        self,
        execution_df: pd.DataFrame,
        context_df: pd.DataFrame,
        price: float,
    ) -> float:
        atr_value = 0.0
        for frame in (context_df, execution_df):
            if frame is not None and not frame.empty and "atr" in frame.columns:
                atr_value = _safe_float(frame.iloc[-1].get("atr"))
                if atr_value > 0:
                    break
        if atr_value <= 0:
            atr_value = max(price * 0.002, 0.001)
        return max(atr_value * self.config.control_stop_atr_multiple, 0.001)


def load_feedback_examples(config: FeedbackConfig) -> list[FeedbackExample]:
    examples: list[FeedbackExample] = []
    examples.extend(_load_jsonl_examples(config.labels_path))
    if config.legacy_labels_csv is not None:
        examples.extend(_load_csv_examples(config.legacy_labels_csv))
    return examples


def _load_ensemble(path: Path):
    if not path.exists():
        return None
    try:
        from ngn6_bot.learning.ensemble import FeedbackEnsemble

        return FeedbackEnsemble.load(path)
    except Exception:
        return None


def build_feature_snapshot(
    *,
    execution_df: pd.DataFrame,
    confirmation_df: pd.DataFrame,
    context_df: pd.DataFrame,
    orderbook: OrderBookFeatures,
    trade_flow: TradeFlowFeatures,
    now: datetime,
) -> dict[str, float]:
    execution_df = _closed_frame(execution_df, now)
    confirmation_df = _closed_frame(confirmation_df, now)
    context_df = _closed_frame(context_df, now)
    if execution_df.empty:
        return _time_features(now)

    row = execution_df.iloc[-1]
    previous_row = execution_df.iloc[-2] if len(execution_df) >= 2 else row
    close = _safe_float(row.get("close"))
    open_ = _safe_float(row.get("open"))
    high = _safe_float(row.get("high"))
    low = _safe_float(row.get("low"))
    volume = _safe_float(row.get("volume"))
    volume_ma = _safe_float(row.get("volume_ma"))
    ema_fast = _safe_float(row.get("ema_fast"))
    ema_slow = _safe_float(row.get("ema_slow"))
    adx_value = _safe_float(row.get("adx"))
    previous_adx = _safe_float(previous_row.get("adx"), adx_value)
    plus_di = _safe_float(row.get("plus_di"))
    minus_di = _safe_float(row.get("minus_di"))
    atr_value = _safe_float(row.get("atr"))
    macd_value = _safe_float(row.get("macd"))
    macd_signal = _safe_float(row.get("macd_signal"))
    macd_hist = _safe_float(row.get("macd_hist"))
    bb_upper = _safe_float(row.get("bb_upper"))
    bb_lower = _safe_float(row.get("bb_lower"))
    bb_width = _safe_float(row.get("bb_width_pct"))
    rsi = _safe_float(row.get("rsi"), 50.0)
    recent = execution_df.tail(20)
    recent_high = _safe_float(recent["high"].max() if not recent.empty else high)
    recent_low = _safe_float(recent["low"].min() if not recent.empty else low)

    features = {
        "return_1": _normalized_return(execution_df, 1),
        "return_3": _normalized_return(execution_df, 3),
        "return_8": _normalized_return(execution_df, 8),
        "body_pct": _clip(_pct(open_, close), -0.04, 0.04) / 0.04,
        "ema_fast_distance": _clip(_pct(ema_fast, close), -0.03, 0.03) / 0.03,
        "ema_slow_distance": _clip(_pct(ema_slow, close), -0.04, 0.04) / 0.04,
        "ema_spread": _clip(_pct(ema_slow, ema_fast), -0.025, 0.025) / 0.025,
        "ema_fast": _clip(_pct(close, ema_fast), -0.04, 0.04) / 0.04,
        "ema_slow": _clip(_pct(close, ema_slow), -0.04, 0.04) / 0.04,
        "ema_fast_slope_3": _normalized_slope(execution_df, "ema_fast", 3),
        "adx": _clip(adx_value / 100.0, 0.0, 1.0),
        "adx_slope": _clip((adx_value - previous_adx) / 100.0, -1.0, 1.0),
        "plus_di": _clip(plus_di / 100.0, 0.0, 1.0),
        "minus_di": _clip(minus_di / 100.0, 0.0, 1.0),
        "atr": _clip(atr_value / max(close, 1e-12), 0.0, 0.05) / 0.05,
        "atr_pct": _clip(_pct(close, close + atr_value), 0.0, 0.05) / 0.05,
        "macd": _clip(macd_value / max(close, 1e-12), -0.03, 0.03) / 0.03,
        "macd_signal": _clip(macd_signal / max(close, 1e-12), -0.03, 0.03) / 0.03,
        "macd_hist": _clip(macd_hist / max(close, 1e-12), -0.02, 0.02) / 0.02,
        "trend_strength": _clip(abs(ema_fast - ema_slow) / max(close, 1e-12), 0.0, 0.05) / 0.05,
        "bb_width": _clip(bb_width / 4.0, 0.0, 2.0) / 2.0,
        "bb_position": _range_position(close, bb_lower, bb_upper) * 2 - 1,
        "rsi_centered": _clip((rsi - 50.0) / 50.0, -1.0, 1.0),
        "volume_ratio": _clip(_ratio(volume, volume_ma, 1.0) - 1.0, -1.0, 3.0) / 3.0,
        "range_position": _range_position(close, recent_low, recent_high) * 2 - 1,
        "confirmation_bias": _frame_bias(confirmation_df),
        "context_bias": _frame_bias(context_df),
        "orderbook_pressure": _clip((orderbook.bid_ask_imbalance - 0.5) * 2, -1.0, 1.0),
        "orderbook_depth_pressure": _clip(orderbook.depth_pressure, -1.0, 1.0),
        "spread_bps": _clip(_safe_float(orderbook.spread_bps) / 40.0, 0.0, 2.0) / 2.0,
        "mid_price_change_bps": _clip(
            _safe_float(orderbook.mid_price_change_bps) / 20.0, -1.0, 1.0
        ),
        "spread_change_bps": _clip(
            _safe_float(orderbook.spread_change_bps) / 20.0, -1.0, 1.0
        ),
        "imbalance_change": _clip(orderbook.imbalance_change, -1.0, 1.0),
        "bid_depth_change": _clip(
            _safe_float(orderbook.bid_depth_change_pct) / 100.0, -1.0, 1.0
        ),
        "ask_depth_change": _clip(
            _safe_float(orderbook.ask_depth_change_pct) / 100.0, -1.0, 1.0
        ),
        "bid_wall_closeness": _wall_closeness(orderbook.bid_wall_distance_bps),
        "ask_wall_closeness": _wall_closeness(orderbook.ask_wall_distance_bps),
        "bid_wall_strength": _wall_strength(orderbook.bid_wall_notional, close),
        "ask_wall_strength": _wall_strength(orderbook.ask_wall_notional, close),
        "trade_pressure": _clip((trade_flow.buy_ratio - 0.5) * 2, -1.0, 1.0),
        "trade_flow_imbalance": _clip(trade_flow.buy_sell_imbalance, -1.0, 1.0),
        "trade_activity": _clip(float(trade_flow.trade_count) / 60.0, 0.0, 2.0) / 2.0,
        "average_trade_size": _clip(trade_flow.average_trade_size / 100.0, 0.0, 2.0) / 2.0,
        "vwap_distance": _clip(_pct(_safe_float(trade_flow.vwap), close), -0.02, 0.02) / 0.02,
        "last_trade_distance": _clip(
            _pct(_safe_float(trade_flow.last_trade_price), close), -0.02, 0.02
        )
        / 0.02,
        "last_trade_side": _trade_side_value(trade_flow.last_trade_side),
        "position_side": 0.0,
        **_time_features(now),
    }
    return {key: float(value) for key, value in features.items() if math.isfinite(float(value))}


def _closed_frame(frame: pd.DataFrame, now: datetime) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    if not isinstance(frame.index, pd.DatetimeIndex):
        return frame
    timestamp = pd.Timestamp(now)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    index = frame.index
    if index.tz is None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    else:
        timestamp = timestamp.tz_convert(index.tz)
    return frame.loc[index <= timestamp]


def feature_similarity(left: dict[str, float], right: dict[str, float]) -> float:
    keys = [key for key in FEATURE_KEYS if key in left and key in right]
    if not keys:
        return 0.0
    distance = math.sqrt(sum((left[key] - right[key]) ** 2 for key in keys) / len(keys))
    return 1.0 / (1.0 + distance)


def annotation_payload(
    *,
    timestamp: datetime,
    label: str,
    timeframe: str,
    features: dict[str, float],
    source_image: str,
    notes: str = "",
) -> dict[str, Any]:
    return {
        "timestamp": timestamp.astimezone(timezone.utc).isoformat(),
        "label": label,
        "target": label_to_target(label),
        "timeframe": timeframe,
        "features": {key: round(value, 6) for key, value in features.items()},
        "source_image": source_image,
        "notes": notes,
    }


def label_to_target(label: str) -> str:
    normalized = str(label).strip().upper()
    return LABEL_TARGETS.get(normalized, "unknown")


def _load_jsonl_examples(path: Path) -> list[FeedbackExample]:
    if not path.exists():
        return []
    examples: list[FeedbackExample] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            example = _example_from_payload(payload, source=str(path))
            if example is not None:
                examples.append(example)
    return examples


def _load_csv_examples(path: Path) -> list[FeedbackExample]:
    if not path.exists():
        return []
    examples: list[FeedbackExample] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            features = {
                key.removeprefix("feature_"): _safe_float(value)
                for key, value in row.items()
                if key.startswith("feature_") and value not in {None, ""}
            }
            payload = {
                "label": row.get("label"),
                "target": row.get("target"),
                "timestamp": row.get("timestamp") or row.get("date"),
                "features": features,
            }
            example = _example_from_payload(payload, source=str(path))
            if example is not None:
                examples.append(example)
    return examples


def _example_from_payload(payload: dict[str, Any], source: str) -> FeedbackExample | None:
    label = str(payload.get("label") or "").strip().upper()
    target = str(payload.get("target") or label_to_target(label)).strip().lower()
    target = _coarse_target(target)
    task = str(payload.get("task") or _task_for_target(target)).strip().lower()
    features = payload.get("features") or payload.get("feature_snapshot") or {}
    if target == "unknown" or not isinstance(features, dict):
        return None
    parsed_features = {
        key: _safe_float(value)
        for key, value in features.items()
        if key in FEATURE_KEYS and math.isfinite(_safe_float(value, math.nan))
    }
    if not parsed_features:
        return None
    outcomes = payload.get("outcomes")
    parsed_outcomes = None
    if isinstance(outcomes, dict):
        parsed_outcomes = {
            str(key): _safe_float(value)
            for key, value in outcomes.items()
            if math.isfinite(_safe_float(value, math.nan))
        }
    return FeedbackExample(
        label=label,
        target=target,
        features=parsed_features,
        source=source,
        timestamp=str(payload.get("timestamp") or "") or None,
        task=task,
        pnl_pct=_safe_float(payload.get("pnl_pct")),
        outcomes=parsed_outcomes,
        feature_complete=bool(payload.get("feature_complete", True)),
        label_matured=bool(payload.get("label_matured", payload.get("matured", True))),
        market_data_trusted=bool(payload.get("market_data_trusted", True)),
        reject_reason=(
            str(payload.get("reject_reason"))
            if payload.get("reject_reason") is not None
            else None
        ),
    )


def _with_metadata(signal: Signal, metadata: dict[str, Any]) -> Signal:
    return Signal(
        side=signal.side,
        confidence=signal.confidence,
        reason=signal.reason,
        price=signal.price,
        stop_price=signal.stop_price,
        timestamp=signal.timestamp,
        take_profit1=signal.take_profit1,
        take_profit2=signal.take_profit2,
        metadata=metadata,
    )


def _matches_signal(target: str, side: Side) -> bool:
    target_side = _target_side(target)
    return target_side is not None and target_side == side


def _opposes_signal(target: str, side: Side) -> bool:
    target_side = _target_side(target)
    return target_side is not None and target_side != side


def _target_side(target: str) -> Side | None:
    if target in LONG_TARGETS:
        return Side.LONG
    if target in SHORT_TARGETS:
        return Side.SHORT
    return None


def _entry_target_side(target: str) -> Side | None:
    if target == "long":
        return Side.LONG
    if target == "short":
        return Side.SHORT
    if target in WEAK_ENTRY_TARGETS:
        return None
    return _target_side(target)


def _is_sideways_target(target: str) -> bool:
    return target in SIDEWAYS_TARGETS


def _is_exit_target(target: str) -> bool:
    return target in EXIT_TARGETS


def _task_for_target(target: str) -> str:
    if target in EXIT_CONTROL_TARGETS:
        return "exit"
    return "entry"


def _alternatives_from_probabilities(probabilities: dict[str, float]) -> list[dict[str, Any]]:
    return [
        {"target": target, "score": round(float(score), 4)}
        for target, score in sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    ]


def _class_counts(head: dict[str, Any]) -> dict[str, int]:
    raw_counts = head.get("class_counts") or head.get("target_counts") or {}
    if not isinstance(raw_counts, dict):
        return {}
    return {
        str(target): int(float(count))
        for target, count in raw_counts.items()
        if str(target) and _safe_float(count, math.nan) >= 0
    }


def _missing_model_metadata(ensemble) -> list[str]:
    metadata = getattr(ensemble, "metadata", {}) or {}
    heads = getattr(ensemble, "heads", {}) or {}
    entry_head = heads.get("entry") or {}
    checks = {
        "schema_version": getattr(ensemble, "schema_version", None),
        "heads": heads or None,
        "entry_classes": entry_head.get("classes"),
        "training_started_at": metadata.get("training_started_at")
        or getattr(ensemble, "training_started_at", None),
        "training_finished_at": metadata.get("training_finished_at")
        or getattr(ensemble, "training_finished_at", None)
        or getattr(ensemble, "trained_at", None),
        "dataset_hash": metadata.get("dataset_hash") or getattr(ensemble, "dataset_hash", None),
        "feature_schema_version": metadata.get("feature_schema_version")
        or getattr(ensemble, "feature_schema_version", None),
        "promotion_status": metadata.get("promotion_status")
        or getattr(ensemble, "promotion_status", None),
        "promotion_metrics": metadata.get("promotion_metrics")
        or getattr(ensemble, "promotion_metrics", None),
    }
    return [key for key, value in checks.items() if value in (None, "", [], {})]


def _promotion_metrics_eligible(
    metrics: dict[str, Any],
    config: FeedbackConfig,
) -> tuple[bool, str]:
    if not isinstance(metrics, dict) or not metrics:
        return False, "promotion_metrics_missing"

    min_trades = int(config.promotion_min_oos_trades_total)
    if min_trades > 0:
        trades = _first_metric(metrics, "oos_trades_total", "trades_total", "trades")
        if trades is None or int(trades) < min_trades:
            return False, "promotion_oos_trades_below_min"

    min_pf = float(config.promotion_min_profit_factor_oos)
    if min_pf > 0:
        profit_factor = _first_metric(
            metrics,
            "oos_profit_factor",
            "profit_factor_oos",
            "profit_factor",
        )
        if profit_factor is None or float(profit_factor) < min_pf:
            return False, "promotion_profit_factor_below_min"

    max_drawdown = float(config.promotion_max_total_oos_drawdown_pct)
    if max_drawdown > 0:
        drawdown = _first_metric(
            metrics,
            "max_total_oos_drawdown_pct",
            "total_oos_drawdown_pct",
            "max_drawdown_pct",
            "max_drawdown",
        )
        if drawdown is None:
            return False, "promotion_drawdown_missing"
        if _fraction_from_percent_or_fraction(abs(float(drawdown))) > _fraction_from_percent_or_fraction(
            max_drawdown
        ):
            return False, "promotion_drawdown_above_max"

    return True, "accepted"


def _first_metric(metrics: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key not in metrics:
            continue
        value = _safe_float(metrics.get(key), math.nan)
        if math.isfinite(value):
            return value
    return None


def _fraction_from_percent_or_fraction(value: float) -> float:
    parsed = abs(float(value))
    if parsed > 1.0:
        return parsed / 100.0
    return parsed


def _runner_up_directional_target(alternatives: list[dict[str, Any]]) -> dict[str, Any] | None:
    for item in alternatives:
        target = str(item.get("target") or "")
        if _entry_target_side(target) is not None:
            return item
    return None


def _max_alternative_score(alternatives: list[dict[str, Any]], targets: set[str]) -> float:
    scores = [
        float(item.get("score", 0.0) or 0.0)
        for item in alternatives
        if str(item.get("target") or "") in targets
    ]
    return max(scores) if scores else 0.0


def _coarse_target(target: str) -> str:
    if target in ENTRY_TARGETS or target in EXIT_CONTROL_TARGETS:
        return target
    if target in LONG_TARGETS:
        return "long"
    if target in SHORT_TARGETS:
        return "short"
    if target in SIDEWAYS_TARGETS:
        return "flat"
    if target in EXIT_TARGETS:
        return "exit"
    return target


def _position_side_value(side: Side) -> float:
    if side == Side.LONG:
        return 1.0
    if side == Side.SHORT:
        return -1.0
    return 0.0


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(parsed) or math.isinf(parsed):
        return default
    return parsed


def _pct(base: float, value: float) -> float:
    if not base:
        return 0.0
    return (value - base) / abs(base)


def _ratio(value: float, base: float, default: float) -> float:
    if not base:
        return default
    return value / base


def _clip(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def _range_position(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.5
    return _clip((value - low) / (high - low), 0.0, 1.0)


def _wall_closeness(distance_bps: float | None) -> float:
    if distance_bps is None:
        return 0.0
    return 1.0 - _clip(distance_bps / 80.0, 0.0, 1.0)


def _wall_strength(notional: float, price: float) -> float:
    if notional <= 0 or price <= 0:
        return 0.0
    return _clip((notional / price) / 500.0, 0.0, 2.0) / 2.0


def _trade_side_value(side: str | None) -> float:
    if side == "buy":
        return 1.0
    if side == "sell":
        return -1.0
    return 0.0


def _normalized_return(df: pd.DataFrame, periods: int) -> float:
    if len(df) <= periods:
        return 0.0
    current = _safe_float(df["close"].iloc[-1])
    previous = _safe_float(df["close"].iloc[-1 - periods])
    return _clip(_pct(previous, current), -0.04, 0.04) / 0.04


def _normalized_slope(df: pd.DataFrame, column: str, periods: int) -> float:
    if column not in df or len(df) <= periods:
        return 0.0
    current = _safe_float(df[column].iloc[-1])
    previous = _safe_float(df[column].iloc[-1 - periods])
    close = _safe_float(df["close"].iloc[-1])
    if not close:
        return 0.0
    return _clip((current - previous) / close, -0.03, 0.03) / 0.03


def _frame_bias(df: pd.DataFrame) -> float:
    if df.empty or len(df) < 3:
        return 0.0
    last = df.iloc[-1]
    prev = df.iloc[-3]
    close = _safe_float(last.get("close"))
    ema_fast = _safe_float(last.get("ema_fast"))
    ema_slow = _safe_float(last.get("ema_slow"))
    prev_fast = _safe_float(prev.get("ema_fast"))
    prev_slow = _safe_float(prev.get("ema_slow"))
    if close > ema_fast > ema_slow and ema_fast >= prev_fast and ema_slow >= prev_slow:
        return 1.0
    if close < ema_fast < ema_slow and ema_fast <= prev_fast and ema_slow <= prev_slow:
        return -1.0
    return 0.0


def _time_features(now: datetime) -> dict[str, float]:
    timestamp = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
    minutes = timestamp.hour * 60 + timestamp.minute
    angle = 2 * math.pi * (minutes / 1440)
    return {"hour_sin": math.sin(angle), "hour_cos": math.cos(angle)}
