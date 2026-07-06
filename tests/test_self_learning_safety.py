from __future__ import annotations

import logging
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

from ngn6_bot.bot import TradingBot
from ngn6_bot.backtest import run_replay_backtest
from ngn6_bot.config import RuntimeConfig, load_config
from ngn6_bot.costs import ExecutionCostConfig, trade_covers_costs
from ngn6_bot.learning.ensemble import FeedbackEnsemblePrediction
from ngn6_bot.learning.ensemble import FeedbackEnsemble
from ngn6_bot.learning.feedback_model import FeedbackConfig, FeedbackModel, build_feature_snapshot
from ngn6_bot.learning.promotion import check_model_eligibility
from ngn6_bot.learning.purged_walk_forward import purged_walk_forward_splits
from ngn6_bot.learning.shadow import evaluate_shadow_predictions
from ngn6_bot.learning.triple_barrier import TripleBarrierConfig, label_entry_decision
from ngn6_bot.models import Candle, OrderBookFeatures, Position, Side, Signal, TradeFlowFeatures
from ngn6_bot.risk import RiskConfig, calculate_position_lots


def _frame(rows: int = 40) -> pd.DataFrame:
    index = pd.date_range("2026-01-01 07:00", periods=rows, freq="1min", tz="UTC")
    close = pd.Series([100 + i * 0.03 for i in range(rows)], index=index)
    frame = pd.DataFrame(
        {
            "open": close - 0.01,
            "high": close + 0.04,
            "low": close - 0.04,
            "close": close,
            "volume": 1000,
        }
    )
    frame["ema_fast"] = frame["close"].ewm(span=5, adjust=False).mean()
    frame["ema_slow"] = frame["close"].ewm(span=10, adjust=False).mean()
    frame["atr"] = 0.4
    frame["rsi"] = 55.0
    frame["bb_upper"] = frame["close"] + 0.2
    frame["bb_lower"] = frame["close"] - 0.2
    frame["bb_width_pct"] = 0.5
    frame["volume_ma"] = 1000
    return frame


def _features() -> dict[str, float]:
    return build_feature_snapshot(
        execution_df=_frame(),
        confirmation_df=_frame(),
        context_df=_frame(),
        orderbook=OrderBookFeatures(101, 101.1, 101.05, 1.0, 0.62),
        trade_flow=TradeFlowFeatures(),
        now=datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc),
    )


class _CandidateLong:
    schema_version = 2
    model_status = "candidate"
    promotion_status = "candidate"

    def predict(self, features, *, task="entry"):
        return FeedbackEnsemblePrediction(
            target="long",
            score=0.91,
            examples=6000,
            model_scores={"candidate": 0.91},
            probabilities={"long": 0.91, "short": 0.03, "flat": 0.06},
        )


def test_model_schema_v1_blocked():
    model = FeedbackModel(
        FeedbackConfig(
            enabled=True,
            mode="control",
            control_require_ensemble_model=True,
            control_require_promoted_model=True,
            ensemble_enabled=False,
        ),
        [],
    )
    model.ensemble = SimpleNamespace(
        schema_version=1,
        heads={"entry": {"classes": ["long", "short", "flat"], "examples": 6000}},
        promotion_score=1.0,
    )

    ready, reason, details = model.control_model_validation()

    assert not ready
    assert reason == "ensemble_schema_below_v2"
    assert details["schema_version"] == 1


def test_candidate_cannot_trade_when_active_is_not_eligible():
    model = FeedbackModel(
        FeedbackConfig(
            enabled=True,
            mode="shadow_then_control",
            control_require_ensemble_model=True,
            control_require_promoted_model=True,
            ensemble_enabled=False,
            candidate_can_trade=False,
        ),
        [],
    )
    model.ensemble = SimpleNamespace(
        schema_version=1,
        heads={"entry": {"classes": ["long", "short", "flat"], "examples": 6000}},
        promotion_score=1.0,
    )
    model.candidate_ensemble = _CandidateLong()

    signal = model.signal_from_prediction(
        execution_df=_frame(),
        confirmation_df=_frame(),
        context_df=_frame(),
        orderbook=OrderBookFeatures(101, 101.1, 101.05, 1.0, 0.62),
        trade_flow=TradeFlowFeatures(),
        now=datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc),
        price=101.0,
    )

    assert signal.side == Side.FLAT
    assert signal.reason == "ml_control_model_not_ready:ensemble_schema_below_v2"
    assert signal.metadata["candidate_shadow"]["target"] == "long"
    assert signal.metadata["candidate_shadow"]["can_trade"] is False


def test_no_fallback_when_ml_not_ready():
    config = load_config("config/ngn6.yaml")
    bot = TradingBot(config, logging.getLogger("test"), runtime_services=False)
    signal = Signal(
        Side.LONG,
        0.7,
        "ema_adx_macd_fallback_long:test",
        3.2,
        3.1,
        datetime.now(timezone.utc),
    )

    assert bot._entry_signal_block_reason(signal) == "fallback_entry_disabled"


def test_risk_gateway_blocks_stale_orderbook():
    config = load_config("config/ngn6.yaml")
    bot = TradingBot(config, logging.getLogger("test"), runtime_services=False)
    orderbook = OrderBookFeatures(
        3.2,
        3.201,
        3.2005,
        2.0,
        0.7,
        bid_depth=10,
        ask_depth=10,
        age_seconds=30,
        source="live",
    )

    reason, details = bot._entry_market_data_block(datetime.now(timezone.utc), orderbook)

    assert reason == "entry_market_data_stale"
    assert details["age_seconds"] == 30


def test_risk_gateway_blocks_low_edge():
    result = trade_covers_costs(
        price=3.2,
        expected_move_pct=0.01,
        spread_bps=1.0,
        config=ExecutionCostConfig(
            slippage_bps_assumption=4,
            commission_per_lot_per_side=0.0,
            commission_round_trip_bps=8,
            min_expected_net_ticks=8,
            min_price_increment=0.001,
            money_value_per_price_step=7.79293,
        ),
    )

    assert not result.accepted
    assert result.reason == "expected_move_below_costs"


def test_position_size_cap():
    signal = Signal(Side.LONG, 0.9, "ml_entry:long:0.9", 100, 99, datetime.now(timezone.utc))
    lots = calculate_position_lots(
        signal,
        RiskConfig(
            deposit_value=300000,
            risk_per_trade_pct=0.50,
            max_risk_per_trade_pct=0.50,
            max_position_lots=1,
            min_position_lots=1,
            stop_buffer_ticks=0,
            min_price_increment=0.001,
            money_value_per_price_step=7.79293,
            partial_take_profit_pct=0,
            partial_take_fraction=0,
            trailing_stop_pct=0,
            close_before_clearing_minutes=0,
            clearings=[],
            timezone="Europe/Moscow",
        ),
    )

    assert lots == 1


def test_triple_barrier_label_long_short_flat():
    config = TripleBarrierConfig(
        tick_size=0.001,
        tick_value=7.79293,
        atr_stop_multiple=1.0,
        take_profit_r_multiple=2.0,
        time_barrier_bars=12,
        min_trainable_r=0.2,
        min_edge_r=0.15,
    )
    index = pd.date_range("2026-01-01 10:15", periods=12, freq="15min", tz="UTC")
    long_path = pd.DataFrame({"open": 100, "high": 103, "low": 99.5, "close": 102}, index=index)
    short_path = pd.DataFrame({"open": 100, "high": 100.5, "low": 97, "close": 98}, index=index)
    flat_path = pd.DataFrame({"open": 100, "high": 100.1, "low": 99.9, "close": 100}, index=index)

    assert (
        label_entry_decision(
            decision_timestamp=datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc),
            future_ohlcv=long_path,
            config=config,
            entry_price=100,
            atr=1,
        ).label
        == "long"
    )
    assert (
        label_entry_decision(
            decision_timestamp=datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc),
            future_ohlcv=short_path,
            config=config,
            entry_price=100,
            atr=1,
        ).label
        == "short"
    )
    assert (
        label_entry_decision(
            decision_timestamp=datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc),
            future_ohlcv=flat_path,
            config=config,
            entry_price=100,
            atr=1,
        ).label
        == "flat"
    )


def test_label_not_created_before_maturity():
    index = pd.date_range("2026-01-01 10:15", periods=3, freq="15min", tz="UTC")
    path = pd.DataFrame({"open": 100, "high": 103, "low": 99.5, "close": 102}, index=index)

    result = label_entry_decision(
        decision_timestamp=datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc),
        future_ohlcv=path,
        config=TripleBarrierConfig(tick_size=0.001, tick_value=7.79293, time_barrier_bars=12),
        entry_price=100,
        atr=1,
    )

    assert result.matured is False
    assert result.label == "flat"


def test_purged_walk_forward_no_overlap_and_embargo():
    timestamps = pd.date_range("2026-01-01", periods=30, freq="1D", tz="UTC")
    events = pd.DataFrame(
        {
            "timestamp": timestamps,
            "event_end": timestamps + pd.Timedelta(days=1),
        }
    )
    events.loc[8, "event_end"] = timestamps[11]

    fold = purged_walk_forward_splits(
        events,
        folds=1,
        train_window_days=10,
        validation_window_days=0,
        test_window_days=2,
        embargo_bars=2,
        purge_by_event_end=True,
    )[0]

    assert 8 not in fold.train_indices
    assert max(fold.train_indices) < min(fold.test_indices) - 2
    for idx in fold.train_indices:
        assert not (
            events.loc[idx, "timestamp"] < fold.test_end
            and events.loc[idx, "event_end"] >= fold.test_start
        )


def test_signal_flip_hysteresis_blocks_single_early_flip():
    config = load_config("config/ngn6.yaml")
    bot = TradingBot(config, logging.getLogger("test"), runtime_services=False)
    now = datetime(2026, 1, 1, 10, 20, tzinfo=timezone.utc)
    position = Position(
        side=Side.LONG,
        lots=1,
        avg_price=100,
        opened_at=now - timedelta(minutes=10),
        stop_price=99,
    )

    confirmed, details = bot._exit_signal_confirmed(
        kind="signal_flip",
        position=position,
        reason="signal-flip",
        now=now,
        price=100,
        target_side=Side.SHORT,
        signal_confidence=0.9,
    )

    assert confirmed is False
    assert details["signal_flip_blocked"] == "min_hold_bars"


def test_live_disabled(tmp_path):
    data = load_config("config/ngn6.yaml").raw
    data = {key: value.copy() if isinstance(value, dict) else value for key, value in data.items()}
    data["trading"]["live_enabled"] = True
    path = tmp_path / "ngn6.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="Live execution is prohibited"):
        load_config(path)


def test_shadow_evaluation_blocks_without_matured_labels(tmp_path):
    decisions = tmp_path / "decisions.jsonl"
    decisions.write_text(
        json.dumps(
            {
                "timestamp": "2026-01-01T10:15:00+00:00",
                "details": {
                    "metadata": {
                        "candidate_shadow": {
                            "target": "long",
                            "score": 0.8,
                            "model_status": "candidate",
                            "promotion_status": "candidate",
                            "can_trade": False,
                        }
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    base = load_config("config/ngn6.yaml")
    raw = json.loads(json.dumps(base.raw))
    raw["data_collection"]["decisions_file"] = str(decisions)
    raw["learning"]["oracle_labels_csv"] = str(tmp_path / "missing.csv")

    report = evaluate_shadow_predictions(RuntimeConfig(raw=raw, path=base.path))

    assert report.passed is False
    assert report.reason == "insufficient_matured_labels"
    assert report.shadow_trade_signals == 1


def test_shadow_evaluation_can_pass_with_matured_labels(tmp_path):
    decisions = tmp_path / "decisions.jsonl"
    labels = tmp_path / "labels.csv"
    decisions.write_text(
        json.dumps(
            {
                "timestamp": "2026-01-01T10:15:00+00:00",
                "details": {
                    "metadata": {
                        "candidate_shadow": {
                            "target": "long",
                            "score": 0.8,
                            "model_status": "candidate",
                            "promotion_status": "candidate",
                            "can_trade": False,
                        }
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    labels.write_text(
        "\n".join(
            [
                "date,start_time,end_time,label,timeframe,score",
                "2026-01-01,13:00:00,13:30:00,LONG_CONTINUATION,15min,1.2",
            ]
        ),
        encoding="utf-8",
    )
    base = load_config("config/ngn6.yaml")
    raw = json.loads(json.dumps(base.raw))
    raw["data_collection"]["decisions_file"] = str(decisions)
    raw["learning"]["oracle_labels_csv"] = str(labels)
    raw["shadow"]["min_days"] = 1
    raw["shadow"]["min_trade_signals"] = 1
    raw["shadow"]["min_profit_factor"] = 1.0
    raw["shadow"]["max_drawdown_pct"] = 1.0

    report = evaluate_shadow_predictions(RuntimeConfig(raw=raw, path=base.path))

    assert report.passed is True
    assert report.reason == "accepted"
    assert report.shadow_profit_factor == float("inf")


def test_promotion_check_blocks_candidate_status(tmp_path):
    model_path = tmp_path / "candidate.joblib"
    ensemble = FeedbackEnsemble(
        feature_keys=["return_1"],
        classes=["flat", "long", "short"],
        models=[],
        examples=9000,
        trained_at="2026-01-01T00:00:00+00:00",
        heads={
            "entry": {
                "classes": ["flat", "long", "short"],
                "class_counts": {"flat": 3000, "long": 3000, "short": 3000},
                "models": [],
                "examples": 9000,
            },
            "exit": {
                "classes": ["hold", "exit"],
                "class_counts": {"hold": 2000, "exit": 2000},
                "models": [],
                "examples": 4000,
            },
        },
        promotion_score=1.0,
        model_status="candidate",
        promotion_status="candidate",
        promotion_metrics={
            "oos_trades_total": 200,
            "oos_profit_factor": 1.5,
            "max_total_oos_drawdown_pct": 0.02,
        },
        dataset_hash="abc",
    )
    ensemble.save(model_path)

    report = check_model_eligibility(load_config("config/ngn6.yaml"), model_path=model_path)

    assert report.ready is False
    assert report.reason == "model_not_promoted"


def test_promoted_only_backtest_uses_fail_closed_model_gate():
    config = load_config("config/ngn6.yaml")
    start = datetime(2026, 1, 1, 7, 0, tzinfo=timezone.utc)
    candles = [
        Candle(
            timestamp=start + timedelta(minutes=i),
            open=100 + i * 0.01,
            high=100.2 + i * 0.01,
            low=99.8 + i * 0.01,
            close=100.1 + i * 0.01,
            volume=1000,
            timeframe="1min",
        )
        for i in range(120)
    ]

    report = run_replay_backtest(config, candles, "TESTFIGI", promoted_only=True)

    assert report.metrics.trades == 0
