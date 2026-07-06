from datetime import datetime, timezone

import pandas as pd

from ngn6_bot.learning.feedback_model import (
    FeedbackConfig,
    FeedbackExample,
    FeedbackModel,
    build_feature_snapshot,
    label_to_target,
)
from ngn6_bot.learning.ensemble import FeedbackEnsemblePrediction
from ngn6_bot.learning.ensemble import train_feedback_ensemble
from ngn6_bot.models import OrderBookFeatures, Side, Signal, TradeFlowFeatures


def _frame(rows=40):
    index = pd.date_range("2026-01-01 07:00", periods=rows, freq="1min", tz="UTC")
    close = pd.Series([100 + i * 0.03 for i in range(rows)], index=index)
    df = pd.DataFrame(
        {
            "open": close - 0.01,
            "high": close + 0.04,
            "low": close - 0.04,
            "close": close,
            "volume": 1000,
        }
    )
    df["ema_fast"] = df["close"].ewm(span=5, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=10, adjust=False).mean()
    df["rsi"] = 50.0
    df["bb_upper"] = df["close"] + 0.2
    df["bb_lower"] = df["close"] - 0.2
    df["bb_width_pct"] = 0.4
    df["volume_ma"] = 1000
    return df


def _features():
    return build_feature_snapshot(
        execution_df=_frame(),
        confirmation_df=_frame(),
        context_df=_frame(),
        orderbook=OrderBookFeatures(101, 101.1, 101.05, 1.0, 0.5),
        trade_flow=TradeFlowFeatures(),
        now=datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc),
    )


def test_label_to_target_maps_existing_user_labels():
    assert label_to_target("VERY_STRONG_LONG") == "long"
    assert label_to_target("GOOD_SHORT") == "short"
    assert label_to_target("FAST_SHORT") == "short"
    assert label_to_target("SHORT_FROM_PEAK") == "short"
    assert label_to_target("WEAK_BOUNCE") == "flat"
    assert label_to_target("EXIT_BY_ORDERBOOK") == "exit"


def test_feedback_shadow_adds_metadata_without_blocking():
    features = _features()
    model = FeedbackModel(
        FeedbackConfig(enabled=True, mode="shadow", min_examples=1, ensemble_enabled=False),
        [FeedbackExample("SIDEWAYS", "sideways", features, "test")],
    )
    signal = Signal(Side.LONG, 0.7, "test", 101.0, 99.0, datetime.now(timezone.utc))

    adjusted = model.apply(
        signal,
        execution_df=_frame(),
        confirmation_df=_frame(),
        context_df=_frame(),
        orderbook=OrderBookFeatures(101, 101.1, 101.05, 1.0, 0.5),
        trade_flow=TradeFlowFeatures(),
        now=datetime.now(timezone.utc),
    )

    assert adjusted.side == Side.LONG
    assert adjusted.metadata["feedback"]["target"] == "sideways"


def test_feedback_filter_blocks_sideways_match():
    features = _features()
    model = FeedbackModel(
        FeedbackConfig(
            enabled=True,
            mode="filter",
            ensemble_enabled=False,
            min_examples=1,
            block_similarity_threshold=0.5,
        ),
        [FeedbackExample("SIDEWAYS", "flat", features, "test")],
    )
    signal = Signal(Side.SHORT, 0.7, "test", 101.0, 103.0, datetime.now(timezone.utc))

    adjusted = model.apply(
        signal,
        execution_df=_frame(),
        confirmation_df=_frame(),
        context_df=_frame(),
        orderbook=OrderBookFeatures(101, 101.1, 101.05, 1.0, 0.5),
        trade_flow=TradeFlowFeatures(),
        now=datetime.now(timezone.utc),
    )

    assert adjusted.side == Side.FLAT
    assert adjusted.reason.startswith("feedback_blocked_flat")


def test_feedback_filter_blocks_opposite_phase_target():
    features = _features()
    model = FeedbackModel(
        FeedbackConfig(
            enabled=True,
            mode="filter",
            ensemble_enabled=False,
            min_examples=1,
            opposite_similarity_threshold=0.5,
        ),
        [FeedbackExample("FAST_SHORT", "short", features, "test")],
    )
    signal = Signal(Side.LONG, 0.7, "test", 101.0, 99.0, datetime.now(timezone.utc))

    adjusted = model.apply(
        signal,
        execution_df=_frame(),
        confirmation_df=_frame(),
        context_df=_frame(),
        orderbook=OrderBookFeatures(101, 101.1, 101.05, 1.0, 0.5),
        trade_flow=TradeFlowFeatures(),
        now=datetime.now(timezone.utc),
    )

    assert adjusted.side == Side.FLAT
    assert adjusted.reason.startswith("feedback_blocked_opposite:short")


def test_feedback_filter_confirms_matching_phase_target():
    features = _features()
    model = FeedbackModel(
        FeedbackConfig(
            enabled=True,
            mode="filter",
            ensemble_enabled=False,
            min_examples=1,
            confirmation_similarity_threshold=0.5,
        ),
        [FeedbackExample("LONG_CONTINUATION", "long", features, "test")],
    )
    signal = Signal(Side.LONG, 0.7, "test", 101.0, 99.0, datetime.now(timezone.utc))

    adjusted = model.apply(
        signal,
        execution_df=_frame(),
        confirmation_df=_frame(),
        context_df=_frame(),
        orderbook=OrderBookFeatures(101, 101.1, 101.05, 1.0, 0.5),
        trade_flow=TradeFlowFeatures(),
        now=datetime.now(timezone.utc),
    )

    assert adjusted.side == Side.LONG
    assert adjusted.confidence > signal.confidence
    assert "feedback_confirmed:long" in adjusted.reason


def test_feedback_control_opens_from_ml_long_prediction():
    features = _features()
    model = FeedbackModel(
        FeedbackConfig(
            enabled=True,
            mode="control",
            ensemble_enabled=False,
            min_examples=1,
            control_entry_threshold=0.5,
        ),
        [FeedbackExample("LONG_CONTINUATION", "long", features, "test")],
    )

    signal = model.signal_from_prediction(
        execution_df=_frame(),
        confirmation_df=_frame(),
        context_df=_frame(),
        orderbook=OrderBookFeatures(101, 101.1, 101.05, 1.0, 0.5),
        trade_flow=TradeFlowFeatures(),
        now=datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc),
        price=101.0,
    )

    assert signal.side == Side.LONG
    assert signal.reason.startswith("ml_entry:long")
    assert signal.stop_price < signal.price
    assert signal.take_profit2 is not None and signal.take_profit2 > signal.price


def test_feedback_control_does_not_trade_weak_long_label():
    features = _features()
    model = FeedbackModel(
        FeedbackConfig(
            enabled=True,
            mode="control",
            ensemble_enabled=False,
            min_examples=1,
            control_entry_threshold=0.5,
        ),
        [FeedbackExample("WEAK_LONG", "weak_long", features, "test")],
    )

    signal = model.signal_from_prediction(
        execution_df=_frame(),
        confirmation_df=_frame(),
        context_df=_frame(),
        orderbook=OrderBookFeatures(101, 101.1, 101.05, 1.0, 0.5),
        trade_flow=TradeFlowFeatures(),
        now=datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc),
        price=101.0,
    )

    assert signal.side == Side.FLAT
    assert signal.reason.startswith("ml_no_entry")


def test_feedback_control_explores_moderate_directional_prediction():
    features = _features()
    model = FeedbackModel(
        FeedbackConfig(
            enabled=True,
            mode="control",
            ensemble_enabled=False,
            min_examples=1,
            control_entry_threshold=1.1,
            exploration_enabled=True,
            exploration_entry_threshold=0.5,
        ),
        [FeedbackExample("LONG", "long", features, "test")],
    )

    signal = model.signal_from_prediction(
        execution_df=_frame(),
        confirmation_df=_frame(),
        context_df=_frame(),
        orderbook=OrderBookFeatures(101, 101.1, 101.05, 1.0, 0.5),
        trade_flow=TradeFlowFeatures(),
        now=datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc),
        price=101.0,
    )

    assert signal.side == Side.LONG
    assert signal.reason.startswith("ml_explore_entry:long")
    assert signal.metadata["exploration"] is True


def test_feedback_control_blocks_directional_runner_up_when_flat_wins_by_default():
    model = FeedbackModel(
        FeedbackConfig(
            enabled=True,
            mode="control",
            ensemble_enabled=False,
            min_examples=99,
            control_entry_threshold=0.62,
            exploration_enabled=True,
            exploration_entry_threshold=0.35,
            exploration_runner_up_enabled=True,
            exploration_runner_up_threshold=0.35,
        ),
        [],
    )

    class _FlatWithLongRunnerUp:
        def predict(self, features, *, task="entry"):
            del features, task
            return FeedbackEnsemblePrediction(
                target="flat",
                score=0.52,
                examples=100,
                model_scores={"test": 0.52},
                probabilities={"flat": 0.52, "long": 0.40, "short": 0.08},
            )

    model.ensemble = _FlatWithLongRunnerUp()

    signal = model.signal_from_prediction(
        execution_df=_frame(),
        confirmation_df=_frame(),
        context_df=_frame(),
        orderbook=OrderBookFeatures(101, 101.1, 101.05, 1.0, 0.5),
        trade_flow=TradeFlowFeatures(),
        now=datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc),
        price=101.0,
    )

    assert signal.side == Side.FLAT
    assert signal.reason.startswith("ml_no_entry_weak_or_neutral:flat")
    assert signal.metadata["feedback"]["target"] == "flat"


def test_feedback_control_explores_directional_runner_up_when_flat_block_disabled():
    model = FeedbackModel(
        FeedbackConfig(
            enabled=True,
            mode="control",
            ensemble_enabled=False,
            min_examples=99,
            control_entry_threshold=0.62,
            control_block_neutral_runner_up=False,
            exploration_enabled=True,
            exploration_entry_threshold=0.35,
            exploration_runner_up_enabled=True,
            exploration_runner_up_threshold=0.35,
        ),
        [],
    )

    class _FlatWithLongRunnerUp:
        def predict(self, features, *, task="entry"):
            del features, task
            return FeedbackEnsemblePrediction(
                target="flat",
                score=0.52,
                examples=100,
                model_scores={"test": 0.52},
                probabilities={"flat": 0.52, "long": 0.40, "short": 0.08},
            )

    model.ensemble = _FlatWithLongRunnerUp()

    signal = model.signal_from_prediction(
        execution_df=_frame(),
        confirmation_df=_frame(),
        context_df=_frame(),
        orderbook=OrderBookFeatures(101, 101.1, 101.05, 1.0, 0.5),
        trade_flow=TradeFlowFeatures(),
        now=datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc),
        price=101.0,
    )

    assert signal.side == Side.LONG
    assert signal.reason.startswith("ml_explore_entry:long:0.40")
    assert signal.metadata["feedback"]["target"] == "flat"
    assert signal.metadata["exploration_target"] == "long"


def test_feedback_control_exits_on_sideways_prediction():
    features = _features()
    model = FeedbackModel(
        FeedbackConfig(
            enabled=True,
            mode="control",
            ensemble_enabled=False,
            min_examples=1,
            control_exit_threshold=0.5,
        ),
        [FeedbackExample("EXIT", "exit", {**features, "position_side": 1.0}, "test", task="exit")],
    )

    reason, prediction = model.exit_reason_from_prediction(
        position_side=Side.LONG,
        execution_df=_frame(),
        confirmation_df=_frame(),
        context_df=_frame(),
        orderbook=OrderBookFeatures(101, 101.1, 101.05, 1.0, 0.5),
        trade_flow=TradeFlowFeatures(),
        now=datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc),
    )

    assert prediction.target == "exit"
    assert reason is not None and reason.startswith("ml_exit:exit")


def test_feedback_ensemble_trains_phase_targets(tmp_path):
    long_features = _features()
    short_features = {**long_features, "return_1": -0.2, "return_3": -0.2}
    long_features = {**long_features, "return_1": 0.2, "return_3": 0.2}

    report = train_feedback_ensemble(
        [
            FeedbackExample("FAST_SHORT", "fast_short", short_features, "test"),
            FeedbackExample("LONG_CONTINUATION", "long_continuation", long_features, "test"),
        ],
        output_path=tmp_path / "feedback_ensemble.joblib",
        min_examples=2,
    )

    assert sorted(report.classes) == ["long", "short"]
