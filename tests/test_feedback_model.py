from datetime import datetime, timezone

import pandas as pd

from ngn6_bot.learning.feedback_model import (
    FEATURE_KEYS,
    ClassBalanceConfig,
    FeedbackConfig,
    FeedbackExample,
    FeedbackModel,
    build_feature_snapshot,
    label_to_target,
)
from ngn6_bot.learning.ensemble import FeedbackEnsemblePrediction
from ngn6_bot.learning.ensemble import (
    FeedbackEnsemble,
    _balance_entry_examples,
    _features_to_frame,
    _quality_filtered_examples,
    train_feedback_ensemble,
)
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


def test_feature_snapshot_schema_includes_adx_family():
    features = _features()
    expected = {
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
    }

    assert expected.issubset(FEATURE_KEYS)
    assert expected.issubset(features)
    frame = _features_to_frame([features], FEATURE_KEYS)
    assert expected.issubset(set(frame.columns))


def test_feature_snapshot_clips_future_rows_to_decision_time():
    index = pd.to_datetime(
        ["2026-01-01T10:00:00+00:00", "2026-01-01T10:01:00+00:00"]
    )
    frame = pd.DataFrame(
        {
            "open": [100.0, 999.0],
            "high": [101.0, 999.0],
            "low": [99.0, 999.0],
            "close": [100.0, 999.0],
            "volume": [1000.0, 999.0],
            "volume_ma": [1000.0, 999.0],
            "ema_fast": [100.0, 999.0],
            "ema_slow": [100.0, 999.0],
            "adx": [20.0, 99.0],
            "plus_di": [25.0, 99.0],
            "minus_di": [15.0, 99.0],
            "atr": [1.0, 99.0],
            "macd": [0.1, 99.0],
            "macd_signal": [0.05, 99.0],
            "macd_hist": [0.05, 99.0],
            "bb_upper": [101.0, 999.0],
            "bb_lower": [99.0, 999.0],
            "bb_width_pct": [0.4, 99.0],
            "rsi": [50.0, 99.0],
        },
        index=index,
    )

    features = build_feature_snapshot(
        execution_df=frame,
        confirmation_df=frame,
        context_df=frame,
        orderbook=OrderBookFeatures(100, 100.1, 100.05, 1.0, 0.5),
        trade_flow=TradeFlowFeatures(),
        now=datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc),
    )

    assert features["adx"] == 0.20
    assert features["plus_di"] == 0.25


def test_training_quality_filter_excludes_missing_unmatured_or_untrusted_examples():
    features = _features()
    examples = [
        FeedbackExample("LONG", "long", features, "test"),
        FeedbackExample("LONG", "long", features, "test", feature_complete=False),
        FeedbackExample("LONG", "long", features, "test", label_matured=False),
        FeedbackExample("LONG", "long", features, "test", market_data_trusted=False),
    ]

    result = _quality_filtered_examples(examples)

    assert len(result) == 1
    assert result[0].feature_complete is True


def test_balance_entry_examples_downsamples_flat_to_ratio():
    features = _features()
    examples = [
        FeedbackExample("LONG", "long", features, "test", timestamp=f"2026-01-01T00:{i:02d}:00+00:00")
        for i in range(5)
    ] + [
        FeedbackExample("SHORT", "short", features, "test", timestamp=f"2026-01-01T01:{i:02d}:00+00:00")
        for i in range(5)
    ] + [
        FeedbackExample("FLAT", "flat", features, "test", timestamp=f"2026-01-01T02:{i:02d}:00+00:00")
        for i in range(100)
    ]

    balanced, summary = _balance_entry_examples(
        examples,
        ClassBalanceConfig(
            enabled=True,
            flat_downsample_ratio=2.0,
            min_directional_examples=1,
            max_flat_share_after_balance=1.0,
        ),
    )
    counts = {target: sum(1 for item in balanced if item.target == target) for target in ["long", "short", "flat"]}

    assert counts == {"long": 5, "short": 5, "flat": 20}
    assert summary["dropped_flat_examples"] == 80


class _StaticProbaModel:
    def __init__(self, classes, probabilities):
        self.classes_ = classes
        self._probabilities = probabilities

    def predict_proba(self, frame):
        return [self._probabilities for _ in range(len(frame))]


def _two_stage_ensemble():
    return FeedbackEnsemble(
        feature_keys=list(FEATURE_KEYS),
        classes=["flat", "long", "short"],
        models=[],
        examples=100,
        trained_at="2026-01-01T00:00:00+00:00",
        heads={
            "entry": {"classes": ["flat", "long", "short"], "models": [], "examples": 100},
            "opportunity": {
                "classes": ["no_trade", "trade"],
                "models": [
                    {
                        "name": "static",
                        "kind": "sklearn",
                        "model": _StaticProbaModel(["no_trade", "trade"], [0.30, 0.70]),
                        "classes": ["no_trade", "trade"],
                    }
                ],
                "examples": 100,
            },
            "direction": {
                "classes": ["long", "short"],
                "models": [
                    {
                        "name": "static",
                        "kind": "sklearn",
                        "model": _StaticProbaModel(["long", "short"], [0.60, 0.40]),
                        "classes": ["long", "short"],
                    }
                ],
                "examples": 20,
            },
        },
    )


def test_two_stage_entry_uses_joint_trade_direction_probability_for_threshold():
    model = FeedbackModel(
        FeedbackConfig(enabled=True, mode="control", ensemble_enabled=False, control_entry_threshold=0.50),
        [],
    )
    model.ensemble = _two_stage_ensemble()

    blocked = model.signal_from_prediction(
        execution_df=_frame(),
        confirmation_df=_frame(),
        context_df=_frame(),
        orderbook=OrderBookFeatures(101, 101.1, 101.05, 1.0, 0.5),
        trade_flow=TradeFlowFeatures(),
        now=datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc),
        price=101.0,
    )
    assert blocked.side == Side.FLAT

    model.config = FeedbackConfig(
        enabled=True,
        mode="control",
        ensemble_enabled=False,
        control_entry_threshold=0.40,
    )
    allowed = model.signal_from_prediction(
        execution_df=_frame(),
        confirmation_df=_frame(),
        context_df=_frame(),
        orderbook=OrderBookFeatures(101, 101.1, 101.05, 1.0, 0.5),
        trade_flow=TradeFlowFeatures(),
        now=datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc),
        price=101.0,
    )

    assert allowed.side == Side.LONG
    assert allowed.reason.startswith("ml_entry:long")


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
