from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from ngn6_bot.config import RuntimeConfig
from ngn6_bot.learning.daily_oracle import DailyOracleResult, DailyOracleScheduler, generate_daily_oracle
from ngn6_bot.learning.feedback_model import FeedbackConfig
from ngn6_bot.learning.training import build_training_examples
from ngn6_bot.models import Candle


def _config(tmp_path: Path, oracle_csv: Path | None = None) -> RuntimeConfig:
    return RuntimeConfig(
        {
            "bot": {"timezone": "Europe/Moscow", "dry_run": True},
            "instrument": {"ticker": "NGN6"},
            "data_collection": {"decisions_file": str(tmp_path / "decisions.jsonl")},
            "learning": {
                "legacy_labels_csv": "",
                "oracle_labels_csv": str(oracle_csv or tmp_path / "missing.csv"),
            },
            "daily_oracle": {
                "enabled": True,
                "run_time": "23:50",
                "minutes": 120,
                "training_minutes": 240,
                "output_dir": str(tmp_path / "oracle"),
                "state_file": str(tmp_path / "oracle_runs.json"),
                "horizon_bars": 4,
                "min_mfe_pct": 0.5,
                "max_mae_pct": 0.4,
                "max_trades_per_day": 3,
                "min_spacing_bars": 2,
                "stop_buffer_pct": 0.05,
                "take_profit_pct": 0.7,
                "sideways_horizon_bars": 2,
                "sideways_max_mfe_pct": 0.2,
            },
            "indicators": {
                "ema_fast": 5,
                "ema_slow": 10,
                "rsi_period": 5,
                "bollinger_period": 5,
                "bollinger_std": 2,
                "volume_ma_period": 5,
            },
            "signals": {"legacy_min_candles": 10},
        },
        tmp_path / "config.yaml",
    )


def _candles() -> list[Candle]:
    start = datetime(2026, 7, 3, 7, 0, tzinfo=timezone.utc)
    closes = []
    price = 100.0
    for index in range(120):
        if 35 <= index <= 70:
            price += 0.08
        elif 80 <= index <= 100:
            price -= 0.02
        else:
            price += 0.005
        closes.append(price)
    return [
        Candle(
            timestamp=start + timedelta(minutes=index),
            open=close - 0.02,
            high=close + 0.05,
            low=close - 0.05,
            close=close,
            volume=1000,
            timeframe="1min",
        )
        for index, close in enumerate(closes)
    ]


def test_daily_oracle_writes_review_and_ml_labels(tmp_path):
    config = _config(tmp_path)

    result = generate_daily_oracle(
        config,
        figi="figi",
        candles_1m=_candles(),
        trading_date=datetime(2026, 7, 3).date(),
        output_dir=tmp_path / "oracle",
    )

    assert result.oracle_trades
    assert result.json_path.exists()
    assert result.labels_csv_path.exists()
    text = result.labels_csv_path.read_text(encoding="utf-8")
    assert "LONG_CONTINUATION" in text
    assert (tmp_path / "oracle" / "latest_oracle_labels.csv").exists()


def test_training_reads_oracle_labels_without_future_features(tmp_path):
    output_dir = tmp_path / "oracle"
    config = _config(tmp_path)
    result = generate_daily_oracle(
        config,
        figi="figi",
        candles_1m=_candles(),
        trading_date=datetime(2026, 7, 3).date(),
        output_dir=output_dir,
    )
    training_config = _config(tmp_path, result.labels_csv_path)

    examples, generated = build_training_examples(
        training_config,
        _candles(),
        FeedbackConfig(
            enabled=True,
            legacy_labels_csv=None,
            oracle_labels_csv=result.labels_csv_path,
            ensemble_enabled=False,
            generate_pnl_examples=False,
        ),
    )

    assert generated > 0
    assert examples
    assert {example.source for example in examples} == {str(result.labels_csv_path)}
    assert all(example.timestamp is not None for example in examples)


def test_training_generates_pnl_entry_and_exit_examples(tmp_path):
    config = _config(tmp_path)

    examples, generated = build_training_examples(
        config,
        _candles(),
        FeedbackConfig(
            enabled=True,
            legacy_labels_csv=None,
            oracle_labels_csv=None,
            ensemble_enabled=False,
            generate_pnl_examples=True,
            min_entry_net_pct=0.01,
            hold_min_net_pct=0.01,
        ),
    )

    assert generated > 0
    assert {example.task for example in examples} == {"entry", "exit"}
    assert {"long", "short", "flat"} & {example.target for example in examples}
    assert {"hold", "exit"} & {example.target for example in examples}


def test_daily_oracle_scheduler_runs_once_and_reloads_model(tmp_path, monkeypatch):
    calls = {"oracle": 0, "train": 0, "reload": 0}

    def fake_oracle(config, logger, *, trading_date, minutes, output_dir):
        del config, logger, minutes, output_dir
        calls["oracle"] += 1
        return DailyOracleResult(
            figi="figi",
            trading_date=trading_date.isoformat(),
            candles_1m=10,
            candles_15m=2,
            oracle_trades=[],
            sideways_labels=0,
            json_path=tmp_path / "oracle.json",
            labels_csv_path=tmp_path / "labels.csv",
        )

    class _Report:
        path = tmp_path / "feedback_ensemble.joblib"
        classes = ["long_continuation"]

    class _Training:
        report = _Report()
        total_examples = 1

    def fake_train(config, logger, *, minutes):
        del config, logger, minutes
        calls["train"] += 1
        return _Training()

    monkeypatch.setattr("ngn6_bot.learning.daily_oracle.generate_daily_oracle_from_api", fake_oracle)
    monkeypatch.setattr("ngn6_bot.learning.daily_oracle._train_feedback_from_api", fake_train)
    scheduler = DailyOracleScheduler(
        _config(tmp_path),
        logger=_NullLogger(),
        reload_feedback_model=lambda: calls.__setitem__("reload", calls["reload"] + 1),
    )
    now = datetime(2026, 7, 3, 20, 55, tzinfo=timezone.utc)

    scheduler.maybe_run(now)
    scheduler.maybe_run(now)

    assert calls == {"oracle": 1, "train": 1, "reload": 1}
    assert (tmp_path / "oracle_runs.json").exists()


def test_daily_oracle_scheduler_backfills_previous_trading_day_before_run_time(
    tmp_path,
    monkeypatch,
):
    seen_dates = []

    def fake_oracle(config, logger, *, trading_date, minutes, output_dir):
        del config, logger, minutes, output_dir
        seen_dates.append(trading_date.isoformat())
        return DailyOracleResult(
            figi="figi",
            trading_date=trading_date.isoformat(),
            candles_1m=10,
            candles_15m=2,
            oracle_trades=[],
            sideways_labels=0,
            json_path=tmp_path / "oracle.json",
            labels_csv_path=tmp_path / "labels.csv",
        )

    class _Report:
        path = tmp_path / "feedback_ensemble.joblib"
        classes = ["long_continuation"]

    class _Training:
        report = _Report()
        total_examples = 1

    def fake_train(config, logger, *, minutes):
        del config, logger, minutes
        return _Training()

    monkeypatch.setattr("ngn6_bot.learning.daily_oracle.generate_daily_oracle_from_api", fake_oracle)
    monkeypatch.setattr("ngn6_bot.learning.daily_oracle._train_feedback_from_api", fake_train)
    scheduler = DailyOracleScheduler(_config(tmp_path), logger=_NullLogger())

    scheduler.maybe_run(datetime(2026, 7, 6, 7, 0, tzinfo=timezone.utc))

    assert seen_dates == ["2026-07-03"]


def test_daily_oracle_scheduler_does_not_mark_failed_run_generated(tmp_path, monkeypatch):
    calls = {"oracle": 0}

    def fake_oracle(config, logger, *, trading_date, minutes, output_dir):
        del config, logger, trading_date, minutes, output_dir
        calls["oracle"] += 1
        raise RuntimeError("api down")

    monkeypatch.setattr("ngn6_bot.learning.daily_oracle.generate_daily_oracle_from_api", fake_oracle)
    scheduler = DailyOracleScheduler(_config(tmp_path), logger=_CaptureLogger())
    now = datetime(2026, 7, 3, 20, 55, tzinfo=timezone.utc)

    scheduler.maybe_run(now)
    scheduler.maybe_run(now + timedelta(minutes=1))
    scheduler.maybe_run(now + timedelta(minutes=31))

    assert calls == {"oracle": 2}
    assert not (tmp_path / "oracle_runs.json").exists()


class _NullLogger:
    def info(self, *args, **kwargs):
        pass

    def exception(self, *args, **kwargs):
        raise AssertionError(kwargs)


class _CaptureLogger:
    def info(self, *args, **kwargs):
        pass

    def exception(self, *args, **kwargs):
        pass
