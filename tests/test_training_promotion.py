from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ngn6_bot.config import RuntimeConfig
from ngn6_bot.learning.ensemble import FeedbackEnsembleReport
from ngn6_bot.learning.feedback_model import FeedbackConfig, FeedbackExample
from ngn6_bot.learning.training import _limit_examples, _promotion_decision


def _config(tmp_path: Path) -> RuntimeConfig:
    return RuntimeConfig(
        {
            "bot": {"timezone": "Europe/Moscow", "dry_run": True},
            "learning": {"mode": "control"},
            "instrument": {"ticker": "NGN6"},
        },
        tmp_path / "config.yaml",
    )


@dataclass(frozen=True)
class _Metrics:
    trades: int
    final_equity_pct: float
    max_drawdown_pct: float
    profit_factor: float | None


@dataclass(frozen=True)
class _Fold:
    metrics: _Metrics


@dataclass(frozen=True)
class _WalkForward:
    folds: list[_Fold]


def test_promotion_rejects_candidate_that_fails_backtest_gate(tmp_path, monkeypatch):
    candidate = tmp_path / "candidate.joblib"
    current = tmp_path / "current.joblib"
    candidate.write_bytes(b"candidate")

    def fake_walk_forward(config, candles, figi, folds):
        del config, candles, figi, folds
        return _WalkForward([_Fold(_Metrics(1, -1.0, -1.0, 0.5))])

    monkeypatch.setattr("ngn6_bot.learning.training.run_walk_forward", fake_walk_forward)

    decision = _promotion_decision(
        report=FeedbackEnsembleReport(
            path=candidate,
            examples=10,
            classes=["long", "short", "flat"],
            models=["entry:logistic"],
            holdout_accuracy=0.7,
            trained_at="2026-01-01T00:00:00+00:00",
            promotion_score=1.0,
        ),
        target_path=current,
        candidate_path=candidate,
        feedback_config=FeedbackConfig(
            promote_enabled=True,
            promotion_backtest_enabled=True,
            promotion_min_score=-99.0,
            promotion_min_trades=3,
            promotion_min_profit_factor=1.0,
        ),
        runtime_config=_config(tmp_path),
        candles=[],
        figi="figi",
    )

    assert decision["promoted"] is False
    assert decision["reason"] == "candidate_backtest_too_few_trades"


def test_training_example_limit_keeps_most_recent_items():
    examples = [
        FeedbackExample("A", "flat", {"return_1": 0.0}, "test", timestamp=f"2026-01-0{i}")
        for i in range(1, 6)
    ]

    limited = _limit_examples(examples, 2)

    assert [item.timestamp for item in limited] == ["2026-01-04", "2026-01-05"]
