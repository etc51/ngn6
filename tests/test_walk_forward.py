from datetime import datetime, timedelta, timezone

from ngn6_bot.backtest import run_walk_forward
from ngn6_bot.config import load_config
from ngn6_bot.models import Candle


def test_walk_forward_creates_requested_folds():
    config = load_config("config/ngn6.yaml")
    start = datetime(2026, 1, 1, 7, 0, tzinfo=timezone.utc)
    candles = [
        Candle(start + timedelta(minutes=i), 100, 101, 99, 100 + i * 0.01, 100, "1min")
        for i in range(120)
    ]

    report = run_walk_forward(config, candles, "TESTFIGI", folds=3)

    assert len(report.folds) == 3
    assert sum(fold.candles for fold in report.folds) == 120
