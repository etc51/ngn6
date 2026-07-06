from __future__ import annotations

import json
from datetime import datetime, timezone

from ngn6_bot.backtest import BacktestMetrics, BacktestReport
from ngn6_bot.models import Position, Side
from ngn6_bot.paper import PaperPortfolio, PaperPortfolioConfig
from ngn6_bot.recorder import StrategyRecorder
from ngn6_bot.runtime_metadata import current_commit_hash


def test_current_commit_hash_prefers_environment(monkeypatch):
    current_commit_hash.cache_clear()
    monkeypatch.setenv("NGN6_COMMIT_HASH", "abc123")

    assert current_commit_hash() == "abc123"

    current_commit_hash.cache_clear()


def test_recorder_adds_commit_hash_to_runtime_jsonl(tmp_path, monkeypatch):
    current_commit_hash.cache_clear()
    monkeypatch.setenv("NGN6_COMMIT_HASH", "recorder-hash")
    recorder = StrategyRecorder(
        True,
        tmp_path / "market.jsonl",
        tmp_path / "decisions.jsonl",
    )

    recorder.record_decision(
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        action="skip",
        reason="test",
    )

    row = json.loads((tmp_path / "decisions.jsonl").read_text(encoding="utf-8"))
    assert row["commit_hash"] == "recorder-hash"

    current_commit_hash.cache_clear()


def test_paper_state_and_events_add_commit_hash(tmp_path, monkeypatch):
    current_commit_hash.cache_clear()
    monkeypatch.setenv("NGN6_COMMIT_HASH", "paper-hash")
    portfolio = PaperPortfolio(
        PaperPortfolioConfig(
            initial_cash=300000,
            max_margin_notional=1000,
            state_file=tmp_path / "paper_state.json",
            events_file=tmp_path / "paper_events.jsonl",
            lot_size=1,
            notional_multiplier=1,
            min_price_increment=0.01,
            money_value_per_price_step=1,
            initial_margin_on_buy=0,
            initial_margin_on_sell=0,
            commission_per_lot_per_side=0,
            commission_round_trip_bps=0,
        )
    )

    portfolio.mark_to_market(Position(), 100, datetime(2026, 1, 1, tzinfo=timezone.utc))
    portfolio.open_position(
        side=Side.LONG,
        lots=1,
        price=100,
        stop_price=99,
        reason="test",
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )

    state = json.loads((tmp_path / "paper_state.json").read_text(encoding="utf-8"))
    event = json.loads((tmp_path / "paper_events.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert state["commit_hash"] == "paper-hash"
    assert event["commit_hash"] == "paper-hash"

    current_commit_hash.cache_clear()


def test_backtest_report_json_adds_commit_hash(monkeypatch):
    current_commit_hash.cache_clear()
    monkeypatch.setenv("NGN6_COMMIT_HASH", "backtest-hash")
    report = BacktestReport(
        ticker="NGN6",
        figi="figi",
        candles=0,
        started_at="",
        finished_at="",
        metrics=BacktestMetrics(0, 0.0, 0.0, 0.0, 0.0, None, 0.0, 0.0),
        trades=[],
        limitations=[],
    )

    payload = json.loads(report.to_json())

    assert payload["commit_hash"] == "backtest-hash"

    current_commit_hash.cache_clear()
