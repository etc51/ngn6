from datetime import datetime, timedelta, timezone

from ngn6_bot.models import Position, Side
from ngn6_bot.paper import PaperPortfolio, PaperPortfolioConfig


def _portfolio(tmp_path, max_margin_notional=1000):
    return PaperPortfolio(
        PaperPortfolioConfig(
            initial_cash=300000,
            max_margin_notional=max_margin_notional,
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


def test_paper_portfolio_tracks_realized_pnl(tmp_path):
    portfolio = _portfolio(tmp_path)
    now = datetime.now(timezone.utc)

    accepted, lots, _ = portfolio.open_position(
        side=Side.LONG,
        lots=2,
        price=100,
        stop_price=99,
        take_profit1=101,
        take_profit2=102,
        reason="test",
        timestamp=now,
    )
    assert accepted
    assert lots == 2
    restored = portfolio.restore_position()
    assert restored.take_profit1 == 101
    assert restored.take_profit2 == 102

    accepted, realized, _ = portfolio.close_position(
        position=Position(side=Side.LONG, lots=2, avg_price=100, opened_at=now),
        price=101,
        lots=2,
        reason="target",
        timestamp=now,
    )

    assert accepted
    assert realized == 200
    state = portfolio.mark_to_market(Position(), 101, now)
    assert state["cash"] == 300200
    assert state["equity"] == 300200


def test_paper_portfolio_rejects_when_margin_limit_is_too_low(tmp_path):
    portfolio = _portfolio(tmp_path, max_margin_notional=50)

    accepted, lots, reason = portfolio.open_position(
        side=Side.SHORT,
        lots=1,
        price=100,
        stop_price=101,
        reason="test",
        timestamp=datetime.now(timezone.utc),
    )

    assert not accepted
    assert lots == 0
    assert reason == "paper_margin_limit"


def test_paper_portfolio_uses_futures_initial_margin_for_lot_limit(tmp_path):
    portfolio = PaperPortfolio(
        PaperPortfolioConfig(
            initial_cash=300000,
            max_margin_notional=15000,
            state_file=tmp_path / "paper_state.json",
            events_file=tmp_path / "paper_events.jsonl",
            lot_size=1,
            notional_multiplier=7792.93,
            min_price_increment=0.001,
            money_value_per_price_step=7.79293,
            initial_margin_on_buy=7118.76,
            initial_margin_on_sell=7260.29,
            commission_per_lot_per_side=0,
            commission_round_trip_bps=0,
        )
    )

    accepted, lots, reason = portfolio.open_position(
        side=Side.LONG,
        lots=6,
        price=3.2555,
        stop_price=3.245,
        reason="test",
        timestamp=datetime.now(timezone.utc),
    )

    assert accepted
    assert lots == 2
    assert reason == "paper_margin_lots_reduced"

    state = portfolio.mark_to_market(portfolio.restore_position(), 3.258, datetime.now(timezone.utc))
    assert state["margin_used"] == 14237.52
    assert round(state["unrealized_pnl"], 5) == 38.96465
    assert round(state["contract_value"], 5) == 25389.36594


def test_risk_snapshot_persists_daily_loss_streak_and_last_exit(tmp_path):
    portfolio = _portfolio(tmp_path)
    now = datetime(2026, 1, 5, 10, 0, tzinfo=timezone.utc)
    for index in range(2):
        opened_at = now + timedelta(minutes=index * 2)
        accepted, _, _ = portfolio.open_position(
            side=Side.LONG,
            lots=1,
            price=100,
            stop_price=99,
            reason="test",
            timestamp=opened_at,
        )
        assert accepted
        accepted, _, _ = portfolio.close_position(
            position=Position(side=Side.LONG, lots=1, avg_price=100, opened_at=opened_at),
            price=99,
            lots=1,
            reason="hard_stop_hit",
            timestamp=opened_at + timedelta(minutes=1),
        )
        assert accepted

    snapshot = portfolio.risk_snapshot(now + timedelta(hours=1), "UTC")

    assert snapshot.daily_net_pnl == -200
    assert snapshot.completed_trades_today == 2
    assert snapshot.consecutive_losses == 2
    assert snapshot.consecutive_hard_stops == 2
    assert snapshot.last_exit_side == Side.LONG
    assert snapshot.last_exit_reason == "hard_stop_hit"
    assert snapshot.last_exit_pnl_ticks == -100
