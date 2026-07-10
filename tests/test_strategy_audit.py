import json

from ngn6_bot.config import RuntimeConfig
from ngn6_bot.strategy_audit import pair_paper_events, run_strategy_audit, trade_metrics


def _config(tmp_path):
    return RuntimeConfig(
        raw={
            "bot": {"timezone": "Europe/Moscow"},
            "instrument": {
                "ticker": "NGN6",
                "min_price_increment": 0.001,
                "money_value_per_price_step": 7.79293,
            },
            "paper": {
                "events_file": str(tmp_path / "events.jsonl"),
                "initial_cash": 300_000,
            },
            "data_collection": {
                "decisions_file": str(tmp_path / "decisions.jsonl"),
                "market_structure_file": str(tmp_path / "market.jsonl"),
            },
            "execution": {"slippage_bps_assumption": 4},
            "session": {"trading_start": "10:00", "trading_end": "23:45"},
            "risk": {"stop_after_consecutive_losses": 3, "daily_max_loss_pct": 0.01},
        },
        path=tmp_path / "config.yaml",
    )


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_pair_paper_events_aggregates_partial_closes(tmp_path):
    config = _config(tmp_path)
    _write_jsonl(
        tmp_path / "events.jsonl",
        [
            {
                "timestamp": "2026-07-03T10:00:00+00:00",
                "event": "paper_open",
                "details": {
                    "side": "long",
                    "lots": 2,
                    "price": 3.2,
                    "stop_price": 3.19,
                    "reason": "ml_entry:long:0.70",
                    "commission": 10.0,
                },
            },
            {
                "timestamp": "2026-07-03T10:01:00+00:00",
                "event": "paper_close",
                "details": {
                    "lots": 1,
                    "price": 3.201,
                    "gross_pnl": 7.79293,
                    "commission": 5.0,
                    "realized_pnl": 2.79293,
                    "remaining_lots": 1,
                    "reason": "partial_take_profit",
                },
            },
            {
                "timestamp": "2026-07-03T10:02:00+00:00",
                "event": "paper_close",
                "details": {
                    "lots": 1,
                    "price": 3.199,
                    "gross_pnl": -7.79293,
                    "commission": 5.0,
                    "realized_pnl": -12.79293,
                    "remaining_lots": 0,
                    "reason": "hard_stop_hit",
                },
            },
        ],
    )

    trades = pair_paper_events(tmp_path / "events.jsonl", config)

    assert len(trades) == 1
    assert trades[0]["partial_close_count"] == 1
    assert trades[0]["gross_pnl_rub"] == 0.0
    assert trades[0]["total_commission_rub"] == 20.0
    assert trades[0]["net_pnl_rub"] == -20.0
    assert trades[0]["exit_price"] == 3.2


def test_strategy_audit_writes_reconciled_report(tmp_path):
    config = _config(tmp_path)
    _write_jsonl(
        tmp_path / "events.jsonl",
        [
            {
                "timestamp": "2026-07-03T10:00:00+00:00",
                "event": "paper_open",
                "details": {
                    "side": "long",
                    "lots": 1,
                    "price": 3.2,
                    "stop_price": 3.19,
                    "reason": "ml_entry:long:0.70",
                    "commission": 10.0,
                },
            },
            {
                "timestamp": "2026-07-03T10:05:00+00:00",
                "event": "paper_close",
                "details": {
                    "lots": 1,
                    "price": 3.21,
                    "gross_pnl": 77.9293,
                    "commission": 10.0,
                    "realized_pnl": 67.9293,
                    "remaining_lots": 0,
                    "reason": "ml_exit:exit:0.60",
                },
            },
        ],
    )
    _write_jsonl(tmp_path / "decisions.jsonl", [])
    _write_jsonl(tmp_path / "market.jsonl", [])

    report = run_strategy_audit(config, output_dir=tmp_path / "report")

    assert report["trade_period"]["completed_trades"] == 1
    assert report["reconciliation"]["paper_net_pnl_rub"] == 57.9293
    assert report["reconciliation"]["gross_plus_cost_equals_net"] is True
    assert report["analyzer_commit_hash"]
    assert (tmp_path / "report" / "paper_trade_forensics.csv").exists()
    assert (tmp_path / "report" / "strategy_audit_summary.json").exists()


def test_trade_metrics_requires_time_stability_for_gate():
    rows = [
        {
            "trading_date_msk": "2026-07-03",
            "net_pnl_rub": 10.0,
            "gross_pnl_rub": 12.0,
            "total_commission_rub": 2.0,
            "net_ticks_per_lot": 1.0,
            "duration_seconds": 60.0,
        }
        for _ in range(40)
    ]

    metrics = trade_metrics(rows)

    assert metrics["profit_factor"] == float("inf")
    assert metrics["stability_gate_passed"] is False
