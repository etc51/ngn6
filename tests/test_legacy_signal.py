from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd

from ngn6_bot.legacy_signal import generate_legacy_signal
from ngn6_bot.models import OrderBookFeatures, Side


def _frame(rows=100):
    index = pd.date_range("2026-01-01 07:00", periods=rows, freq="15min", tz="UTC")
    close = pd.Series([100 + i * 0.05 for i in range(rows)], index=index)
    return pd.DataFrame(
        {
            "open": close - 0.02,
            "high": close + 0.08,
            "low": close - 0.08,
            "close": close,
            "volume": 1000,
        }
    )


def _config(bridge_path):
    return SimpleNamespace(
        legacy_bridge_path=str(bridge_path),
        legacy_node_command="node",
        legacy_timeout_seconds=3.0,
        legacy_timeframe="intraday",
        legacy_min_candles=80,
        legacy_max_candles=320,
        legacy_min_probability=52.0,
        legacy_news_bias="auto",
        legacy_retest="auto",
        legacy_structure="auto",
        legacy_event_risk="none",
        legacy_impulse_enabled=True,
        legacy_impulse_move_pct=1.2,
        legacy_impulse_breakout_buffer_pct=0.03,
        legacy_impulse_min_trend=0.28,
        legacy_impulse_min_momentum=0.24,
        legacy_impulse_min_candles=2,
        legacy_impulse_min_probability=55.5,
        legacy_impulse_max_probability=63.5,
    )


def _orderbook(price):
    return OrderBookFeatures(
        best_bid=price - 0.01,
        best_ask=price + 0.01,
        mid_price=price,
        spread_bps=1.0,
        bid_ask_imbalance=0.5,
    )


def test_legacy_signal_maps_executable_plan(monkeypatch, tmp_path):
    bridge = tmp_path / "bridge.js"
    bridge.write_text("// test bridge", encoding="utf-8")

    def fake_call_bridge(**_kwargs):
        return {
            "ok": True,
            "payload": {"signal": "north", "probability": 60.0, "signalOrigin": "test"},
            "plan": {
                "side": "long",
                "allowed": True,
                "entry": 104.0,
                "stop": 100.0,
                "takeProfit1": 106.0,
                "takeProfit2": 108.0,
            },
            "entryReached": True,
            "manual": {"eventRisk": "none"},
        }

    monkeypatch.setattr("ngn6_bot.legacy_signal._call_bridge", fake_call_bridge)

    signal = generate_legacy_signal(
        execution_df=_frame(),
        context_df=_frame(),
        orderbook=_orderbook(105.0),
        config=_config(bridge),
        now=datetime.now(timezone.utc),
    )

    assert signal.side == Side.LONG
    assert signal.confidence == 0.6
    assert signal.stop_price == 100.0
    assert signal.take_profit1 == 106.0
    assert signal.take_profit2 == 108.0
    assert signal.metadata["entry"] == 104.0


def test_legacy_signal_waits_for_plan_entry(monkeypatch, tmp_path):
    bridge = tmp_path / "bridge.js"
    bridge.write_text("// test bridge", encoding="utf-8")

    def fake_call_bridge(**_kwargs):
        return {
            "ok": True,
            "payload": {"signal": "north", "probability": 60.0},
            "plan": {"side": "long", "allowed": True, "entry": 106.0, "stop": 100.0},
            "entryReached": False,
        }

    monkeypatch.setattr("ngn6_bot.legacy_signal._call_bridge", fake_call_bridge)

    signal = generate_legacy_signal(
        execution_df=_frame(),
        context_df=_frame(),
        orderbook=_orderbook(105.0),
        config=_config(bridge),
        now=datetime.now(timezone.utc),
    )

    assert signal.side == Side.FLAT
    assert signal.reason == "legacy_entry_not_reached:long:106.0"
