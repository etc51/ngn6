from datetime import datetime, timedelta, timezone

import pandas as pd

from ngn6_bot.indicators import add_indicators, candles_to_frame
from ngn6_bot.models import Candle, OrderBookFeatures, Side, TradeFlowFeatures
from ngn6_bot.signals import SignalConfig, _microstructure_allows, generate_signal


def test_generate_long_signal_on_momentum_and_orderbook():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    candles = []
    for i in range(40):
        close = 114 + i * 0.2
        candles.append(
            Candle(
                timestamp=start + timedelta(minutes=i),
                open=close - 0.1,
                high=close + 0.2,
                low=close - 0.2,
                close=close,
                volume=100 if i < 39 else 200,
                timeframe="1min",
            )
        )
    df = add_indicators(candles_to_frame(candles))
    context_candles = []
    for i in range(20):
        close = 100 + i * 0.8
        context_candles.append(
            Candle(
                timestamp=start + timedelta(minutes=i * 15),
                open=close - 0.3,
                high=close + 0.4,
                low=close - 0.4,
                close=close,
                volume=1000 + i,
                timeframe="15min",
            )
        )
    context = add_indicators(candles_to_frame(context_candles))
    orderbook = OrderBookFeatures(107, 107.1, 107.05, 9, 0.7, ask_wall_absorbed=True)
    config = SignalConfig(
        rsi_overbought=95,
        rsi_oversold=20,
        min_bollinger_width_pct=0.01,
        volume_multiplier=1.1,
        hold_above_ema_bars=2,
        support_resistance_lookback=20,
        max_level_distance_pct=0.35,
        min_confidence=0.65,
        min_imbalance=0.58,
        trade_flow_min_buy_ratio=0.58,
        trade_flow_min_sell_ratio=0.58,
        news_halt=False,
        allow_rsi_extreme_with_strong_trend=True,
        require_15m_direction=True,
    )

    signal = generate_signal(
        df,
        df,
        context,
        orderbook,
        TradeFlowFeatures(buy_volume=100, sell_volume=20, buy_ratio=0.83, total_volume=120),
        config,
        datetime.now(timezone.utc),
    )

    assert signal.side == Side.LONG


def test_1m_long_signal_is_blocked_by_down_15m_context():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    execution_candles = []
    for i in range(40):
        close = 100 + i * 0.2
        execution_candles.append(
            Candle(
                timestamp=start + timedelta(minutes=i),
                open=close - 0.1,
                high=close + 0.2,
                low=close - 0.2,
                close=close,
                volume=100 if i < 39 else 200,
                timeframe="1min",
            )
        )
    context_candles = []
    for i in range(20):
        close = 120 - i * 0.8
        context_candles.append(
            Candle(
                timestamp=start + timedelta(minutes=i * 15),
                open=close + 0.3,
                high=close + 0.4,
                low=close - 0.4,
                close=close,
                volume=1000 + i,
                timeframe="15min",
            )
        )
    df = add_indicators(candles_to_frame(execution_candles))
    context = add_indicators(candles_to_frame(context_candles))
    orderbook = OrderBookFeatures(107, 107.1, 107.05, 9, 0.7, ask_wall_absorbed=True)
    config = SignalConfig(
        rsi_overbought=95,
        rsi_oversold=20,
        min_bollinger_width_pct=0.01,
        volume_multiplier=1.1,
        hold_above_ema_bars=2,
        support_resistance_lookback=20,
        max_level_distance_pct=0.35,
        min_confidence=0.65,
        min_imbalance=0.58,
        trade_flow_min_buy_ratio=0.58,
        trade_flow_min_sell_ratio=0.58,
        news_halt=False,
        allow_rsi_extreme_with_strong_trend=True,
        require_15m_direction=True,
    )

    signal = generate_signal(
        df,
        df,
        context,
        orderbook,
        TradeFlowFeatures(buy_volume=100, sell_volume=20, buy_ratio=0.83, total_volume=120),
        config,
        datetime.now(timezone.utc),
    )

    assert signal.side == Side.FLAT


def test_news_halt_blocks_signal():
    config = SignalConfig(
        rsi_overbought=70,
        rsi_oversold=30,
        min_bollinger_width_pct=0.1,
        volume_multiplier=1.2,
        hold_above_ema_bars=2,
        support_resistance_lookback=20,
        max_level_distance_pct=0.35,
        min_confidence=0.65,
        min_imbalance=0.58,
        trade_flow_min_buy_ratio=0.58,
        trade_flow_min_sell_ratio=0.58,
        news_halt=True,
        allow_rsi_extreme_with_strong_trend=True,
        require_15m_direction=True,
    )
    signal = generate_signal(
        execution_df=add_indicators(candles_to_frame([])),
        confirmation_df=add_indicators(candles_to_frame([])),
        context_df=add_indicators(candles_to_frame([])),
        orderbook=OrderBookFeatures(None, None, None, None, 0.5),
        trade_flow=TradeFlowFeatures(),
        config=config,
        now=datetime.now(timezone.utc),
    )
    assert signal.side == Side.FLAT


def test_microstructure_blocks_stale_orderbook():
    config = SignalConfig(
        rsi_overbought=70,
        rsi_oversold=30,
        min_bollinger_width_pct=0.1,
        volume_multiplier=1.2,
        hold_above_ema_bars=2,
        support_resistance_lookback=20,
        max_level_distance_pct=0.35,
        min_confidence=0.65,
        min_imbalance=0.58,
        trade_flow_min_buy_ratio=0.58,
        trade_flow_min_sell_ratio=0.58,
        news_halt=False,
        allow_rsi_extreme_with_strong_trend=True,
        require_15m_direction=True,
        require_microstructure=True,
        max_orderbook_age_seconds=8,
    )

    ok, reason = _microstructure_allows(
        Side.LONG,
        OrderBookFeatures(
            100,
            100.1,
            100.05,
            10,
            0.7,
            depth_pressure=0.4,
            age_seconds=20,
            source="live",
        ),
        TradeFlowFeatures(
            buy_volume=30,
            sell_volume=10,
            buy_ratio=0.75,
            total_volume=40,
            directional_volume=40,
            buy_sell_imbalance=0.5,
        ),
        config,
    )

    assert not ok
    assert reason.startswith("microstructure_orderbook_stale")


def test_ema_adx_macd_long_signal_uses_15m_rules_and_atr_targets():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    context = pd.DataFrame(
        [
            {
                "open": 99.5,
                "high": 100.5,
                "low": 99,
                "close": 100,
                "volume": 1000,
                "ema_fast": 99,
                "ema_slow": 98,
                "atr": 1,
                "adx": 25,
                "rsi": 60,
                "macd_hist": 0.1,
            }
            for _ in range(80)
        ],
        index=pd.date_range(now, periods=80, freq="15min"),
    )
    orderbook = OrderBookFeatures(
        99.99,
        100.01,
        100,
        2,
        0.55,
        bid_depth=20,
        ask_depth=20,
        depth_pressure=0.1,
        age_seconds=1,
        source="live",
    )
    config = SignalConfig(
        rsi_overbought=70,
        rsi_oversold=30,
        min_bollinger_width_pct=0.1,
        volume_multiplier=1.2,
        hold_above_ema_bars=2,
        support_resistance_lookback=20,
        max_level_distance_pct=0.35,
        min_confidence=0.65,
        min_imbalance=0.58,
        trade_flow_min_buy_ratio=0.58,
        trade_flow_min_sell_ratio=0.58,
        news_halt=False,
        allow_rsi_extreme_with_strong_trend=True,
        require_15m_direction=True,
        engine="ema_adx_macd",
        require_microstructure=True,
        max_orderbook_age_seconds=8,
        min_book_pressure=0.1,
        ema_adx_macd_warmup_candles=80,
        ema_adx_macd_min_adx=20,
        ema_adx_macd_min_signal_strength=0.30,
        ema_adx_macd_min_trend_strength=0.002,
        ema_adx_macd_signal_strength_trend_scale=0.0125,
        ema_adx_macd_stop_atr_multiple=1.5,
        ema_adx_macd_take_profit_r_multiple=2.5,
        ema_adx_macd_orderbook_required=True,
    )

    signal = generate_signal(
        execution_df=pd.DataFrame(),
        confirmation_df=pd.DataFrame(),
        context_df=context,
        orderbook=orderbook,
        trade_flow=TradeFlowFeatures(),
        config=config,
        now=now,
    )

    assert signal.side == Side.LONG
    assert signal.stop_price == 98.5
    assert signal.take_profit2 == 103.75


def test_ema_adx_macd_always_trade_fallback_does_not_block_weak_trend():
    now = datetime(2026, 7, 2, tzinfo=timezone.utc)
    context = pd.DataFrame(
        [
            {
                "open": 3.18,
                "high": 3.19,
                "low": 3.17,
                "close": close,
                "volume": 1000,
                "ema_fast": 3.18,
                "ema_slow": 3.181,
                "atr": 0.0,
                "adx": 0.0,
                "rsi": 48,
                "macd_hist": -0.001,
            }
            for close in [3.184, 3.183, 3.182]
        ],
        index=pd.date_range(now, periods=3, freq="15min"),
    )
    config = SignalConfig(
        rsi_overbought=70,
        rsi_oversold=30,
        min_bollinger_width_pct=0.1,
        volume_multiplier=1.2,
        hold_above_ema_bars=2,
        support_resistance_lookback=20,
        max_level_distance_pct=0.35,
        min_confidence=0.65,
        min_imbalance=0.58,
        trade_flow_min_buy_ratio=0.58,
        trade_flow_min_sell_ratio=0.58,
        news_halt=False,
        allow_rsi_extreme_with_strong_trend=True,
        engine="ema_adx_macd",
        ema_adx_macd_warmup_candles=3,
        ema_adx_macd_min_adx=20,
        ema_adx_macd_min_signal_strength=0.30,
        ema_adx_macd_min_trend_strength=0.002,
        ema_adx_macd_orderbook_required=False,
        ema_adx_macd_always_trade=True,
    )

    signal = generate_signal(
        execution_df=pd.DataFrame(),
        confirmation_df=pd.DataFrame(),
        context_df=context,
        orderbook=OrderBookFeatures(None, None, None, None, 0.5),
        trade_flow=TradeFlowFeatures(),
        config=config,
        now=now,
    )

    assert signal.side == Side.SHORT
    assert signal.stop_price is not None
    assert "soft=trend_strength_below_min" in signal.reason


def test_ema_adx_macd_default_blocks_fallback_when_setup_is_weak():
    now = datetime(2026, 7, 2, tzinfo=timezone.utc)
    context = pd.DataFrame(
        [
            {
                "open": 3.18,
                "high": 3.19,
                "low": 3.17,
                "close": close,
                "volume": 1000,
                "ema_fast": 3.18,
                "ema_slow": 3.181,
                "atr": 0.0,
                "adx": 0.0,
                "rsi": 48,
                "macd_hist": -0.001,
            }
            for close in [3.184, 3.183, 3.182]
        ],
        index=pd.date_range(now, periods=3, freq="15min"),
    )
    config = SignalConfig(
        rsi_overbought=70,
        rsi_oversold=30,
        min_bollinger_width_pct=0.1,
        volume_multiplier=1.2,
        hold_above_ema_bars=2,
        support_resistance_lookback=20,
        max_level_distance_pct=0.35,
        min_confidence=0.65,
        min_imbalance=0.58,
        trade_flow_min_buy_ratio=0.58,
        trade_flow_min_sell_ratio=0.58,
        news_halt=False,
        allow_rsi_extreme_with_strong_trend=True,
        engine="ema_adx_macd",
        ema_adx_macd_warmup_candles=3,
        ema_adx_macd_min_adx=20,
        ema_adx_macd_min_signal_strength=0.30,
        ema_adx_macd_min_trend_strength=0.002,
        ema_adx_macd_orderbook_required=False,
    )

    signal = generate_signal(
        execution_df=pd.DataFrame(),
        confirmation_df=pd.DataFrame(),
        context_df=context,
        orderbook=OrderBookFeatures(None, None, None, None, 0.5),
        trade_flow=TradeFlowFeatures(),
        config=config,
        now=now,
    )

    assert signal.side == Side.FLAT
    assert signal.reason.startswith("ema_adx_macd_filters_not_met")


def test_ema_adx_macd_strict_filters_block_soft_setup():
    now = datetime(2026, 7, 2, tzinfo=timezone.utc)
    context = pd.DataFrame(
        [
            {
                "open": 3.21,
                "high": 3.22,
                "low": 3.20,
                "close": 3.215,
                "volume": 1000,
                "ema_fast": 3.214,
                "ema_slow": 3.213,
                "atr": 0.011,
                "adx": 12,
                "rsi": 60,
                "macd_hist": 0.001,
            }
            for _ in range(80)
        ],
        index=pd.date_range(now, periods=80, freq="15min"),
    )
    config = SignalConfig(
        rsi_overbought=70,
        rsi_oversold=30,
        min_bollinger_width_pct=0.1,
        volume_multiplier=1.2,
        hold_above_ema_bars=2,
        support_resistance_lookback=20,
        max_level_distance_pct=0.35,
        min_confidence=0.65,
        min_imbalance=0.58,
        trade_flow_min_buy_ratio=0.58,
        trade_flow_min_sell_ratio=0.58,
        news_halt=False,
        allow_rsi_extreme_with_strong_trend=True,
        engine="ema_adx_macd",
        require_microstructure=True,
        max_orderbook_age_seconds=8,
        min_book_pressure=0.1,
        ema_adx_macd_warmup_candles=80,
        ema_adx_macd_min_adx=20,
        ema_adx_macd_min_signal_strength=0.30,
        ema_adx_macd_min_trend_strength=0.002,
        ema_adx_macd_orderbook_required=True,
        ema_adx_macd_always_trade=False,
    )

    signal = generate_signal(
        execution_df=pd.DataFrame(),
        confirmation_df=pd.DataFrame(),
        context_df=context,
        orderbook=OrderBookFeatures(
            3.214,
            3.215,
            3.2145,
            3.11,
            0.55,
            bid_depth=20,
            ask_depth=20,
            depth_pressure=0.1,
            age_seconds=1,
            source="live",
        ),
        trade_flow=TradeFlowFeatures(),
        config=config,
        now=now,
    )

    assert signal.side == Side.FLAT
    assert signal.reason.startswith("ema_adx_macd_filters_not_met")
