from __future__ import annotations

import pandas as pd

from ngn6_bot.models import Candle


def candles_to_frame(candles: list[Candle]) -> pd.DataFrame:
    rows = [
        {
            "timestamp": c.timestamp,
            "open": c.open,
            "high": c.high,
            "low": c.low,
            "close": c.close,
            "volume": c.volume,
        }
        for c in candles
    ]
    if not rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    return pd.DataFrame(rows).set_index("timestamp").sort_index()


def add_indicators(
    frame: pd.DataFrame,
    ema_fast: int = 5,
    ema_slow: int = 10,
    rsi_period: int = 14,
    atr_period: int = 14,
    adx_period: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    bollinger_period: int = 20,
    bollinger_std: float = 2.0,
    volume_ma_period: int = 20,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    df = frame.copy()
    df["ema_fast"] = df["close"].ewm(span=ema_fast, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=ema_slow, adjust=False).mean()
    df["rsi"] = rsi(df["close"], period=rsi_period)
    df["atr"] = atr(df, period=atr_period)
    df["adx"] = adx(df, period=adx_period)
    macd_line, macd_signal_line, macd_hist = macd(
        df["close"],
        fast=macd_fast,
        slow=macd_slow,
        signal=macd_signal,
    )
    df["macd"] = macd_line
    df["macd_signal"] = macd_signal_line
    df["macd_hist"] = macd_hist

    basis = df["close"].rolling(bollinger_period).mean()
    deviation = df["close"].rolling(bollinger_period).std(ddof=0)
    df["bb_mid"] = basis
    df["bb_upper"] = basis + bollinger_std * deviation
    df["bb_lower"] = basis - bollinger_std * deviation
    df["bb_width_pct"] = ((df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]) * 100
    df["volume_ma"] = df["volume"].rolling(volume_ma_period).mean()
    return df


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    result = 100 - (100 / (1 + rs))
    return result.fillna(50.0)


def atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)

    high_low = frame["high"] - frame["low"]
    high_close = (frame["high"] - frame["close"].shift(1)).abs()
    low_close = (frame["low"] - frame["close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.ewm(alpha=1 / period, min_periods=period, adjust=False).mean().fillna(0.0)


def adx(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)

    up_move = frame["high"].diff()
    down_move = -frame["low"].diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    true_range = atr(frame, period=period)
    plus_di = 100 * (
        plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        / true_range.where(true_range != 0)
    )
    minus_di = 100 * (
        minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        / true_range.where(true_range != 0)
    )
    denominator = plus_di + minus_di
    dx = 100 * (plus_di - minus_di).abs() / denominator.where(denominator != 0)
    return dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean().fillna(0.0)


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    fast_ema = series.ewm(span=fast, adjust=False).mean()
    slow_ema = series.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist
