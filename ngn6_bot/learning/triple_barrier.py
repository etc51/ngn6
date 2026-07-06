from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import pandas as pd


EntryLabel = Literal["long", "short", "flat"]
BarrierType = Literal["tp", "sl", "time", "unmatured"]
ExitLabel = Literal["hold", "exit_now"]


@dataclass(frozen=True)
class TripleBarrierConfig:
    tick_size: float
    tick_value: float
    commission_roundtrip_bps: float = 8.0
    slippage_bps_per_side: float = 4.0
    atr_stop_multiple: float = 1.2
    take_profit_r_multiple: float = 2.0
    time_barrier_bars: int = 12
    min_trainable_r: float = 0.20
    min_edge_r: float = 0.15
    min_stop_ticks: int = 1


@dataclass(frozen=True)
class TripleBarrierResult:
    label: EntryLabel
    long_net_r: float
    short_net_r: float
    long_barrier: BarrierType
    short_barrier: BarrierType
    horizon_start: datetime | None
    horizon_end: datetime | None
    matured: bool


@dataclass(frozen=True)
class ExitLabelResult:
    label: ExitLabel
    hold_net_r: float
    exit_now_r: float
    horizon_start: datetime | None
    horizon_end: datetime | None
    matured: bool


def label_entry_decision(
    *,
    decision_timestamp: datetime,
    future_ohlcv: pd.DataFrame,
    config: TripleBarrierConfig,
    entry_price: float | None = None,
    atr: float | None = None,
) -> TripleBarrierResult:
    path = _future_path(decision_timestamp, future_ohlcv, config.time_barrier_bars)
    if len(path) < config.time_barrier_bars:
        return TripleBarrierResult(
            label="flat",
            long_net_r=0.0,
            short_net_r=0.0,
            long_barrier="unmatured",
            short_barrier="unmatured",
            horizon_start=_first_timestamp(path),
            horizon_end=_last_timestamp(path),
            matured=False,
        )

    price = _entry_price(path, entry_price)
    stop_distance = _stop_distance(path, price, atr, config)
    long_net_r, long_barrier = _path_net_r(
        path,
        side="long",
        entry_price=price,
        stop_distance=stop_distance,
        config=config,
    )
    short_net_r, short_barrier = _path_net_r(
        path,
        side="short",
        entry_price=price,
        stop_distance=stop_distance,
        config=config,
    )

    if (
        long_net_r >= config.min_trainable_r
        and long_net_r > short_net_r + config.min_edge_r
    ):
        label: EntryLabel = "long"
    elif (
        short_net_r >= config.min_trainable_r
        and short_net_r > long_net_r + config.min_edge_r
    ):
        label = "short"
    else:
        label = "flat"

    return TripleBarrierResult(
        label=label,
        long_net_r=long_net_r,
        short_net_r=short_net_r,
        long_barrier=long_barrier,
        short_barrier=short_barrier,
        horizon_start=_first_timestamp(path),
        horizon_end=_last_timestamp(path),
        matured=True,
    )


def label_exit_decision(
    *,
    decision_timestamp: datetime,
    future_ohlcv: pd.DataFrame,
    side: Literal["long", "short"],
    config: TripleBarrierConfig,
    entry_price: float,
    current_price: float | None = None,
    atr: float | None = None,
) -> ExitLabelResult:
    path = _future_path(decision_timestamp, future_ohlcv, config.time_barrier_bars)
    if len(path) < config.time_barrier_bars:
        return ExitLabelResult(
            label="hold",
            hold_net_r=0.0,
            exit_now_r=0.0,
            horizon_start=_first_timestamp(path),
            horizon_end=_last_timestamp(path),
            matured=False,
        )

    price = current_price if current_price is not None else _entry_price(path, None)
    stop_distance = _stop_distance(path, price, atr, config)
    hold_net_r, _ = _path_net_r(
        path,
        side=side,
        entry_price=price,
        stop_distance=stop_distance,
        config=config,
    )
    label: ExitLabel = "hold" if hold_net_r >= config.min_trainable_r else "exit_now"
    return ExitLabelResult(
        label=label,
        hold_net_r=hold_net_r,
        exit_now_r=0.0,
        horizon_start=_first_timestamp(path),
        horizon_end=_last_timestamp(path),
        matured=True,
    )


def _future_path(
    decision_timestamp: datetime,
    ohlcv: pd.DataFrame,
    bars: int,
) -> pd.DataFrame:
    if ohlcv.empty:
        return ohlcv
    frame = ohlcv.sort_index()
    if isinstance(frame.index, pd.DatetimeIndex):
        timestamp = pd.Timestamp(decision_timestamp)
        if timestamp.tzinfo is None and frame.index.tz is not None:
            timestamp = timestamp.tz_localize(frame.index.tz)
        elif timestamp.tzinfo is not None and frame.index.tz is None:
            timestamp = timestamp.tz_convert("UTC").tz_localize(None)
        frame = frame.loc[frame.index > timestamp]
    return frame.head(max(0, int(bars)))


def _entry_price(path: pd.DataFrame, entry_price: float | None) -> float:
    if entry_price is not None:
        return float(entry_price)
    return float(path.iloc[0]["open"] if "open" in path.columns else path.iloc[0]["close"])


def _stop_distance(
    path: pd.DataFrame,
    entry_price: float,
    atr: float | None,
    config: TripleBarrierConfig,
) -> float:
    atr_value = float(atr) if atr is not None else 0.0
    if atr_value <= 0 and "atr" in path.columns:
        atr_value = float(path.iloc[0].get("atr") or 0.0)
    if atr_value <= 0:
        atr_value = max(entry_price * 0.002, config.tick_size)
    min_stop = max(1, config.min_stop_ticks) * config.tick_size
    return max(atr_value * config.atr_stop_multiple, min_stop)


def _path_net_r(
    path: pd.DataFrame,
    *,
    side: Literal["long", "short"],
    entry_price: float,
    stop_distance: float,
    config: TripleBarrierConfig,
) -> tuple[float, BarrierType]:
    cost_r = _cost_r(entry_price, stop_distance, config)
    take_profit_distance = stop_distance * config.take_profit_r_multiple
    last_close = float(path.iloc[-1]["close"])

    for _, row in path.iterrows():
        high = float(row["high"])
        low = float(row["low"])
        if side == "long":
            if low <= entry_price - stop_distance:
                return -1.0 - cost_r, "sl"
            if high >= entry_price + take_profit_distance:
                return config.take_profit_r_multiple - cost_r, "tp"
        else:
            if high >= entry_price + stop_distance:
                return -1.0 - cost_r, "sl"
            if low <= entry_price - take_profit_distance:
                return config.take_profit_r_multiple - cost_r, "tp"

    gross = (
        (last_close - entry_price) / stop_distance
        if side == "long"
        else (entry_price - last_close) / stop_distance
    )
    return gross - cost_r, "time"


def _cost_r(entry_price: float, stop_distance: float, config: TripleBarrierConfig) -> float:
    total_bps = config.commission_roundtrip_bps + 2 * config.slippage_bps_per_side
    cost_price = entry_price * total_bps / 10_000
    return cost_price / max(stop_distance, config.tick_size)


def _first_timestamp(path: pd.DataFrame) -> datetime | None:
    if path.empty:
        return None
    value = path.index[0]
    return value.to_pydatetime() if hasattr(value, "to_pydatetime") else None


def _last_timestamp(path: pd.DataFrame) -> datetime | None:
    if path.empty:
        return None
    value = path.index[-1]
    return value.to_pydatetime() if hasattr(value, "to_pydatetime") else None
