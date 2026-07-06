from __future__ import annotations

import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ngn6_bot.models import OrderBookFeatures, Side, Signal


def generate_legacy_signal(
    *,
    execution_df: pd.DataFrame,
    context_df: pd.DataFrame,
    orderbook: OrderBookFeatures,
    config: Any,
    now: datetime,
) -> Signal:
    timeframe = str(getattr(config, "legacy_timeframe", "intraday"))
    source_df = context_df if timeframe != "execution" else execution_df
    min_candles = int(getattr(config, "legacy_min_candles", 80))
    candles = _candles_from_frame(source_df, max_candles=int(getattr(config, "legacy_max_candles", 320)))
    if len(candles) < min_candles:
        return _flat(now, f"legacy_not_enough_candles:{len(candles)}/{min_candles}")

    current_price = _current_price(orderbook, execution_df, source_df)
    if current_price is None:
        return _flat(now, "legacy_no_current_price")

    bridge_path = Path(str(getattr(config, "legacy_bridge_path", ""))).expanduser()
    if not bridge_path.is_absolute():
        bridge_path = Path.cwd() / bridge_path
    if not bridge_path.exists():
        return _flat(now, f"legacy_bridge_missing:{bridge_path}")

    request = {
        "snapshot": _snapshot_from_candles(candles, timeframe=timeframe),
        "currentPrice": current_price,
        "manual": _manual_context(config),
        "impulseOptions": _impulse_options(config),
    }
    result = _call_bridge(
        node_command=str(getattr(config, "legacy_node_command", "node")),
        bridge_path=bridge_path,
        request=request,
        timeout_seconds=float(getattr(config, "legacy_timeout_seconds", 3.0)),
    )
    if not result.get("ok"):
        return _flat(now, f"legacy_bridge_failed:{result.get('error', 'unknown')}")

    payload = result.get("payload") or {}
    plan = result.get("plan") or {}
    probability = _finite_float(payload.get("probability"))
    min_probability = float(getattr(config, "legacy_min_probability", 52.0))
    if probability is None or probability < min_probability:
        return _flat(now, f"legacy_probability_too_low:{probability}")

    side = _plan_side(plan)
    if side == Side.FLAT:
        return _flat(now, f"legacy_no_executable_plan:{payload.get('signal', 'neutral')}")
    if plan.get("allowed") is False:
        return _flat(now, f"legacy_plan_not_allowed:{side.value}")
    if not bool(result.get("entryReached")):
        entry = _finite_float(plan.get("entry"))
        return _flat(now, f"legacy_entry_not_reached:{side.value}:{entry}")

    stop_price = _finite_float(plan.get("stop"))
    if stop_price is None:
        return _flat(now, "legacy_stop_missing")
    if side == Side.LONG and stop_price >= current_price:
        return _flat(now, f"legacy_invalid_long_stop:{stop_price}")
    if side == Side.SHORT and stop_price <= current_price:
        return _flat(now, f"legacy_invalid_short_stop:{stop_price}")

    take_profit1 = _finite_float(plan.get("takeProfit1"))
    take_profit2 = _finite_float(plan.get("takeProfit2"))
    origin = payload.get("signalOrigin") or payload.get("hybridMode") or "signal"
    entry = _finite_float(plan.get("entry"))
    metadata = {
        "engine": "legacy_ngn6",
        "signal": payload.get("signal"),
        "headline": payload.get("headline"),
        "origin": origin,
        "probability": probability,
        "score": payload.get("score"),
        "entry": entry,
        "stop": stop_price,
        "take_profit1": take_profit1,
        "take_profit2": take_profit2,
        "manual": result.get("manual"),
    }
    return Signal(
        side=side,
        confidence=probability / 100,
        reason=f"legacy_ngn6:{side.value}:prob={probability}:entry={entry}:origin={origin}",
        price=current_price,
        stop_price=stop_price,
        timestamp=now,
        take_profit1=take_profit1,
        take_profit2=take_profit2,
        metadata=metadata,
    )


def _call_bridge(
    *,
    node_command: str,
    bridge_path: Path,
    request: dict[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [node_command, str(bridge_path)],
            input=json.dumps(request, ensure_ascii=False),
            text=True,
            encoding="utf-8",
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout"}
    except OSError as exc:
        return {"ok": False, "error": "node_unavailable", "details": str(exc)}

    if completed.returncode != 0:
        return {
            "ok": False,
            "error": "node_exit",
            "details": completed.stderr.strip() or completed.stdout.strip(),
        }
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        return {"ok": False, "error": "invalid_bridge_json", "details": str(exc)}
    return payload if isinstance(payload, dict) else {"ok": False, "error": "invalid_bridge_payload"}


def _snapshot_from_candles(candles: list[dict[str, Any]], *, timeframe: str) -> dict[str, Any]:
    return {
        "timeframe": "intraday" if timeframe in {"intraday", "execution"} else timeframe,
        "timeframeLabel": "Intraday 15m",
        "forecastLabel": "next 15 minutes",
        "source": "python-tbank-bridge",
        "generatedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "gas": _market("NGN6", "NGN6 T-Bank", candles),
        "dxy": _neutral_market("DX-Y.NYB", "Neutral DXY bridge", candles, 100.0),
        "brent": _neutral_market("BZ=F", "Neutral Brent bridge", candles, 80.0),
    }


def _market(symbol: str, short_name: str, candles: list[dict[str, Any]]) -> dict[str, Any]:
    return {"symbol": symbol, "shortName": short_name, "candles": candles, "latest": candles[-1]}


def _neutral_market(
    symbol: str,
    short_name: str,
    source_candles: list[dict[str, Any]],
    price: float,
) -> dict[str, Any]:
    candles = [
        {
            "date": candle["date"],
            "timestamp": candle["timestamp"],
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": 0,
        }
        for candle in source_candles
    ]
    return _market(symbol, short_name, candles)


def _candles_from_frame(frame: pd.DataFrame, *, max_candles: int) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    candles: list[dict[str, Any]] = []
    for timestamp, row in frame.tail(max_candles).iterrows():
        open_ = _finite_float(row.get("open"))
        high = _finite_float(row.get("high"))
        low = _finite_float(row.get("low"))
        close = _finite_float(row.get("close"))
        if None in {open_, high, low, close}:
            continue
        ts = _timestamp_iso(timestamp)
        candles.append(
            {
                "date": ts[:16].replace("T", " ") + " UTC",
                "timestamp": ts,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": _finite_float(row.get("volume")) or 0,
            }
        )
    return candles


def _timestamp_iso(timestamp: Any) -> str:
    if hasattr(timestamp, "to_pydatetime"):
        value = timestamp.to_pydatetime()
    else:
        value = timestamp
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.isoformat() + "Z"
        return value.isoformat().replace("+00:00", "Z")
    return str(value)


def _current_price(
    orderbook: OrderBookFeatures,
    execution_df: pd.DataFrame,
    context_df: pd.DataFrame,
) -> float | None:
    for value in [
        orderbook.mid_price,
        _last_close(execution_df),
        _last_close(context_df),
    ]:
        number = _finite_float(value)
        if number is not None:
            return number
    return None


def _last_close(frame: pd.DataFrame) -> float | None:
    if frame.empty:
        return None
    return _finite_float(frame.iloc[-1].get("close"))


def _manual_context(config: Any) -> dict[str, str]:
    values = {
        "newsBias": str(getattr(config, "legacy_news_bias", "auto")),
        "retest": str(getattr(config, "legacy_retest", "auto")),
        "structure": str(getattr(config, "legacy_structure", "auto")),
        "eventRisk": str(getattr(config, "legacy_event_risk", "none")),
    }
    return {key: value for key, value in values.items() if value and value != "auto"}


def _impulse_options(config: Any) -> dict[str, Any]:
    return {
        "enabled": bool(getattr(config, "legacy_impulse_enabled", True)),
        "movePct": float(getattr(config, "legacy_impulse_move_pct", 1.2)),
        "breakoutBufferPct": float(getattr(config, "legacy_impulse_breakout_buffer_pct", 0.03)),
        "minTrend": float(getattr(config, "legacy_impulse_min_trend", 0.28)),
        "minMomentum": float(getattr(config, "legacy_impulse_min_momentum", 0.24)),
        "minCandles": int(getattr(config, "legacy_impulse_min_candles", 2)),
        "minProbability": float(getattr(config, "legacy_impulse_min_probability", 55.5)),
        "maxProbability": float(getattr(config, "legacy_impulse_max_probability", 63.5)),
    }


def _plan_side(plan: dict[str, Any]) -> Side:
    value = str(plan.get("side", "")).lower()
    if value == "long":
        return Side.LONG
    if value == "short":
        return Side.SHORT
    return Side.FLAT


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _flat(now: datetime, reason: str) -> Signal:
    return Signal(Side.FLAT, 0.0, reason, 0.0, None, now)
