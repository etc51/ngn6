from __future__ import annotations

from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from ngn6_bot.config import RuntimeConfig
from ngn6_bot.indicators import add_indicators, candles_to_frame
from ngn6_bot.models import Candle
from ngn6_bot.tbank import TInvestGateway, candle_interval_for_polling


def fetch_day_candles(
    config: RuntimeConfig,
    logger,
    trading_date: date,
    timeframe: str,
) -> tuple[str, list[Candle]]:
    tz = ZoneInfo(config.timezone)
    start_local = datetime.combine(
        trading_date,
        time.fromisoformat(config.get("session", "trading_start")),
        tz,
    )
    end_local = datetime.combine(
        trading_date,
        time.fromisoformat(config.get("session", "trading_end")),
        tz,
    )

    with TInvestGateway(config.token, config.raw, logger) as gateway:
        figi, _ = gateway.resolve_instrument()
        response = gateway.client.market_data.get_candles(
            figi=figi,
            from_=start_local.astimezone(timezone.utc),
            to=end_local.astimezone(timezone.utc),
            interval=candle_interval_for_polling(timeframe),
        )
        candles = [
            Candle(
                timestamp=candle.time,
                open=_quotation_to_float(candle.open),
                high=_quotation_to_float(candle.high),
                low=_quotation_to_float(candle.low),
                close=_quotation_to_float(candle.close),
                volume=float(candle.volume),
                timeframe=timeframe,
            )
            for candle in response.candles
            if getattr(candle, "is_complete", True)
        ]
        return figi, candles


def plot_indicator_chart(
    config: RuntimeConfig,
    candles: list[Candle],
    trading_date: date,
    timeframe: str,
    output: str | Path,
    decisions: list[dict[str, Any]] | None = None,
    regimes: list[dict[str, Any]] | None = None,
    backtest_trades: list[dict[str, Any]] | None = None,
    title_suffix: str = "",
) -> Path:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    if not candles:
        raise RuntimeError(f"No {timeframe} candles for {trading_date.isoformat()}.")

    df = add_indicators(
        candles_to_frame(candles),
        ema_fast=int(config.get("indicators", "ema_fast")),
        ema_slow=int(config.get("indicators", "ema_slow")),
        rsi_period=int(config.get("indicators", "rsi_period")),
        bollinger_period=int(config.get("indicators", "bollinger_period")),
        bollinger_std=float(config.get("indicators", "bollinger_std")),
        volume_ma_period=int(config.get("indicators", "volume_ma_period")),
    )

    tz = ZoneInfo(config.timezone)
    local_index = df.index.tz_convert(tz)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(18, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [5, 1.6, 1.6]},
    )
    fig.patch.set_facecolor("#101418")
    for axis in axes:
        axis.set_facecolor("#101418")
        axis.grid(True, color="#2b343d", linewidth=0.7, alpha=0.7)
        axis.tick_params(colors="#d6dde5")
        for spine in axis.spines.values():
            spine.set_color("#38434f")

    price_axis, rsi_axis, volume_axis = axes
    ema_fast = int(config.get("indicators", "ema_fast"))
    ema_slow = int(config.get("indicators", "ema_slow"))
    price_axis.plot(local_index, df["close"], color="#f3f4f6", linewidth=1.4, label="Close")
    price_axis.plot(local_index, df["ema_fast"], color="#38bdf8", linewidth=1.1, label=f"EMA{ema_fast}")
    price_axis.plot(local_index, df["ema_slow"], color="#f59e0b", linewidth=1.1, label=f"EMA{ema_slow}")
    price_axis.plot(local_index, df["bb_upper"], color="#94a3b8", linewidth=0.9, alpha=0.8, label="BB upper")
    price_axis.plot(local_index, df["bb_lower"], color="#94a3b8", linewidth=0.9, alpha=0.8, label="BB lower")
    price_axis.fill_between(
        local_index,
        df["bb_lower"].astype(float).to_numpy(),
        df["bb_upper"].astype(float).to_numpy(),
        color="#334155",
        alpha=0.2,
        label="Bollinger 20/2",
    )
    for start, end, color, alpha, label in _regime_spans(regimes or [], trading_date, tz):
        price_axis.axvspan(start, end, color=color, alpha=alpha, label=label)
    for marker_time, marker_price, marker, color, label, note in _decision_markers(
        decisions or [], trading_date, tz
    ):
        price_axis.axvline(marker_time, color=color, linewidth=1.0, alpha=0.45, zorder=3)
        price_axis.scatter(
            [marker_time],
            [marker_price],
            marker=marker,
            s=180,
            color=color,
            edgecolors="#101418",
            linewidths=1.2,
            zorder=7,
            label=label,
        )
        price_axis.annotate(
            note,
            xy=(marker_time, marker_price),
            xytext=(10, 18 if marker in {"^", "o"} else -34),
            textcoords="offset points",
            color="#f8fafc",
            fontsize=8.5,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.28", "facecolor": "#101418", "edgecolor": color, "alpha": 0.92},
            arrowprops={"arrowstyle": "->", "color": color, "linewidth": 1.0},
            zorder=8,
        )
    if backtest_trades is not None:
        _draw_backtest_trades(price_axis, backtest_trades, trading_date, tz)

    suffix = f" | {title_suffix}" if title_suffix else ""
    price_axis.set_title(
        f"{config.get('instrument', 'ticker')} {trading_date.isoformat()} {timeframe} | EMA{ema_fast}/EMA{ema_slow}, Bollinger, RSI14{suffix}",
        color="#f8fafc",
        fontsize=16,
        pad=14,
    )
    price_axis.legend(loc="upper left", ncols=5, fontsize=9, frameon=False, labelcolor="#d6dde5")

    rsi_axis.plot(local_index, df["rsi"], color="#22c55e", linewidth=1.2, label="RSI14")
    rsi_axis.axhline(float(config.get("indicators", "rsi_overbought")), color="#ef4444", linewidth=0.9)
    rsi_axis.axhline(float(config.get("indicators", "rsi_oversold")), color="#60a5fa", linewidth=0.9)
    rsi_axis.set_ylim(0, 100)
    rsi_axis.legend(loc="upper left", fontsize=9, frameon=False, labelcolor="#d6dde5")

    colors = ["#22c55e" if close >= open_ else "#ef4444" for open_, close in zip(df["open"], df["close"])]
    volume_axis.bar(local_index, df["volume"], color=colors, width=_bar_width(timeframe), alpha=0.75)
    volume_axis.plot(local_index, df["volume_ma"], color="#facc15", linewidth=1.0, label="Volume MA20")
    volume_axis.legend(loc="upper left", fontsize=9, frameon=False, labelcolor="#d6dde5")

    volume_axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=tz))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _bar_width(timeframe: str) -> float:
    return {"1min": 1 / 1440, "5min": 5 / 1440, "15min": 15 / 1440}.get(timeframe, 1 / 1440)


def _decision_markers(
    decisions: list[dict[str, Any]],
    trading_date: date,
    tz: ZoneInfo,
) -> list[tuple[datetime, float, str, str, str, str]]:
    markers: list[tuple[datetime, float, str, str, str, str]] = []
    seen_labels: set[str] = set()
    for item in decisions:
        action = str(item.get("action", ""))
        if action not in {"open_accepted", "close_accepted", "partial_accepted"}:
            continue
        timestamp = _parse_timestamp(item.get("timestamp"))
        price = item.get("price")
        if timestamp is None or price is None:
            continue
        local_time = timestamp.astimezone(tz)
        if local_time.date() != trading_date:
            continue
        side = str(item.get("side", ""))
        if action == "close_accepted":
            marker, color, label = "x", "#facc15", "Close"
            note = f"EXIT {str(item.get('reason', '')).replace('_', ' ').upper()}\n{local_time:%H:%M} @ {float(price):.3f}"
        elif side == "short":
            marker, color, label = "v", "#ef4444", "Short"
            note = f"ENTRY SHORT\n{local_time:%H:%M} @ {float(price):.3f}"
        else:
            marker, color, label = "^", "#22c55e", "Long"
            note = f"ENTRY LONG\n{local_time:%H:%M} @ {float(price):.3f}"
        if label in seen_labels:
            label = "_nolegend_"
        else:
            seen_labels.add(label)
        markers.append((local_time, float(price), marker, color, label, note))
    return markers


def _draw_backtest_trades(axis, trades: list[dict[str, Any]], trading_date: date, tz: ZoneInfo) -> None:
    day_trades = _backtest_trades_for_date(trades, trading_date, tz)
    if not day_trades:
        axis.text(
            0.995,
            0.92,
            "NO BACKTEST TRADE",
            transform=axis.transAxes,
            ha="right",
            va="center",
            color="#94a3b8",
            fontsize=10,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "#101418",
                "edgecolor": "#94a3b8",
                "alpha": 0.9,
            },
            zorder=9,
        )
        return

    seen_labels: set[str] = set()
    for trade in day_trades:
        number = trade.get("number", "?")
        side = str(trade.get("side", "")).lower()
        pnl_pct = _float_or_none(trade.get("pnl_pct"))
        result_color = "#22c55e" if pnl_pct is not None and pnl_pct > 0 else "#ef4444"
        entry_time = _parse_timestamp(trade.get("entry_time"))
        exit_time = _parse_timestamp(trade.get("exit_time"))
        entry_price = _float_or_none(trade.get("entry_price"))
        exit_price = _float_or_none(trade.get("exit_price"))

        entry_local = entry_time.astimezone(tz) if entry_time else None
        exit_local = exit_time.astimezone(tz) if exit_time else None

        if (
            entry_local is not None
            and exit_local is not None
            and entry_price is not None
            and exit_price is not None
            and entry_local.date() == trading_date
            and exit_local.date() == trading_date
        ):
            axis.plot(
                [entry_local, exit_local],
                [entry_price, exit_price],
                color=result_color,
                linewidth=1.2,
                alpha=0.55,
                zorder=5,
            )

        if entry_local is not None and entry_local.date() == trading_date and entry_price is not None:
            marker = "v" if side == "short" else "^"
            color = "#ef4444" if side == "short" else "#22c55e"
            label = _unique_label("BT short entry" if side == "short" else "BT long entry", seen_labels)
            axis.axvline(entry_local, color=color, linewidth=0.8, alpha=0.28, zorder=3)
            axis.scatter(
                [entry_local],
                [entry_price],
                marker=marker,
                s=150,
                color=color,
                edgecolors="#101418",
                linewidths=1.1,
                zorder=9,
                label=label,
            )
            _annotate_trade_marker(axis, entry_local, entry_price, f"E{number}", color, 18)

        if exit_local is not None and exit_local.date() == trading_date and exit_price is not None:
            label = _unique_label("BT exit", seen_labels)
            axis.axvline(exit_local, color="#facc15", linewidth=0.8, alpha=0.28, zorder=3)
            axis.scatter(
                [exit_local],
                [exit_price],
                marker="x",
                s=150,
                color="#facc15",
                linewidths=1.8,
                zorder=9,
                label=label,
            )
            _annotate_trade_marker(axis, exit_local, exit_price, f"X{number}", "#facc15", -30)

    summary = _backtest_trade_summary(day_trades, tz)
    axis.text(
        0.995,
        0.88,
        summary,
        transform=axis.transAxes,
        ha="right",
        va="top",
        color="#d6dde5",
        fontsize=8.2,
        linespacing=1.35,
        family="monospace",
        bbox={
            "boxstyle": "round,pad=0.42",
            "facecolor": "#101418",
            "edgecolor": "#475569",
            "alpha": 0.9,
        },
        zorder=10,
    )


def _backtest_trades_for_date(
    trades: list[dict[str, Any]],
    trading_date: date,
    tz: ZoneInfo,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for trade in trades:
        entry_time = _parse_timestamp(trade.get("entry_time"))
        exit_time = _parse_timestamp(trade.get("exit_time"))
        entry_date = entry_time.astimezone(tz).date() if entry_time else None
        exit_date = exit_time.astimezone(tz).date() if exit_time else None
        if trading_date in {entry_date, exit_date}:
            result.append(trade)
    return result


def _annotate_trade_marker(
    axis,
    marker_time: datetime,
    price: float,
    text: str,
    color: str,
    y_offset: int,
) -> None:
    axis.annotate(
        text,
        xy=(marker_time, price),
        xytext=(0, y_offset),
        textcoords="offset points",
        ha="center",
        color="#f8fafc",
        fontsize=8,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "#101418", "edgecolor": color, "alpha": 0.92},
        arrowprops={"arrowstyle": "-", "color": color, "linewidth": 0.9},
        zorder=10,
    )


def _backtest_trade_summary(trades: list[dict[str, Any]], tz: ZoneInfo) -> str:
    lines = ["BACKTEST TRADES", "#  side   in    out    pnl%"]
    max_rows = 10
    for trade in trades[:max_rows]:
        number = str(trade.get("number", "?")).rjust(2)
        side = str(trade.get("side", "")).upper()[:5].ljust(5)
        entry = _format_local_hhmm(trade.get("entry_time"), tz)
        exit_ = _format_local_hhmm(trade.get("exit_time"), tz)
        pnl = _float_or_none(trade.get("pnl_pct"))
        pnl_text = f"{pnl:+.2f}" if pnl is not None else " n/a"
        lines.append(f"{number} {side} {entry} {exit_} {pnl_text}")
    if len(trades) > max_rows:
        lines.append(f"... +{len(trades) - max_rows} more")
    return "\n".join(lines)


def _format_local_hhmm(value: Any, tz: ZoneInfo) -> str:
    timestamp = _parse_timestamp(value)
    if timestamp is None:
        return "--:--"
    return timestamp.astimezone(tz).strftime("%H:%M")


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _unique_label(label: str, seen_labels: set[str]) -> str:
    if label in seen_labels:
        return "_nolegend_"
    seen_labels.add(label)
    return label


def _regime_spans(
    regimes: list[dict[str, Any]],
    trading_date: date,
    tz: ZoneInfo,
) -> list[tuple[datetime, datetime, str, float, str]]:
    spans: list[tuple[datetime, datetime, str, float, str]] = []
    seen_labels: set[str] = set()
    for item in regimes:
        start = _parse_timestamp(item.get("start"))
        end = _parse_timestamp(item.get("end"))
        if start is None or end is None:
            continue
        start_local = start.astimezone(tz)
        end_local = end.astimezone(tz)
        if start_local.date() != trading_date and end_local.date() != trading_date:
            continue
        label = str(item.get("label", "Regime"))
        legend_label = label if label not in seen_labels else "_nolegend_"
        seen_labels.add(label)
        spans.append(
            (
                start_local,
                end_local,
                str(item.get("color", "#64748b")),
                float(item.get("alpha", 0.15)),
                legend_label,
            )
        )
    return spans


def _parse_timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _quotation_to_float(value) -> float:
    return float(value.units) + float(value.nano) / 1_000_000_000
