from __future__ import annotations

import bisect
import csv
import json
import logging
import math
import statistics
from collections import defaultdict
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable
from zoneinfo import ZoneInfo

from ngn6_bot.charting import fetch_day_candles
from ngn6_bot.config import RuntimeConfig
from ngn6_bot.indicators import add_indicators, candles_to_frame
from ngn6_bot.runtime_metadata import current_commit_hash


DEFAULT_CONFIDENCE_THRESHOLDS = (0.50, 0.55, 0.60, 0.62, 0.65, 0.66, 0.68, 0.70, 0.72, 0.75)


def run_strategy_audit(
    config: RuntimeConfig,
    *,
    events_path: str | Path | None = None,
    decisions_path: str | Path | None = None,
    market_path: str | Path | None = None,
    output_dir: str | Path = "reports/strategy_audit",
    fetch_candles: bool = False,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Build a read-only forensic report for every completed paper trade."""
    events = Path(events_path or config.get("paper", "events_file"))
    decisions = Path(decisions_path or config.get("data_collection", "decisions_file"))
    market = Path(market_path or config.get("data_collection", "market_structure_file"))
    destination = Path(output_dir)

    trades = pair_paper_events(events, config)
    if not trades:
        raise RuntimeError(f"No completed paper trades found in {events}.")

    _attach_decisions(trades, decisions, config)
    _attach_market_path(trades, market, config)
    if fetch_candles:
        _attach_candle_context(trades, config, logger or logging.getLogger(__name__))
    _finalize_trade_fields(trades, config)

    summary = build_strategy_summary(trades, config)
    summary.update(
        {
            "schema_version": 1,
            "generated_at": datetime.now().astimezone().isoformat(),
            "ticker": config.get("instrument", "ticker"),
            "analyzer_commit_hash": current_commit_hash(),
            "sources": {
                "paper_events": _file_metadata(events),
                "decisions": _file_metadata(decisions),
                "market_structure": _file_metadata(market),
                "api_candles_fetched": fetch_candles,
            },
        }
    )

    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / "paper_trade_forensics.csv"
    json_path = destination / "strategy_audit_summary.json"
    _write_trade_csv(csv_path, trades)
    summary["outputs"] = {
        "trades_csv": str(csv_path),
        "summary_json": str(json_path),
    }
    json_path.write_text(
        json.dumps(_json_safe(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return _json_safe(summary)


def pair_paper_events(path: str | Path, config: RuntimeConfig) -> list[dict[str, Any]]:
    min_increment = float(config.get("instrument", "min_price_increment"))
    step_value = float(config.get("instrument", "money_value_per_price_step"))
    timezone = ZoneInfo(config.timezone)
    events = _read_jsonl(Path(path))
    trades: list[dict[str, Any]] = []
    opened: dict[str, Any] | None = None

    for event in events:
        event_type = str(event.get("event", ""))
        if event_type == "paper_open":
            if opened is not None:
                raise RuntimeError("Nested paper_open event found before the previous close.")
            opened = {"event": event, "closes": []}
            continue
        if event_type != "paper_close":
            continue
        if opened is None:
            raise RuntimeError("paper_close event found without a matching paper_open.")

        opened["closes"].append(event)
        exit_ = event.get("details", {})
        if int(exit_.get("remaining_lots", 0)) != 0:
            continue

        open_event = opened["event"]
        close_events = opened["closes"]
        entry = open_event.get("details", {})
        close_details = [item.get("details", {}) for item in close_events]
        opened_at = _parse_timestamp(open_event.get("timestamp"))
        closed_at = _parse_timestamp(event.get("timestamp"))
        if opened_at is None or closed_at is None:
            raise RuntimeError("Paper trade contains an invalid timestamp.")

        lots = int(entry.get("lots", 0))
        entry_price = float(entry.get("price", 0.0))
        closed_lots = sum(int(item.get("lots", 0)) for item in close_details)
        exit_price = _safe_div(
            sum(float(item.get("price", 0.0)) * int(item.get("lots", 0)) for item in close_details),
            closed_lots,
        )
        gross_pnl = sum(float(item.get("gross_pnl", 0.0)) for item in close_details)
        open_commission = float(entry.get("commission", 0.0))
        close_commission = sum(float(item.get("commission", 0.0)) for item in close_details)
        net_pnl = sum(float(item.get("realized_pnl", 0.0)) for item in close_details) - open_commission
        side = str(entry.get("side", ""))
        side_sign = 1.0 if side == "long" else -1.0
        stop_price = _optional_float(entry.get("stop_price"))
        risk_ticks = (
            abs(entry_price - stop_price) / min_increment
            if stop_price is not None and min_increment > 0
            else None
        )
        risk_rub = risk_ticks * step_value * lots if risk_ticks is not None else None
        local_open = opened_at.astimezone(timezone)
        entry_reason = str(entry.get("reason", ""))
        exit_reason = str(exit_.get("reason", ""))

        trade = {
            "trade_id": len(trades) + 1,
            "opened_at": opened_at.isoformat(),
            "closed_at": closed_at.isoformat(),
            "opened_at_msk": local_open.isoformat(),
            "trading_date_msk": local_open.date().isoformat(),
            "weekday_msk": local_open.strftime("%A"),
            "hour_msk": local_open.hour,
            "side": side,
            "side_sign": side_sign,
            "lots": lots,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "stop_price": stop_price,
            "take_profit1": _optional_float(entry.get("take_profit1")),
            "take_profit2": _optional_float(entry.get("take_profit2")),
            "entry_reason": entry_reason,
            "entry_target": _reason_target(entry_reason),
            "exit_reason": exit_reason,
            "exit_type": exit_reason.split(":", 1)[0],
            "partial_close_count": max(0, len(close_events) - 1),
            "all_exit_reasons": ";".join(str(item.get("reason", "")) for item in close_details),
            "confidence": _reason_confidence(entry_reason),
            "duration_seconds": (closed_at - opened_at).total_seconds(),
            "gross_pnl_rub": gross_pnl,
            "open_commission_rub": open_commission,
            "close_commission_rub": close_commission,
            "total_commission_rub": open_commission + close_commission,
            "net_pnl_rub": net_pnl,
            "gross_ticks_per_lot": _safe_div(gross_pnl, step_value * lots),
            "net_ticks_per_lot": _safe_div(net_pnl, step_value * lots),
            "initial_risk_ticks": risk_ticks,
            "initial_risk_rub": risk_rub,
            "net_r_multiple": _safe_div(net_pnl, risk_rub),
            "event_open_commit_hash": open_event.get("commit_hash"),
            "event_close_commit_hash": event.get("commit_hash"),
            "_opened_at": opened_at,
            "_closed_at": closed_at,
        }
        trades.append(trade)
        opened = None

    if opened is not None:
        raise RuntimeError("Unclosed paper_open event remains at the end of the event file.")
    return trades


def build_strategy_summary(
    trades: list[dict[str, Any]],
    config: RuntimeConfig,
) -> dict[str, Any]:
    overall = trade_metrics(trades)
    adjusted = trade_metrics(trades, pnl_key="realistic_net_pnl_rub")
    thresholds = {
        f"{threshold:.2f}": trade_metrics(
            [row for row in trades if _number(row.get("confidence"), -1.0) >= threshold]
        )
        for threshold in DEFAULT_CONFIDENCE_THRESHOLDS
    }

    regimes = {
        "day": _group_metrics(trades, lambda row: row.get("trading_date_msk")),
        "side": _group_metrics(trades, lambda row: row.get("side")),
        "hour": _group_metrics(trades, lambda row: row.get("hour_msk")),
        "entry_target": _group_metrics(trades, lambda row: row.get("entry_target")),
        "exit_type": _group_metrics(trades, lambda row: row.get("exit_type")),
        "adx_15m": _group_metrics(trades, lambda row: _adx_bucket(row.get("candle_15min_adx"))),
        "spread": _group_metrics(
            trades, lambda row: _spread_bucket(row.get("entry_spread_ticks"))
        ),
        "volatility_15m": _group_metrics(
            trades, lambda row: _volatility_bucket(row.get("candle_15min_atr_pct"))
        ),
        "time_of_day": _group_metrics(trades, _time_bucket),
        "book_pressure": _group_metrics(
            trades, lambda row: _pressure_bucket(row.get("entry_aligned_depth_pressure"))
        ),
        "candle_age_1m": _group_metrics(
            trades, lambda row: _candle_age_bucket(row.get("candle_1min_age_minutes"), 1)
        ),
    }

    filter_rows = _filter_candidate_metrics(trades)
    stable_positive = [
        {"filter": name, **metrics}
        for name, metrics in filter_rows.items()
        if metrics.get("stability_gate_passed")
    ]

    net_losses = [row for row in trades if _number(row.get("net_pnl_rub")) < 0]
    gross_winners = [row for row in trades if _number(row.get("gross_pnl_rub")) > 0]
    cost_flips = [
        row
        for row in trades
        if _number(row.get("gross_pnl_rub")) > 0
        and _number(row.get("net_pnl_rub")) <= 0
    ]
    flip_exits = [row for row in trades if row.get("exit_type") == "ml_exit_opposite"]
    hard_stops = [row for row in trades if row.get("exit_type") == "hard_stop_hit"]
    very_short = [row for row in trades if _number(row.get("duration_seconds")) < 60]
    rapid_reentries = [
        row
        for row in trades
        if row.get("seconds_since_previous_close") is not None
        and _number(row.get("seconds_since_previous_close")) < 60
    ]

    return {
        "trade_period": {
            "first_open": trades[0]["opened_at"],
            "last_close": trades[-1]["closed_at"],
            "completed_trades": len(trades),
        },
        "reconciliation": {
            "paper_net_pnl_rub": overall["net_pnl_rub"],
            "paper_gross_pnl_rub": overall["gross_pnl_rub"],
            "paper_commissions_rub": overall["cost_rub"],
            "gross_plus_cost_equals_net": math.isclose(
                _number(overall.get("gross_pnl_rub")) - _number(overall.get("cost_rub")),
                _number(overall.get("net_pnl_rub")),
                abs_tol=0.01,
            ),
        },
        "overall": overall,
        "realistic_execution": {
            **adjusted,
            "extra_spread_and_slippage_rub": sum(
                _number(row.get("extra_execution_cost_rub")) for row in trades
            ),
            "assumed_slippage_bps_per_side": float(
                config.get("execution", "slippage_bps_assumption", default=0.0)
            ),
            "note": (
                "Paper fills use the recorded decision price. The adjusted result additionally "
                "charges executable bid/ask and configured per-side slippage."
            ),
        },
        "failure_mechanics": {
            "net_losing_trades": len(net_losses),
            "gross_winning_trades": len(gross_winners),
            "gross_winners_turned_net_loss_by_commission": len(cost_flips),
            "ml_flip_exits": len(flip_exits),
            "ml_flip_exit_net_pnl_rub": sum(_number(row.get("net_pnl_rub")) for row in flip_exits),
            "hard_stops": len(hard_stops),
            "hard_stop_net_pnl_rub": sum(_number(row.get("net_pnl_rub")) for row in hard_stops),
            "hard_stops_under_10_seconds": sum(
                _number(row.get("duration_seconds")) < 10 for row in hard_stops
            ),
            "hard_stops_already_breached_before_entry": sum(
                bool(row.get("stop_breached_before_entry")) for row in hard_stops
            ),
            "all_entries_with_stop_already_breached": sum(
                bool(row.get("stop_breached_before_entry")) for row in trades
            ),
            "trades_under_60_seconds": len(very_short),
            "rapid_reentries_under_60_seconds": len(rapid_reentries),
            "max_consecutive_losses": _max_loss_streak(trades),
            "mfe_covered_paper_cost_but_trade_lost": sum(
                bool(row.get("mfe_covered_paper_cost")) and _number(row.get("net_pnl_rub")) <= 0
                for row in trades
            ),
        },
        "data_quality": {
            "matched_entry_decisions": sum(row.get("entry_decision_matched") is True for row in trades),
            "matched_exit_decisions": sum(row.get("exit_decision_matched") is True for row in trades),
            "market_path_covered": sum(_number(row.get("path_market_records")) > 0 for row in trades),
            "entry_trade_flow_all_unknown": sum(
                _number(row.get("entry_trade_flow_unknown_share"), 1.0) >= 0.999999
                for row in trades
            ),
            "outside_configured_session": sum(bool(row.get("outside_configured_session")) for row in trades),
            "stale_1m_context_over_5min": sum(
                row.get("candle_1min_age_minutes") is not None
                and _number(row.get("candle_1min_age_minutes")) > 5
                for row in trades
            ),
            "missing_runtime_commit_hash": sum(not row.get("runtime_commit_hash") for row in trades),
        },
        "confidence_thresholds": thresholds,
        "regimes": regimes,
        "entry_filter_counterfactuals": filter_rows,
        "stable_positive_filters": stable_positive,
        "cooldown_counterfactuals": _cooldown_counterfactuals(trades),
        "risk_lock_counterfactuals": _risk_lock_counterfactuals(trades, config),
        "top_losses": [_compact_trade(row) for row in sorted(trades, key=_net_pnl)[:15]],
        "top_wins": [
            _compact_trade(row) for row in sorted(trades, key=_net_pnl, reverse=True)[:15]
        ],
        "limitations": [
            "Paper trades are clustered in four calendar days; this is diagnostic evidence, not a promotion sample.",
            "Counterfactual filters select among observed trades and do not reconstruct signals suppressed by earlier choices.",
            "Historical order-book rows are sampled snapshots, not an exchange queue replay.",
            "Tick-rule trade direction is an estimate because the recorded broker trade side is unknown.",
            "A filter is marked stable only with at least 30 trades, three trading days, PF >= 1.25, positive expectancy, and >= 70% positive days.",
        ],
    }


def trade_metrics(
    rows: Iterable[dict[str, Any]],
    *,
    pnl_key: str = "net_pnl_rub",
) -> dict[str, Any]:
    selected = list(rows)
    if not selected:
        return {
            "trades": 0,
            "net_pnl_rub": 0.0,
            "gross_pnl_rub": 0.0,
            "cost_rub": 0.0,
            "expectancy_rub": None,
            "profit_factor": None,
            "win_rate": None,
            "avg_net_ticks_per_lot": None,
            "max_drawdown_rub": 0.0,
            "days": 0,
            "positive_day_share": None,
            "stability_gate_passed": False,
        }

    pnls = [_number(row.get(pnl_key)) for row in selected]
    wins = [value for value in pnls if value > 0]
    losses = [value for value in pnls if value < 0]
    gross_profit = sum(wins)
    gross_loss = -sum(losses)
    day_pnl: dict[str, float] = defaultdict(float)
    for row, pnl in zip(selected, pnls):
        day_pnl[str(row.get("trading_date_msk"))] += pnl
    positive_days = sum(value > 0 for value in day_pnl.values())
    days = len(day_pnl)
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (math.inf if wins else None)
    result = {
        "trades": len(selected),
        "net_pnl_rub": sum(pnls),
        "gross_pnl_rub": sum(_number(row.get("gross_pnl_rub")) for row in selected),
        "cost_rub": sum(_number(row.get("total_commission_rub")) for row in selected),
        "expectancy_rub": statistics.mean(pnls),
        "profit_factor": profit_factor,
        "win_rate": len(wins) / len(selected),
        "avg_win_rub": statistics.mean(wins) if wins else None,
        "avg_loss_rub": statistics.mean(losses) if losses else None,
        "median_pnl_rub": statistics.median(pnls),
        "avg_net_ticks_per_lot": statistics.mean(
            _number(
                row.get(
                    "realistic_net_ticks_per_lot"
                    if pnl_key == "realistic_net_pnl_rub"
                    else "net_ticks_per_lot"
                )
            )
            for row in selected
        ),
        "median_duration_minutes": statistics.median(
            _number(row.get("duration_seconds")) for row in selected
        )
        / 60.0,
        "max_drawdown_rub": _max_drawdown(pnls),
        "days": days,
        "positive_day_share": positive_days / days if days else None,
        "day_pnl_rub": dict(sorted(day_pnl.items())),
    }
    result["stability_gate_passed"] = bool(
        len(selected) >= 30
        and days >= 3
        and result["expectancy_rub"] > 0
        and profit_factor is not None
        and profit_factor >= 1.25
        and result["positive_day_share"] is not None
        and result["positive_day_share"] >= 0.70
    )
    return result


def _attach_decisions(
    trades: list[dict[str, Any]],
    path: Path,
    config: RuntimeConfig,
) -> None:
    accepted: dict[str, list[dict[str, Any]]] = {"open_accepted": [], "close_accepted": []}
    for row in _read_jsonl(path):
        action = row.get("action")
        if action in accepted:
            accepted[action].append(row)

    _match_decisions(trades, accepted["open_accepted"], "entry", config)
    _match_decisions(trades, accepted["close_accepted"], "exit", config)


def _match_decisions(
    trades: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    prefix: str,
    config: RuntimeConfig,
) -> None:
    timestamp_key = "_opened_at" if prefix == "entry" else "_closed_at"
    parsed = [(_parse_timestamp(row.get("timestamp")), row) for row in decisions]
    parsed = [(timestamp, row) for timestamp, row in parsed if timestamp is not None]
    parsed.sort(key=lambda item: item[0])
    timestamps = [item[0] for item in parsed]

    for trade in trades:
        target = trade[timestamp_key]
        index = bisect.bisect_left(timestamps, target)
        candidates = parsed[max(0, index - 1) : min(len(parsed), index + 2)]
        if not candidates:
            trade[f"{prefix}_decision_matched"] = False
            continue
        timestamp, row = min(candidates, key=lambda item: abs((item[0] - target).total_seconds()))
        delta_seconds = (timestamp - target).total_seconds()
        if abs(delta_seconds) > 2.0:
            trade[f"{prefix}_decision_matched"] = False
            trade[f"{prefix}_decision_delta_seconds"] = delta_seconds
            continue
        trade[f"{prefix}_decision_matched"] = True
        trade[f"{prefix}_decision_delta_seconds"] = delta_seconds
        trade.update(_decision_scalars(row, prefix, trade["side"], config))
        commit_hash = row.get("commit_hash")
        if commit_hash:
            trade["runtime_commit_hash"] = commit_hash


def _decision_scalars(
    decision: dict[str, Any],
    prefix: str,
    side: str,
    config: RuntimeConfig,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        f"{prefix}_decision_action": decision.get("action"),
        f"{prefix}_decision_reason": decision.get("reason"),
        f"{prefix}_decision_schema": decision.get("schema_version"),
        f"{prefix}_decision_price": decision.get("price"),
    }
    if prefix == "entry" and decision.get("confidence") is not None:
        result["confidence"] = float(decision["confidence"])

    context = decision.get("market_context") or {}
    book = context.get("orderbook") or {}
    trade_flow = context.get("trade_flow") or {}
    snapshot = context.get("orderbook_snapshot") or {}
    sign = 1.0 if side == "long" else -1.0
    min_increment = float(config.get("instrument", "min_price_increment"))

    for key in (
        "best_bid",
        "best_ask",
        "mid_price",
        "spread_bps",
        "bid_ask_imbalance",
        "bid_depth",
        "ask_depth",
        "depth_pressure",
        "best_bid_qty",
        "best_ask_qty",
        "mid_price_change_bps",
        "spread_change_bps",
        "imbalance_change",
        "bid_depth_change_pct",
        "ask_depth_change_pct",
        "age_seconds",
        "source",
    ):
        result[f"{prefix}_{key}"] = book.get(key)

    spread = _optional_float(book.get("best_ask"))
    best_bid = _optional_float(book.get("best_bid"))
    if spread is not None and best_bid is not None and min_increment > 0:
        result[f"{prefix}_spread_ticks"] = (spread - best_bid) / min_increment
    depth_pressure = _optional_float(book.get("depth_pressure"))
    result[f"{prefix}_aligned_depth_pressure"] = (
        sign * depth_pressure if depth_pressure is not None else None
    )

    bids = list(snapshot.get("bids") or [])
    asks = list(snapshot.get("asks") or [])
    for levels in (1, 3, 5, 10):
        pressure = _depth_pressure(bids[:levels], asks[:levels])
        result[f"{prefix}_book_pressure_l{levels}"] = pressure
        result[f"{prefix}_aligned_book_pressure_l{levels}"] = (
            sign * pressure if pressure is not None else None
        )
    result.update(_microprice_scalars(prefix, bids, asks, sign, min_increment))
    result.update(_wall_scalars(prefix, bids, asks))

    total_volume = _number(trade_flow.get("total_volume"))
    unknown_volume = _number(trade_flow.get("unknown_volume"))
    result[f"{prefix}_trade_flow_total_volume"] = total_volume
    result[f"{prefix}_trade_flow_unknown_volume"] = unknown_volume
    result[f"{prefix}_trade_flow_unknown_share"] = (
        unknown_volume / total_volume if total_volume > 0 else 1.0
    )
    result[f"{prefix}_trade_flow_buy_ratio"] = trade_flow.get("buy_ratio")
    result[f"{prefix}_trade_flow_imbalance"] = trade_flow.get("buy_sell_imbalance")
    result[f"{prefix}_trade_count"] = trade_flow.get("trade_count")
    result[f"{prefix}_average_trade_size"] = trade_flow.get("average_trade_size")
    result[f"{prefix}_trade_flow_source"] = trade_flow.get("source")
    tick_rule = _tick_rule_flow(context.get("recent_trades") or [])
    result[f"{prefix}_tick_rule_pressure"] = tick_rule["pressure"]
    result[f"{prefix}_aligned_tick_rule_pressure"] = sign * tick_rule["pressure"]
    result[f"{prefix}_tick_rule_classified_share"] = tick_rule["classified_share"]

    if prefix == "entry":
        metadata = decision.get("metadata") or {}
        feedback = metadata.get("feedback") or {}
        features = metadata.get("features") or {}
        result["model_target"] = feedback.get("target")
        result["model_score"] = feedback.get("score")
        result["model_examples"] = feedback.get("examples")
        scores: list[float] = []
        for item in feedback.get("nearest") or []:
            model = str(item.get("model", "unknown"))
            score = _optional_float(item.get("score"))
            if score is not None:
                scores.append(score)
                result[f"model_score_{model}"] = score
        if scores:
            result["model_score_min"] = min(scores)
            result["model_score_max"] = max(scores)
            result["model_score_mean"] = statistics.mean(scores)
            result["model_score_std"] = statistics.pstdev(scores)
        for key, value in features.items():
            if isinstance(value, (int, float, bool)) or value is None:
                result[f"feature_{key}"] = value
    return result


def _attach_market_path(
    trades: list[dict[str, Any]],
    path: Path,
    config: RuntimeConfig,
) -> None:
    aggregates = [
        {"prices": [(row["_opened_at"], row["entry_price"]), (row["_closed_at"], row["exit_price"])],
         "spread": [], "aligned_pressure": [], "unknown_share": [], "trusted": []}
        for row in trades
    ]
    index = 0
    for market_row in _read_jsonl(path):
        timestamp = _parse_timestamp(market_row.get("timestamp"))
        if timestamp is None:
            continue
        if timestamp < trades[0]["_opened_at"]:
            continue
        while index < len(trades) and timestamp > trades[index]["_closed_at"]:
            index += 1
        if index >= len(trades):
            break
        trade = trades[index]
        if timestamp < trade["_opened_at"]:
            continue
        aggregate = aggregates[index]
        price = _optional_float(market_row.get("last_price"))
        if price is not None:
            aggregate["prices"].append((timestamp, price))
        book = market_row.get("orderbook") or {}
        spread = _optional_float(book.get("spread_bps"))
        if spread is not None:
            aggregate["spread"].append(spread)
        pressure = _optional_float(book.get("depth_pressure"))
        if pressure is not None:
            aggregate["aligned_pressure"].append(trade["side_sign"] * pressure)
        flow = market_row.get("trade_flow") or {}
        total = _number(flow.get("total_volume"))
        unknown = _number(flow.get("unknown_volume"))
        aggregate["unknown_share"].append(unknown / total if total > 0 else 1.0)
        if market_row.get("market_data_trusted") is not None:
            aggregate["trusted"].append(bool(market_row.get("market_data_trusted")))

    min_increment = float(config.get("instrument", "min_price_increment"))
    for trade, aggregate in zip(trades, aggregates):
        prices = aggregate["prices"]
        sign = trade["side_sign"]
        excursions = [sign * (price - trade["entry_price"]) / min_increment for _, price in prices]
        mfe_index = max(range(len(excursions)), key=excursions.__getitem__)
        mae_index = min(range(len(excursions)), key=excursions.__getitem__)
        trade["path_market_records"] = max(0, len(prices) - 2)
        trade["mfe_ticks"] = max(0.0, excursions[mfe_index])
        trade["mae_ticks"] = abs(min(0.0, excursions[mae_index]))
        trade["time_to_mfe_seconds"] = (
            prices[mfe_index][0] - trade["_opened_at"]
        ).total_seconds()
        trade["time_to_mae_seconds"] = (
            prices[mae_index][0] - trade["_opened_at"]
        ).total_seconds()
        trade["path_avg_spread_bps"] = _mean_or_none(aggregate["spread"])
        trade["path_max_spread_bps"] = max(aggregate["spread"], default=None)
        trade["path_avg_aligned_depth_pressure"] = _mean_or_none(
            aggregate["aligned_pressure"]
        )
        trade["path_avg_unknown_trade_share"] = _mean_or_none(aggregate["unknown_share"])
        trade["path_trusted_share"] = _mean_or_none(
            [1.0 if value else 0.0 for value in aggregate["trusted"]]
        )


def _attach_candle_context(
    trades: list[dict[str, Any]],
    config: RuntimeConfig,
    logger: logging.Logger,
) -> None:
    timezone = ZoneInfo(config.timezone)
    first_date = trades[0]["_opened_at"].astimezone(timezone).date() - timedelta(days=5)
    last_date = trades[-1]["_opened_at"].astimezone(timezone).date()
    frames: dict[str, Any] = {}
    for timeframe in ("1min", "5min", "15min"):
        candles = []
        current = first_date
        while current <= last_date:
            _, day_candles = fetch_day_candles(config, logger, current, timeframe)
            candles.extend(day_candles)
            current += timedelta(days=1)
        by_timestamp = {candle.timestamp: candle for candle in candles}
        frame = add_indicators(
            candles_to_frame(list(by_timestamp.values())),
            ema_fast=int(config.get("indicators", "ema_fast")),
            ema_slow=int(config.get("indicators", "ema_slow")),
            rsi_period=int(config.get("indicators", "rsi_period")),
            atr_period=int(config.get("indicators", "atr_period")),
            adx_period=int(config.get("indicators", "adx_period")),
            macd_fast=int(config.get("indicators", "macd_fast")),
            macd_slow=int(config.get("indicators", "macd_slow")),
            macd_signal=int(config.get("indicators", "macd_signal")),
            bollinger_period=int(config.get("indicators", "bollinger_period")),
            bollinger_std=float(config.get("indicators", "bollinger_std")),
            volume_ma_period=int(config.get("indicators", "volume_ma_period")),
        )
        frames[timeframe] = frame

    for trade in trades:
        for timeframe, frame in frames.items():
            _attach_one_candle(trade, timeframe, frame)
        _attach_pre_entry_bar_extremes(trade, frames["1min"])


def _attach_one_candle(trade: dict[str, Any], timeframe: str, frame: Any) -> None:
    if frame.empty:
        return
    minutes = int(timeframe.replace("min", ""))
    complete_times = [timestamp.to_pydatetime() + timedelta(minutes=minutes) for timestamp in frame.index]
    index = bisect.bisect_right(complete_times, trade["_opened_at"]) - 1
    if index < 0:
        return
    row = frame.iloc[index]
    candle_timestamp = frame.index[index].to_pydatetime()
    completed_at = complete_times[index]
    prefix = f"candle_{timeframe}_"
    close = _number(row.get("close"))
    ema_fast = _number(row.get("ema_fast"))
    ema_slow = _number(row.get("ema_slow"))
    atr = _number(row.get("atr"))
    volume_ma = _number(row.get("volume_ma"))
    bb_mid = _number(row.get("bb_mid"))
    bb_upper = _number(row.get("bb_upper"))
    bb_lower = _number(row.get("bb_lower"))
    trade.update(
        {
            prefix + "timestamp": candle_timestamp.isoformat(),
            prefix + "completed_at": completed_at.isoformat(),
            prefix + "age_minutes": (trade["_opened_at"] - completed_at).total_seconds() / 60.0,
            prefix + "open": _number(row.get("open")),
            prefix + "high": _number(row.get("high")),
            prefix + "low": _number(row.get("low")),
            prefix + "close": close,
            prefix + "volume": _number(row.get("volume")),
            prefix + "return_pct": _safe_div(close - _number(row.get("open")), _number(row.get("open"))) * 100,
            prefix + "ema_fast": ema_fast,
            prefix + "ema_slow": ema_slow,
            prefix + "trend_pct": _safe_div(ema_fast - ema_slow, close) * 100,
            prefix + "trend_aligned": trade["side_sign"] * (ema_fast - ema_slow),
            prefix + "rsi": _number(row.get("rsi")),
            prefix + "atr": atr,
            prefix + "atr_pct": _safe_div(atr, close) * 100,
            prefix + "adx": _number(row.get("adx")),
            prefix + "plus_di": _number(row.get("plus_di")),
            prefix + "minus_di": _number(row.get("minus_di")),
            prefix + "di_aligned": trade["side_sign"]
            * (_number(row.get("plus_di")) - _number(row.get("minus_di"))),
            prefix + "macd": _number(row.get("macd")),
            prefix + "macd_signal": _number(row.get("macd_signal")),
            prefix + "macd_hist": _number(row.get("macd_hist")),
            prefix + "macd_aligned": trade["side_sign"] * _number(row.get("macd_hist")),
            prefix + "bb_width_pct": _number(row.get("bb_width_pct")),
            prefix + "bb_position": _safe_div(close - bb_lower, bb_upper - bb_lower),
            prefix + "volume_ratio": _safe_div(_number(row.get("volume")), volume_ma),
            prefix + "bb_mid": bb_mid,
        }
    )


def _attach_pre_entry_bar_extremes(trade: dict[str, Any], one_minute_frame: Any) -> None:
    if one_minute_frame.empty:
        return
    opened_at = trade["_opened_at"]
    bucket_start = opened_at.replace(
        minute=(opened_at.minute // 15) * 15,
        second=0,
        microsecond=0,
    )
    completed = one_minute_frame.loc[
        (one_minute_frame.index >= bucket_start)
        & (one_minute_frame.index + timedelta(minutes=1) <= opened_at)
    ]
    if completed.empty:
        return
    high = float(completed["high"].max())
    low = float(completed["low"].min())
    stop_price = _optional_float(trade.get("stop_price"))
    trade["pre_entry_15min_high"] = high
    trade["pre_entry_15min_low"] = low
    trade["pre_entry_15min_completed_candles"] = len(completed)
    if stop_price is not None:
        trade["stop_breached_before_entry"] = (
            low <= stop_price if trade["side"] == "long" else high >= stop_price
        )


def _finalize_trade_fields(trades: list[dict[str, Any]], config: RuntimeConfig) -> None:
    min_increment = float(config.get("instrument", "min_price_increment"))
    step_value = float(config.get("instrument", "money_value_per_price_step"))
    slippage_bps = float(config.get("execution", "slippage_bps_assumption", default=0.0))
    timezone = ZoneInfo(config.timezone)
    start_time = time.fromisoformat(str(config.get("session", "trading_start")))
    end_time = time.fromisoformat(str(config.get("session", "trading_end")))

    previous: dict[str, Any] | None = None
    for trade in trades:
        entry_half = _executable_half_spread_ticks(trade, "entry", min_increment, entering=True)
        exit_half = _executable_half_spread_ticks(trade, "exit", min_increment, entering=False)
        if entry_half is None and trade.get("entry_spread_ticks") is not None:
            entry_half = _number(trade.get("entry_spread_ticks")) / 2.0
        if exit_half is None and trade.get("entry_spread_ticks") is not None:
            exit_half = _number(trade.get("entry_spread_ticks")) / 2.0
        executable_spread_ticks = _number(entry_half) + _number(exit_half)
        average_price = (trade["entry_price"] + trade["exit_price"]) / 2.0
        slippage_ticks = average_price * (2.0 * slippage_bps / 10_000.0) / min_increment
        added_ticks = executable_spread_ticks + slippage_ticks
        extra_cost = added_ticks * step_value * trade["lots"]
        trade["executable_spread_cost_ticks"] = executable_spread_ticks
        trade["slippage_cost_ticks"] = slippage_ticks
        trade["extra_execution_cost_ticks"] = added_ticks
        trade["extra_execution_cost_rub"] = extra_cost
        trade["realistic_net_pnl_rub"] = trade["net_pnl_rub"] - extra_cost
        trade["realistic_net_ticks_per_lot"] = _safe_div(
            trade["realistic_net_pnl_rub"], step_value * trade["lots"]
        )
        commission_ticks = _safe_div(trade["total_commission_rub"], step_value * trade["lots"])
        trade["paper_commission_ticks"] = commission_ticks
        trade["paper_break_even_ticks"] = commission_ticks
        trade["realistic_break_even_ticks"] = commission_ticks + added_ticks
        trade["mfe_covered_paper_cost"] = _number(trade.get("mfe_ticks")) >= commission_ticks
        trade["mfe_covered_realistic_cost"] = (
            _number(trade.get("mfe_ticks")) >= commission_ticks + added_ticks
        )
        trade["gross_winner_net_loser"] = (
            trade["gross_pnl_rub"] > 0 and trade["net_pnl_rub"] <= 0
        )
        trade["exit_capture_of_mfe"] = _safe_div(
            trade["gross_ticks_per_lot"], _number(trade.get("mfe_ticks"))
        )
        local_time = trade["_opened_at"].astimezone(timezone).timetz().replace(tzinfo=None)
        trade["outside_configured_session"] = not (start_time <= local_time <= end_time)
        trade["seconds_since_previous_close"] = (
            (trade["_opened_at"] - previous["_closed_at"]).total_seconds()
            if previous is not None
            else None
        )
        trade["previous_trade_was_loss"] = (
            previous is not None and _number(previous.get("net_pnl_rub")) < 0
        )
        trade["failure_flags"] = ";".join(_failure_flags(trade))
        previous = trade


def _filter_candidate_metrics(trades: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    predicates: dict[str, Callable[[dict[str, Any]], bool]] = {
        "configured_session_only": lambda row: not bool(row.get("outside_configured_session")),
        "confidence_ge_0.62": lambda row: _number(row.get("confidence"), -1.0) >= 0.62,
        "confidence_ge_0.68": lambda row: _number(row.get("confidence"), -1.0) >= 0.68,
        "aligned_depth_pressure_ge_0": lambda row: _number(
            row.get("entry_aligned_depth_pressure"), -9.0
        )
        >= 0,
        "aligned_depth_pressure_ge_0.18": lambda row: _number(
            row.get("entry_aligned_depth_pressure"), -9.0
        )
        >= 0.18,
        "aligned_tick_rule_pressure_ge_0": lambda row: _number(
            row.get("entry_aligned_tick_rule_pressure"), -9.0
        )
        >= 0,
        "adx_15m_ge_23": lambda row: _number(row.get("candle_15min_adx"), -1.0) >= 23,
        "trend_5m_aligned": lambda row: _number(row.get("candle_5min_trend_aligned"), -1.0) > 0,
        "trend_15m_aligned": lambda row: _number(row.get("candle_15min_trend_aligned"), -1.0) > 0,
        "fresh_1m_candle_le_5m": lambda row: _number(
            row.get("candle_1min_age_minutes"), 1e9
        )
        <= 5,
    }
    predicates.update(
        {
            "session_conf68": lambda row: predicates["configured_session_only"](row)
            and predicates["confidence_ge_0.68"](row),
            "session_book18": lambda row: predicates["configured_session_only"](row)
            and predicates["aligned_depth_pressure_ge_0.18"](row),
            "session_adx23_trend15": lambda row: predicates["configured_session_only"](row)
            and predicates["adx_15m_ge_23"](row)
            and predicates["trend_15m_aligned"](row),
            "session_conf62_book0_trend15": lambda row: predicates[
                "configured_session_only"
            ](row)
            and predicates["confidence_ge_0.62"](row)
            and predicates["aligned_depth_pressure_ge_0"](row)
            and predicates["trend_15m_aligned"](row),
            "strict_quality_gate": lambda row: predicates["configured_session_only"](row)
            and predicates["fresh_1m_candle_le_5m"](row)
            and predicates["confidence_ge_0.68"](row)
            and predicates["aligned_depth_pressure_ge_0.18"](row)
            and predicates["aligned_tick_rule_pressure_ge_0"](row)
            and predicates["adx_15m_ge_23"](row)
            and predicates["trend_5m_aligned"](row)
            and predicates["trend_15m_aligned"](row),
        }
    )
    return {name: trade_metrics([row for row in trades if predicate(row)]) for name, predicate in predicates.items()}


def _cooldown_counterfactuals(trades: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {}
    for loss_minutes in (5, 10, 30, 45, 60):
        selected = []
        next_allowed: datetime | None = None
        for row in trades:
            if next_allowed is not None and row["_opened_at"] < next_allowed:
                continue
            selected.append(row)
            cooldown = loss_minutes if _number(row.get("net_pnl_rub")) < 0 else 10
            next_allowed = row["_closed_at"] + timedelta(minutes=cooldown)
        result[f"loss_{loss_minutes}m_win_10m"] = trade_metrics(selected)
    return result


def _risk_lock_counterfactuals(
    trades: list[dict[str, Any]],
    config: RuntimeConfig,
) -> dict[str, dict[str, Any]]:
    max_losses = int(config.get("risk", "stop_after_consecutive_losses", default=3))
    daily_loss_limit = float(config.get("risk", "daily_max_loss_pct", default=0.01)) * float(
        config.get("paper", "initial_cash", default=300_000)
    )
    selected_streak = []
    selected_daily = []
    for _, rows in _rows_by_day(trades).items():
        streak = 0
        daily_pnl = 0.0
        streak_locked = False
        daily_locked = False
        for row in rows:
            if not streak_locked:
                selected_streak.append(row)
                if _number(row.get("net_pnl_rub")) < 0:
                    streak += 1
                else:
                    streak = 0
                streak_locked = streak >= max_losses
            if not daily_locked:
                selected_daily.append(row)
                daily_pnl += _number(row.get("net_pnl_rub"))
                daily_locked = daily_pnl <= -daily_loss_limit
    return {
        f"stop_after_{max_losses}_consecutive_losses_per_day": trade_metrics(selected_streak),
        f"daily_loss_limit_{daily_loss_limit:.0f}_rub": trade_metrics(selected_daily),
    }


def _group_metrics(
    trades: list[dict[str, Any]],
    key: Callable[[dict[str, Any]], Any],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trades:
        grouped[str(key(row))].append(row)
    return {name: trade_metrics(rows) for name, rows in sorted(grouped.items())}


def _rows_by_day(trades: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trades:
        grouped[str(row.get("trading_date_msk"))].append(row)
    return dict(grouped)


def _compact_trade(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "trade_id",
        "opened_at_msk",
        "side",
        "entry_target",
        "confidence",
        "exit_type",
        "duration_seconds",
        "gross_ticks_per_lot",
        "net_ticks_per_lot",
        "net_pnl_rub",
        "realistic_net_pnl_rub",
        "mfe_ticks",
        "mae_ticks",
        "entry_aligned_depth_pressure",
        "entry_aligned_tick_rule_pressure",
        "candle_15min_adx",
        "candle_15min_atr_pct",
        "stop_breached_before_entry",
        "failure_flags",
    )
    return {key: row.get(key) for key in keys}


def _failure_flags(row: dict[str, Any]) -> list[str]:
    flags = []
    if row.get("outside_configured_session"):
        flags.append("outside_session")
    if row.get("candle_1min_age_minutes") is not None and _number(
        row.get("candle_1min_age_minutes")
    ) > 5:
        flags.append("stale_1m_candle")
    if _number(row.get("entry_trade_flow_unknown_share"), 0.0) >= 0.999999:
        flags.append("trade_flow_direction_unknown")
    if _number(row.get("confidence"), 1.0) < 0.68:
        flags.append("confidence_below_0.68")
    if _number(row.get("entry_aligned_depth_pressure"), 1.0) < 0:
        flags.append("book_pressure_against_side")
    if row.get("exit_type") == "ml_exit_opposite" and _number(row.get("duration_seconds")) < 120:
        flags.append("fast_signal_flip_exit")
    if row.get("exit_type") == "hard_stop_hit":
        flags.append("hard_stop")
    if row.get("stop_breached_before_entry"):
        flags.append("stop_already_breached_before_entry")
    if row.get("gross_winner_net_loser"):
        flags.append("commission_flipped_result")
    if _number(row.get("gross_pnl_rub")) < 0:
        flags.append("direction_wrong")
    return flags


def _microprice_scalars(
    prefix: str,
    bids: list[dict[str, Any]],
    asks: list[dict[str, Any]],
    sign: float,
    min_increment: float,
) -> dict[str, Any]:
    if not bids or not asks:
        return {}
    bid = _number(bids[0].get("price"))
    ask = _number(asks[0].get("price"))
    bid_qty = _number(bids[0].get("quantity"))
    ask_qty = _number(asks[0].get("quantity"))
    total = bid_qty + ask_qty
    if total <= 0 or min_increment <= 0:
        return {}
    mid = (bid + ask) / 2.0
    microprice = (ask * bid_qty + bid * ask_qty) / total
    edge_ticks = (microprice - mid) / min_increment
    return {
        f"{prefix}_microprice": microprice,
        f"{prefix}_microprice_edge_ticks": edge_ticks,
        f"{prefix}_aligned_microprice_edge_ticks": sign * edge_ticks,
    }


def _wall_scalars(
    prefix: str,
    bids: list[dict[str, Any]],
    asks: list[dict[str, Any]],
) -> dict[str, Any]:
    result = {}
    for side, levels in (("bid", bids), ("ask", asks)):
        quantities = [_number(level.get("quantity")) for level in levels if level.get("quantity")]
        if not quantities:
            continue
        median = statistics.median(quantities)
        result[f"{prefix}_{side}_max_qty"] = max(quantities)
        result[f"{prefix}_{side}_wall_ratio"] = _safe_div(max(quantities), median)
    return result


def _tick_rule_flow(trades: list[dict[str, Any]]) -> dict[str, float]:
    previous_price: float | None = None
    previous_sign = 0.0
    signed_volume = 0.0
    classified_volume = 0.0
    total_volume = 0.0
    for row in trades:
        price = _optional_float(row.get("price"))
        quantity = max(0.0, _number(row.get("quantity")))
        if price is None or quantity <= 0:
            continue
        total_volume += quantity
        if previous_price is not None:
            if price > previous_price:
                previous_sign = 1.0
            elif price < previous_price:
                previous_sign = -1.0
        if previous_sign != 0:
            signed_volume += previous_sign * quantity
            classified_volume += quantity
        previous_price = price
    return {
        "pressure": signed_volume / classified_volume if classified_volume > 0 else 0.0,
        "classified_share": classified_volume / total_volume if total_volume > 0 else 0.0,
    }


def _executable_half_spread_ticks(
    trade: dict[str, Any],
    prefix: str,
    min_increment: float,
    *,
    entering: bool,
) -> float | None:
    bid = _optional_float(trade.get(f"{prefix}_best_bid"))
    ask = _optional_float(trade.get(f"{prefix}_best_ask"))
    observed = _optional_float(trade.get(f"{prefix}_decision_price"))
    if observed is None:
        observed = trade["entry_price" if entering else "exit_price"]
    if bid is None or ask is None or min_increment <= 0:
        return None
    if trade["side"] == "long":
        executable = ask if entering else bid
        adverse = executable - observed if entering else observed - executable
    else:
        executable = bid if entering else ask
        adverse = observed - executable if entering else executable - observed
    return max(0.0, adverse / min_increment)


def _depth_pressure(
    bids: list[dict[str, Any]],
    asks: list[dict[str, Any]],
) -> float | None:
    bid_qty = sum(_number(level.get("quantity")) for level in bids)
    ask_qty = sum(_number(level.get("quantity")) for level in asks)
    total = bid_qty + ask_qty
    return (bid_qty - ask_qty) / total if total > 0 else None


def _adx_bucket(value: Any) -> str:
    number = _optional_float(value)
    if number is None:
        return "missing"
    if number < 15:
        return "00_lt_15"
    if number < 23:
        return "01_15_23"
    if number < 30:
        return "02_23_30"
    return "03_ge_30"


def _spread_bucket(value: Any) -> str:
    number = _optional_float(value)
    if number is None:
        return "missing"
    if number <= 1:
        return "00_le_1_tick"
    if number <= 2:
        return "01_1_2_ticks"
    return "02_gt_2_ticks"


def _volatility_bucket(value: Any) -> str:
    number = _optional_float(value)
    if number is None:
        return "missing"
    if number < 0.05:
        return "00_lt_0.05pct"
    if number < 0.10:
        return "01_0.05_0.10pct"
    if number < 0.20:
        return "02_0.10_0.20pct"
    return "03_ge_0.20pct"


def _pressure_bucket(value: Any) -> str:
    number = _optional_float(value)
    if number is None:
        return "missing"
    if number < -0.18:
        return "00_against_lt_-0.18"
    if number < 0:
        return "01_against_-0.18_0"
    if number < 0.18:
        return "02_aligned_0_0.18"
    return "03_aligned_ge_0.18"


def _time_bucket(row: dict[str, Any]) -> str:
    hour = int(_number(row.get("hour_msk")))
    if row.get("outside_configured_session"):
        return "00_outside_session"
    if hour < 12:
        return "01_10_12"
    if hour < 14:
        return "02_12_14"
    if hour < 16:
        return "03_14_16"
    if hour < 19:
        return "04_16_19"
    return "05_19_late"


def _candle_age_bucket(value: Any, timeframe_minutes: int) -> str:
    number = _optional_float(value)
    if number is None:
        return "missing"
    if number <= timeframe_minutes:
        return "00_fresh"
    if number <= 5 * timeframe_minutes:
        return "01_delayed"
    if number <= 60:
        return "02_stale_under_1h"
    return "03_stale_over_1h"


def _reason_target(reason: str) -> str:
    parts = reason.split(":")
    return parts[1] if len(parts) >= 2 else reason


def _reason_confidence(reason: str) -> float | None:
    try:
        return float(reason.rsplit(":", 1)[-1])
    except ValueError:
        return None


def _max_drawdown(pnls: Iterable[float]) -> float:
    equity = 0.0
    peak = 0.0
    maximum = 0.0
    for pnl in pnls:
        equity += pnl
        peak = max(peak, equity)
        maximum = max(maximum, peak - equity)
    return maximum


def _max_loss_streak(trades: Iterable[dict[str, Any]]) -> int:
    maximum = 0
    current = 0
    for row in trades:
        if _number(row.get("net_pnl_rub")) < 0:
            current += 1
            maximum = max(maximum, current)
        else:
            current = 0
    return maximum


def _net_pnl(row: dict[str, Any]) -> float:
    return _number(row.get("net_pnl_rub"))


def _file_metadata(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(),
    }


def _write_trade_csv(path: Path, trades: list[dict[str, Any]]) -> None:
    internal = {"_opened_at", "_closed_at"}
    preferred = [
        "trade_id",
        "opened_at_msk",
        "closed_at",
        "side",
        "lots",
        "entry_price",
        "exit_price",
        "entry_target",
        "confidence",
        "exit_type",
        "duration_seconds",
        "gross_pnl_rub",
        "total_commission_rub",
        "net_pnl_rub",
        "realistic_net_pnl_rub",
        "gross_ticks_per_lot",
        "net_ticks_per_lot",
        "mfe_ticks",
        "mae_ticks",
        "failure_flags",
        "runtime_commit_hash",
    ]
    all_keys = {key for row in trades for key in row if key not in internal}
    columns = [key for key in preferred if key in all_keys]
    columns.extend(sorted(all_keys - set(columns)))
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in trades:
            writer.writerow({key: _csv_value(row.get(key)) for key in columns})


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                yield payload


def _parse_timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _optional_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _number(value: Any, default: float = 0.0) -> float:
    parsed = _optional_float(value)
    return parsed if parsed is not None else default


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _mean_or_none(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _csv_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float) and not math.isfinite(value):
        return "inf" if value > 0 else "-inf"
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items() if not str(key).startswith("_")}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, float):
        if not math.isfinite(value):
            return "Infinity" if value > 0 else "-Infinity"
        return round(value, 8)
    return value
