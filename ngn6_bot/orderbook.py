from __future__ import annotations

from datetime import datetime

from ngn6_bot.models import OrderBookFeatures, OrderBookSnapshot


def analyze_order_book(
    current: OrderBookSnapshot | None,
    previous: OrderBookSnapshot | None,
    *,
    levels: int,
    wall_multiplier: float,
    absorption_drop_pct: float,
    min_wall_notional: float,
    now: datetime | None = None,
) -> OrderBookFeatures:
    if current is None or not current.bids or not current.asks:
        return OrderBookFeatures(
            best_bid=None,
            best_ask=None,
            mid_price=None,
            spread_bps=None,
            bid_ask_imbalance=0.5,
            source="missing",
        )

    best_bid = current.bids[0].price
    best_ask = current.asks[0].price
    mid_price = (best_bid + best_ask) / 2
    spread_bps = ((best_ask - best_bid) / mid_price) * 10000 if mid_price else None

    bid_qty = sum(level.quantity for level in current.bids[:levels])
    ask_qty = sum(level.quantity for level in current.asks[:levels])
    total_qty = bid_qty + ask_qty
    imbalance = bid_qty / total_qty if total_qty else 0.5
    depth_pressure = (imbalance - 0.5) * 2

    bid_wall = _find_wall(current.bids[:levels], wall_multiplier, min_wall_notional)
    ask_wall = _find_wall(current.asks[:levels], wall_multiplier, min_wall_notional)
    previous_depth = _depth_snapshot(previous, levels)

    bid_wall_absorbed = _wall_absorbed(previous, current, "bid", absorption_drop_pct)
    ask_wall_absorbed = _wall_absorbed(previous, current, "ask", absorption_drop_pct)

    return OrderBookFeatures(
        best_bid=best_bid,
        best_ask=best_ask,
        mid_price=mid_price,
        spread_bps=spread_bps,
        bid_ask_imbalance=imbalance,
        bid_depth=bid_qty,
        ask_depth=ask_qty,
        depth_pressure=depth_pressure,
        best_bid_qty=float(current.bids[0].quantity),
        best_ask_qty=float(current.asks[0].quantity),
        mid_price_change_bps=_change_bps(previous_depth["mid_price"], mid_price),
        spread_change_bps=_spread_change(previous_depth["spread_bps"], spread_bps),
        imbalance_change=imbalance - previous_depth["imbalance"],
        bid_depth_change_pct=_change_pct(previous_depth["bid_qty"], bid_qty),
        ask_depth_change_pct=_change_pct(previous_depth["ask_qty"], ask_qty),
        bid_wall_price=bid_wall["price"],
        bid_wall_qty=bid_wall["quantity"],
        bid_wall_notional=bid_wall["notional"],
        bid_wall_distance_bps=_wall_distance_bps(best_bid, bid_wall["price"], mid_price, "bid"),
        ask_wall_price=ask_wall["price"],
        ask_wall_qty=ask_wall["quantity"],
        ask_wall_notional=ask_wall["notional"],
        ask_wall_distance_bps=_wall_distance_bps(best_ask, ask_wall["price"], mid_price, "ask"),
        bid_wall_absorbed=bid_wall_absorbed,
        ask_wall_absorbed=ask_wall_absorbed,
        age_seconds=_age_seconds(current, now),
        source="live",
    )


def spread_is_acceptable(features: OrderBookFeatures, max_spread_bps: float) -> bool:
    return features.spread_bps is not None and features.spread_bps <= max_spread_bps


def _find_wall(levels, wall_multiplier: float, min_wall_notional: float) -> dict[str, float | None]:
    empty = {"price": None, "quantity": 0.0, "notional": 0.0}
    if not levels:
        return empty
    avg_qty = sum(level.quantity for level in levels) / len(levels)
    for level in levels:
        notional = level.price * level.quantity
        if level.quantity >= avg_qty * wall_multiplier and notional >= min_wall_notional:
            return {
                "price": level.price,
                "quantity": float(level.quantity),
                "notional": float(notional),
            }
    return empty


def _depth_snapshot(snapshot: OrderBookSnapshot | None, levels: int) -> dict[str, float]:
    if snapshot is None or not snapshot.bids or not snapshot.asks:
        return {
            "mid_price": 0.0,
            "spread_bps": 0.0,
            "bid_qty": 0.0,
            "ask_qty": 0.0,
            "imbalance": 0.5,
        }
    best_bid = snapshot.bids[0].price
    best_ask = snapshot.asks[0].price
    mid_price = (best_bid + best_ask) / 2
    spread_bps = ((best_ask - best_bid) / mid_price) * 10000 if mid_price else 0.0
    bid_qty = sum(level.quantity for level in snapshot.bids[:levels])
    ask_qty = sum(level.quantity for level in snapshot.asks[:levels])
    total_qty = bid_qty + ask_qty
    return {
        "mid_price": mid_price,
        "spread_bps": spread_bps,
        "bid_qty": bid_qty,
        "ask_qty": ask_qty,
        "imbalance": bid_qty / total_qty if total_qty else 0.5,
    }


def _change_bps(previous: float, current: float) -> float | None:
    if previous <= 0:
        return None
    return (current - previous) / previous * 10000


def _spread_change(previous: float, current: float | None) -> float | None:
    if current is None or previous <= 0:
        return None
    return current - previous


def _change_pct(previous: float, current: float) -> float | None:
    if previous <= 0:
        return None
    return (current - previous) / previous * 100


def _wall_distance_bps(
    best_price: float,
    wall_price: float | None,
    mid_price: float,
    side: str,
) -> float | None:
    if wall_price is None or mid_price <= 0:
        return None
    if side == "bid":
        return max(0.0, (best_price - wall_price) / mid_price * 10000)
    return max(0.0, (wall_price - best_price) / mid_price * 10000)


def _age_seconds(snapshot: OrderBookSnapshot, now: datetime | None) -> float | None:
    if now is None:
        return None
    timestamp = snapshot.timestamp
    if timestamp.tzinfo is None and now.tzinfo is not None:
        timestamp = timestamp.replace(tzinfo=now.tzinfo)
    return max(0.0, (now - timestamp).total_seconds())


def _wall_absorbed(
    previous: OrderBookSnapshot | None,
    current: OrderBookSnapshot,
    side: str,
    absorption_drop_pct: float,
) -> bool:
    if previous is None:
        return False

    previous_levels = previous.bids if side == "bid" else previous.asks
    current_levels = current.bids if side == "bid" else current.asks
    if not previous_levels or not current_levels:
        return False

    prev_top_price = previous_levels[0].price
    prev_top_qty = previous_levels[0].quantity
    current_same_price = next((level for level in current_levels if level.price == prev_top_price), None)

    if current_same_price is None:
        return True

    if prev_top_qty <= 0:
        return False
    drop_pct = ((prev_top_qty - current_same_price.quantity) / prev_top_qty) * 100
    return drop_pct >= absorption_drop_pct
