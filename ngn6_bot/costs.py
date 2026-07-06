from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionCostConfig:
    slippage_bps_assumption: float
    commission_per_lot_per_side: float
    commission_round_trip_bps: float
    min_expected_net_ticks: float
    min_price_increment: float
    money_value_per_price_step: float


@dataclass(frozen=True)
class CostCheck:
    accepted: bool
    expected_move_ticks: float
    round_trip_cost_ticks: float
    min_required_ticks: float
    reason: str


def expected_move_ticks(price: float, expected_move_pct: float, min_price_increment: float) -> float:
    if price <= 0 or expected_move_pct <= 0 or min_price_increment <= 0:
        return 0.0
    return (price * expected_move_pct / 100) / min_price_increment


def round_trip_cost_ticks(
    price: float,
    spread_bps: float | None,
    config: ExecutionCostConfig,
) -> float:
    if price <= 0 or config.min_price_increment <= 0:
        return 0.0

    spread_ticks = _bps_to_ticks(price, max(spread_bps or 0.0, 0.0), config.min_price_increment)
    slippage_ticks = _bps_to_ticks(
        price,
        max(config.slippage_bps_assumption, 0.0),
        config.min_price_increment,
    )
    commission_ticks = _commission_to_ticks(config)
    commission_bps_ticks = _bps_to_ticks(
        price,
        max(config.commission_round_trip_bps, 0.0),
        config.min_price_increment,
    )
    return spread_ticks + 2 * slippage_ticks + 2 * commission_ticks + commission_bps_ticks


def trade_covers_costs(
    *,
    price: float,
    expected_move_pct: float,
    spread_bps: float | None,
    config: ExecutionCostConfig,
) -> CostCheck:
    expected_ticks = expected_move_ticks(price, expected_move_pct, config.min_price_increment)
    cost_ticks = round_trip_cost_ticks(price, spread_bps, config)
    required_ticks = cost_ticks + max(config.min_expected_net_ticks, 0.0)
    if expected_ticks <= 0:
        return CostCheck(False, expected_ticks, cost_ticks, required_ticks, "expected_move_is_zero")
    if expected_ticks < required_ticks:
        return CostCheck(False, expected_ticks, cost_ticks, required_ticks, "expected_move_below_costs")
    return CostCheck(True, expected_ticks, cost_ticks, required_ticks, "costs_covered")


def cost_pct(price: float, cost_ticks: float, min_price_increment: float) -> float:
    if price <= 0 or min_price_increment <= 0:
        return 0.0
    return cost_ticks * min_price_increment / price * 100


def _bps_to_ticks(price: float, bps: float, min_price_increment: float) -> float:
    return (price * bps / 10_000) / min_price_increment


def _commission_to_ticks(config: ExecutionCostConfig) -> float:
    if config.money_value_per_price_step <= 0:
        return 0.0
    return max(config.commission_per_lot_per_side, 0.0) / config.money_value_per_price_step
