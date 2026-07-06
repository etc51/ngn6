from ngn6_bot.costs import ExecutionCostConfig, round_trip_cost_ticks, trade_covers_costs


def test_round_trip_costs_are_converted_to_ticks():
    config = ExecutionCostConfig(
        slippage_bps_assumption=8,
        commission_per_lot_per_side=1.5,
        commission_round_trip_bps=0,
        min_expected_net_ticks=3,
        min_price_increment=0.001,
        money_value_per_price_step=1.0,
    )

    ticks = round_trip_cost_ticks(3.3, 10, config)

    assert round(ticks, 2) == 11.58


def test_trade_is_blocked_when_expected_move_does_not_cover_costs():
    config = ExecutionCostConfig(
        slippage_bps_assumption=8,
        commission_per_lot_per_side=1.5,
        commission_round_trip_bps=0,
        min_expected_net_ticks=3,
        min_price_increment=0.001,
        money_value_per_price_step=1.0,
    )

    check = trade_covers_costs(
        price=3.3,
        expected_move_pct=0.2,
        spread_bps=18,
        config=config,
    )

    assert not check.accepted
    assert check.reason == "expected_move_below_costs"


def test_trade_is_allowed_when_expected_move_covers_costs():
    config = ExecutionCostConfig(
        slippage_bps_assumption=8,
        commission_per_lot_per_side=1.5,
        commission_round_trip_bps=0,
        min_expected_net_ticks=3,
        min_price_increment=0.001,
        money_value_per_price_step=1.0,
    )

    check = trade_covers_costs(
        price=3.3,
        expected_move_pct=1.2,
        spread_bps=10,
        config=config,
    )

    assert check.accepted
    assert check.reason == "costs_covered"


def test_round_trip_bps_commission_is_included():
    config = ExecutionCostConfig(
        slippage_bps_assumption=0,
        commission_per_lot_per_side=0,
        commission_round_trip_bps=8,
        min_expected_net_ticks=0,
        min_price_increment=0.001,
        money_value_per_price_step=1.0,
    )

    ticks = round_trip_cost_ticks(3.3, 0, config)

    assert round(ticks, 2) == 2.64
