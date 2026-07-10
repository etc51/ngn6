from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from ngn6_bot.models import Position, Side, Signal
from ngn6_bot.paper import PaperPortfolio


@dataclass(frozen=True)
class ExecutionResult:
    accepted: bool
    side: Side
    lots: int
    price: float
    reason: str
    order_id: str | None = None


class BrokerExecutor:
    def __init__(
        self,
        gateway,
        account_id: str | None,
        dry_run: bool,
        logger: logging.Logger,
        paper_portfolio: PaperPortfolio | None = None,
    ):
        self.gateway = gateway
        self.account_id = account_id
        self.dry_run = dry_run
        self.logger = logger
        self.paper_portfolio = paper_portfolio

    def open_position(self, signal: Signal, lots: int, stop_price: float | None) -> ExecutionResult:
        if lots <= 0:
            return ExecutionResult(False, signal.side, lots, signal.price, "lots_is_zero")

        if self.dry_run:
            accepted = True
            accepted_lots = lots
            result_reason = "dry_run"
            if self.paper_portfolio is not None:
                accepted, accepted_lots, result_reason = self.paper_portfolio.open_position(
                    side=signal.side,
                    lots=lots,
                    price=signal.price,
                    stop_price=stop_price,
                    take_profit1=signal.take_profit1,
                    take_profit2=signal.take_profit2,
                    reason=signal.reason,
                    timestamp=datetime.now(timezone.utc),
                )
            self.logger.info(
                "paper_open_position" if accepted else "paper_open_rejected",
                extra={
                    "event": "paper_open_position" if accepted else "paper_open_rejected",
                    "details": {
                        "side": signal.side.value,
                        "requested_lots": lots,
                        "lots": accepted_lots,
                        "price": signal.price,
                        "stop_price": stop_price,
                        "take_profit1": signal.take_profit1,
                        "take_profit2": signal.take_profit2,
                        "reason": result_reason,
                    },
                },
            )
            return ExecutionResult(accepted, signal.side, accepted_lots, signal.price, result_reason)

        order_id = self.gateway.post_market_order(self.account_id, signal.side, lots)
        return ExecutionResult(True, signal.side, lots, signal.price, "sent_to_broker", order_id)

    def close_position(self, position: Position, price: float, reason: str) -> ExecutionResult:
        if not position.is_open:
            return ExecutionResult(False, Side.FLAT, 0, price, "no_open_position")

        closing_side = Side.SHORT if position.side == Side.LONG else Side.LONG
        if self.dry_run:
            accepted = True
            realized_pnl = None
            result_reason = reason
            if self.paper_portfolio is not None:
                accepted, realized_pnl, result_reason = self.paper_portfolio.close_position(
                    position=position,
                    price=price,
                    lots=position.lots,
                    reason=reason,
                    timestamp=datetime.now(timezone.utc),
                )
            self.logger.info(
                "paper_close_position" if accepted else "paper_close_rejected",
                extra={
                    "event": "paper_close_position" if accepted else "paper_close_rejected",
                    "details": {
                        "position_side": position.side.value,
                        "closing_side": closing_side.value,
                        "lots": position.lots,
                        "price": price,
                        "reason": result_reason,
                        "realized_pnl": realized_pnl,
                    },
                },
            )
            return ExecutionResult(accepted, closing_side, position.lots if accepted else 0, price, result_reason)

        order_id = self.gateway.post_market_order(self.account_id, closing_side, position.lots)
        return ExecutionResult(True, closing_side, position.lots, price, reason, order_id)

    @staticmethod
    def apply_open(
        position: Position,
        result: ExecutionResult,
        stop_price: float | None,
        signal: Signal | None = None,
    ) -> None:
        if not result.accepted:
            return
        position.side = result.side
        position.lots = result.lots
        position.avg_price = result.price
        position.opened_at = datetime.now(timezone.utc)
        position.stop_price = stop_price
        position.trailing_stop = stop_price
        position.partial_taken = False
        position.take_profit1 = signal.take_profit1 if signal is not None else None
        position.take_profit2 = signal.take_profit2 if signal is not None else None

    @staticmethod
    def apply_close(position: Position, result: ExecutionResult) -> None:
        if not result.accepted:
            return
        position.side = Side.FLAT
        position.lots = 0
        position.avg_price = 0.0
        position.opened_at = None
        position.stop_price = None
        position.trailing_stop = None
        position.partial_taken = False
        position.take_profit1 = None
        position.take_profit2 = None

    def take_partial(self, position: Position, price: float, fraction: float) -> ExecutionResult:
        if not position.is_open:
            return ExecutionResult(False, Side.FLAT, 0, price, "no_open_position")
        lots = max(1, int(position.lots * fraction))
        lots = min(lots, position.lots)
        closing_side = Side.SHORT if position.side == Side.LONG else Side.LONG
        if self.dry_run:
            accepted = True
            realized_pnl = None
            result_reason = "partial_take_profit"
            if self.paper_portfolio is not None:
                accepted, realized_pnl, result_reason = self.paper_portfolio.close_position(
                    position=position,
                    price=price,
                    lots=lots,
                    reason="partial_take_profit",
                    timestamp=datetime.now(timezone.utc),
                )
            self.logger.info(
                "paper_take_partial" if accepted else "paper_take_partial_rejected",
                extra={
                    "event": "paper_take_partial" if accepted else "paper_take_partial_rejected",
                    "details": {"lots": lots, "price": price, "fraction": fraction},
                },
            )
            return ExecutionResult(accepted, closing_side, lots if accepted else 0, price, result_reason)
        order_id = self.gateway.post_market_order(self.account_id, closing_side, lots)
        return ExecutionResult(True, closing_side, lots, price, "partial_take_profit", order_id)

    @staticmethod
    def apply_partial(position: Position, result: ExecutionResult) -> None:
        if not result.accepted:
            return
        position.lots -= result.lots
        position.partial_taken = True
        if position.lots <= 0:
            BrokerExecutor.apply_close(position, result)
