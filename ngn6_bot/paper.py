from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from ngn6_bot.config import RuntimeConfig
from ngn6_bot.models import Position, Side
from ngn6_bot.runtime_metadata import add_commit_hash


@dataclass(frozen=True)
class PaperPortfolioConfig:
    initial_cash: float
    max_margin_notional: float
    state_file: Path
    events_file: Path
    lot_size: float
    notional_multiplier: float
    min_price_increment: float
    money_value_per_price_step: float
    initial_margin_on_buy: float
    initial_margin_on_sell: float
    commission_per_lot_per_side: float
    commission_round_trip_bps: float


@dataclass(frozen=True)
class PaperRiskSnapshot:
    daily_net_pnl: float = 0.0
    completed_trades_today: int = 0
    consecutive_losses: int = 0
    consecutive_hard_stops: int = 0
    last_exit_side: Side | None = None
    last_exit_time: datetime | None = None
    last_exit_reason: str | None = None
    last_exit_pnl_ticks: float = 0.0


class PaperPortfolio:
    def __init__(self, config: PaperPortfolioConfig):
        self.config = config
        self.config.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.config.events_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.config.state_file.exists():
            self._write_state(self._default_state())

    @classmethod
    def from_runtime_config(cls, config: RuntimeConfig) -> PaperPortfolio:
        min_increment = float(config.get("instrument", "min_price_increment"))
        step_value = float(config.get("instrument", "money_value_per_price_step"))
        default_notional_multiplier = step_value / min_increment if min_increment else 1.0
        return cls(
            PaperPortfolioConfig(
                initial_cash=float(
                    config.get(
                        "paper",
                        "initial_cash",
                        default=config.get("account", "deposit_value", default=300000),
                    )
                ),
                max_margin_notional=float(
                    config.get("paper", "max_margin_notional", default=1500000)
                ),
                state_file=Path(config.get("paper", "state_file", default="data/paper_state.json")),
                events_file=Path(config.get("paper", "events_file", default="data/paper_events.jsonl")),
                lot_size=float(config.get("instrument", "lot", default=1)),
                notional_multiplier=float(
                    config.get("paper", "notional_multiplier", default=default_notional_multiplier)
                ),
                min_price_increment=min_increment,
                money_value_per_price_step=step_value,
                initial_margin_on_buy=float(
                    config.get("instrument", "initial_margin_on_buy", default=0.0)
                ),
                initial_margin_on_sell=float(
                    config.get("instrument", "initial_margin_on_sell", default=0.0)
                ),
                commission_per_lot_per_side=float(
                    config.get("execution", "commission_per_lot_per_side", default=0.0)
                ),
                commission_round_trip_bps=float(
                    config.get("execution", "commission_round_trip_bps", default=0.0)
                ),
            )
        )

    def restore_position(self) -> Position:
        payload = self._load_state().get("position", {})
        side = Side(payload.get("side", Side.FLAT.value))
        if side == Side.FLAT or int(payload.get("lots", 0)) <= 0:
            return Position()
        return Position(
            side=side,
            lots=int(payload.get("lots", 0)),
            avg_price=float(payload.get("avg_price", 0.0)),
            opened_at=_parse_datetime(payload.get("opened_at")),
            stop_price=_optional_float(payload.get("stop_price")),
            trailing_stop=_optional_float(payload.get("trailing_stop")),
            partial_taken=bool(payload.get("partial_taken", False)),
            take_profit1=_optional_float(payload.get("take_profit1")),
            take_profit2=_optional_float(payload.get("take_profit2")),
        )

    def risk_snapshot(self, now: datetime, timezone_name: str) -> PaperRiskSnapshot:
        current = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
        local_date = current.astimezone(ZoneInfo(timezone_name)).date()
        completed: list[dict[str, Any]] = []
        active: dict[str, Any] | None = None

        for event in self._read_events():
            details = event.get("details") or {}
            event_name = event.get("event")
            timestamp = _parse_datetime(event.get("timestamp"))
            if event_name == "paper_open":
                active = {
                    "side": _optional_side(details.get("side")),
                    "lots": max(1, int(details.get("lots", 1) or 1)),
                    "net_pnl": -float(details.get("commission", 0.0) or 0.0),
                }
                continue
            if event_name != "paper_close" or active is None or timestamp is None:
                continue
            active["net_pnl"] += float(details.get("realized_pnl", 0.0) or 0.0)
            if int(details.get("remaining_lots", 0) or 0) > 0:
                continue
            net_pnl = float(active["net_pnl"])
            tick_value = self.config.money_value_per_price_step * int(active["lots"])
            completed.append(
                {
                    "timestamp": timestamp,
                    "side": active["side"],
                    "reason": str(details.get("reason", "unknown")),
                    "net_pnl": net_pnl,
                    "net_ticks": net_pnl / tick_value if tick_value > 0 else net_pnl,
                }
            )
            active = None

        today = [
            trade
            for trade in completed
            if trade["timestamp"].astimezone(ZoneInfo(timezone_name)).date() == local_date
        ]
        last = completed[-1] if completed else None
        return PaperRiskSnapshot(
            daily_net_pnl=sum(float(trade["net_pnl"]) for trade in today),
            completed_trades_today=len(today),
            consecutive_losses=_trailing_count(today, lambda trade: trade["net_pnl"] < 0),
            consecutive_hard_stops=_trailing_count(
                today,
                lambda trade: trade["net_pnl"] < 0
                and str(trade["reason"]).startswith("hard_stop_hit"),
            ),
            last_exit_side=last["side"] if last else None,
            last_exit_time=last["timestamp"] if last else None,
            last_exit_reason=last["reason"] if last else None,
            last_exit_pnl_ticks=float(last["net_ticks"]) if last else 0.0,
        )

    def open_position(
        self,
        *,
        side: Side,
        lots: int,
        price: float,
        stop_price: float | None,
        reason: str,
        take_profit1: float | None = None,
        take_profit2: float | None = None,
        feedback_context: dict[str, Any] | None = None,
        timestamp: datetime | None = None,
    ) -> tuple[bool, int, str]:
        timestamp = timestamp or datetime.now(timezone.utc)
        state = self._load_state()
        current_position = state.get("position", {})
        if current_position.get("side") != Side.FLAT.value and int(current_position.get("lots", 0)) > 0:
            self._append_event(
                "paper_open_rejected",
                timestamp,
                {"reason": "paper_position_already_open", "side": side.value, "lots": lots, "price": price},
            )
            return False, 0, "paper_position_already_open"

        accepted_lots = min(lots, self.max_open_lots(side, price))
        if accepted_lots <= 0:
            self._append_event(
                "paper_open_rejected",
                timestamp,
                {"reason": "paper_margin_limit", "side": side.value, "lots": lots, "price": price},
            )
            return False, 0, "paper_margin_limit"

        commission = self._side_commission(price, accepted_lots)
        state["cash"] = float(state.get("cash", self.config.initial_cash)) - commission
        state["realized_pnl"] = float(state.get("realized_pnl", 0.0)) - commission
        state["position"] = {
            "side": side.value,
            "lots": accepted_lots,
            "avg_price": price,
            "opened_at": timestamp.isoformat(),
            "stop_price": stop_price,
            "trailing_stop": stop_price,
            "partial_taken": False,
            "take_profit1": take_profit1,
            "take_profit2": take_profit2,
        }
        self._refresh_state_totals(
            state,
            Position(
                side=side,
                lots=accepted_lots,
                avg_price=price,
                stop_price=stop_price,
                trailing_stop=stop_price,
                take_profit1=take_profit1,
                take_profit2=take_profit2,
            ),
            price,
            timestamp,
        )
        self._write_state(state)
        event_reason = "paper_margin_lots_reduced" if accepted_lots < lots else reason
        self._append_event(
            "paper_open",
            timestamp,
            {
                "side": side.value,
                "requested_lots": lots,
                "lots": accepted_lots,
                "price": price,
                "stop_price": stop_price,
                "take_profit1": take_profit1,
                "take_profit2": take_profit2,
                "reason": event_reason,
                "commission": commission,
                "feedback_context": feedback_context or {},
            },
        )
        return True, accepted_lots, event_reason

    def close_position(
        self,
        *,
        position: Position,
        price: float,
        lots: int,
        reason: str,
        timestamp: datetime | None = None,
    ) -> tuple[bool, float, str]:
        timestamp = timestamp or datetime.now(timezone.utc)
        if not position.is_open:
            return False, 0.0, "no_open_position"

        closing_lots = min(lots, position.lots)
        pnl = self._pnl_money(position.side, position.avg_price, price, closing_lots)
        commission = self._side_commission(price, closing_lots)
        realized = pnl - commission

        state = self._load_state()
        state["cash"] = float(state.get("cash", self.config.initial_cash)) + realized
        state["realized_pnl"] = float(state.get("realized_pnl", 0.0)) + realized
        remaining_lots = position.lots - closing_lots
        if remaining_lots <= 0:
            remaining_position = Position()
        else:
            remaining_position = Position(
                side=position.side,
                lots=remaining_lots,
                avg_price=position.avg_price,
                opened_at=position.opened_at,
                stop_price=position.stop_price,
                trailing_stop=position.trailing_stop,
                partial_taken=True,
                take_profit1=position.take_profit1,
                take_profit2=position.take_profit2,
            )
        state["position"] = _position_payload(remaining_position)
        self._refresh_state_totals(state, remaining_position, price, timestamp)
        self._write_state(state)
        self._append_event(
            "paper_close",
            timestamp,
            {
                "side": position.side.value,
                "lots": closing_lots,
                "price": price,
                "avg_price": position.avg_price,
                "reason": reason,
                "gross_pnl": pnl,
                "commission": commission,
                "realized_pnl": realized,
                "remaining_lots": remaining_lots,
            },
        )
        return True, realized, reason

    def mark_to_market(self, position: Position, price: float, timestamp: datetime | None = None) -> dict[str, Any]:
        timestamp = timestamp or datetime.now(timezone.utc)
        state = self._load_state()
        state["position"] = _position_payload(position)
        self._refresh_state_totals(state, position, price, timestamp)
        self._write_state(state)
        return state

    def max_open_lots(self, side: Side, price: float) -> int:
        margin_per_lot = self._margin_requirement(side, price, 1)
        if margin_per_lot <= 0:
            return 0
        return max(0, math.floor(self.config.max_margin_notional / margin_per_lot))

    def _refresh_state_totals(
        self,
        state: dict[str, Any],
        position: Position,
        mark_price: float,
        timestamp: datetime,
    ) -> None:
        cash = float(state.get("cash", self.config.initial_cash))
        unrealized = (
            self._pnl_money(position.side, position.avg_price, mark_price, position.lots)
            if position.is_open
            else 0.0
        )
        margin_used = (
            self._margin_requirement(position.side, mark_price, position.lots)
            if position.is_open
            else 0.0
        )
        contract_value = self._contract_value(mark_price, 1) if mark_price is not None else 0.0
        state.update(
            {
                "mode": "paper",
                "initial_cash": self.config.initial_cash,
                "cash": cash,
                "max_margin_notional": self.config.max_margin_notional,
                "margin_used": margin_used,
                "margin_available": max(0.0, self.config.max_margin_notional - margin_used),
                "unrealized_pnl": unrealized,
                "equity": cash + unrealized,
                "updated_at": timestamp.isoformat(),
                "mark_price": mark_price,
                "contract_value": contract_value,
                "min_price_increment": self.config.min_price_increment,
                "money_value_per_price_step": self.config.money_value_per_price_step,
                "initial_margin_on_buy": self.config.initial_margin_on_buy,
                "initial_margin_on_sell": self.config.initial_margin_on_sell,
            }
        )

    def _default_state(self) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        return {
            "mode": "paper",
            "initial_cash": self.config.initial_cash,
            "cash": self.config.initial_cash,
            "equity": self.config.initial_cash,
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "max_margin_notional": self.config.max_margin_notional,
            "margin_used": 0.0,
            "margin_available": self.config.max_margin_notional,
            "mark_price": None,
            "contract_value": 0.0,
            "min_price_increment": self.config.min_price_increment,
            "money_value_per_price_step": self.config.money_value_per_price_step,
            "initial_margin_on_buy": self.config.initial_margin_on_buy,
            "initial_margin_on_sell": self.config.initial_margin_on_sell,
            "position": _position_payload(Position()),
            "updated_at": now,
        }

    def _load_state(self) -> dict[str, Any]:
        try:
            state = json.loads(self.config.state_file.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return self._default_state()
        state.setdefault("cash", self.config.initial_cash)
        state.setdefault("realized_pnl", 0.0)
        state.setdefault("position", _position_payload(Position()))
        state["initial_cash"] = self.config.initial_cash
        state["max_margin_notional"] = self.config.max_margin_notional
        return state

    def _write_state(self, state: dict[str, Any]) -> None:
        tmp_path = self.config.state_file.with_suffix(self.config.state_file.suffix + ".tmp")
        add_commit_hash(state)
        tmp_path.write_text(json.dumps(state, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
        tmp_path.replace(self.config.state_file)

    def _append_event(self, event: str, timestamp: datetime, details: dict[str, Any]) -> None:
        payload = {"timestamp": timestamp.isoformat(), "event": event, "details": details}
        add_commit_hash(payload)
        with self.config.events_file.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, ensure_ascii=False, default=_json_default) + "\n")

    def _read_events(self) -> list[dict[str, Any]]:
        try:
            lines = self.config.events_file.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return []
        events = []
        for line in lines:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                events.append(payload)
        return events

    def _pnl_money(self, side: Side, entry_price: float, exit_price: float, lots: int) -> float:
        if lots <= 0 or self.config.min_price_increment <= 0:
            return 0.0
        move = exit_price - entry_price
        if side == Side.SHORT:
            move = -move
        steps = move / self.config.min_price_increment
        return steps * self.config.money_value_per_price_step * lots

    def _exposure(self, price: float, lots: int) -> float:
        return self._contract_value(price, lots)

    def _contract_value(self, price: float, lots: int) -> float:
        return abs(price * lots * self.config.lot_size * self.config.notional_multiplier)

    def _margin_requirement(self, side: Side, price: float, lots: int) -> float:
        if lots <= 0:
            return 0.0
        margin_per_lot = (
            self.config.initial_margin_on_buy
            if side == Side.LONG
            else self.config.initial_margin_on_sell
        )
        if margin_per_lot > 0:
            return margin_per_lot * lots
        return self._exposure(price, lots)

    def _side_commission(self, price: float, lots: int) -> float:
        fixed = lots * self.config.commission_per_lot_per_side
        bps = max(self.config.commission_round_trip_bps, 0.0) / 2
        percent = self._exposure(price, lots) * bps / 10_000
        return fixed + percent


def _position_payload(position: Position) -> dict[str, Any]:
    return {
        "side": position.side.value,
        "lots": position.lots,
        "avg_price": position.avg_price,
        "opened_at": position.opened_at.isoformat() if position.opened_at else None,
        "stop_price": position.stop_price,
        "trailing_stop": position.trailing_stop,
        "partial_taken": position.partial_taken,
        "take_profit1": position.take_profit1,
        "take_profit2": position.take_profit2,
    }


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_side(value: Any) -> Side | None:
    try:
        side = Side(str(value))
    except ValueError:
        return None
    return side if side != Side.FLAT else None


def _trailing_count(trades: list[dict[str, Any]], predicate) -> int:
    count = 0
    for trade in reversed(trades):
        if not predicate(trade):
            break
        count += 1
    return count


def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    return str(value)
