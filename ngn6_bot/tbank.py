from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any

try:
    from tenacity import retry, stop_after_attempt, wait_fixed
except ModuleNotFoundError:
    def stop_after_attempt(attempts: int):
        return attempts

    def wait_fixed(seconds: float):
        return seconds

    def retry(stop: int, wait: float, reraise: bool = True):
        def decorator(func):
            def wrapper(*args, **kwargs):
                last_exc = None
                for _ in range(stop):
                    try:
                        return func(*args, **kwargs)
                    except Exception as exc:
                        last_exc = exc
                        time.sleep(wait)
                if reraise and last_exc is not None:
                    raise last_exc
                return None

            return wrapper

        return decorator

from ngn6_bot.models import Candle, OrderBookLevel, OrderBookSnapshot, Side, TradeTick


T_INVEST_API_TARGET = "invest-public-api.tbank.ru"
T_INVEST_SSL_VERIFY_ENV = "SSL_TBANK_VERIFY"
T_TECH_INSTALL_COMMAND = (
    "python -m pip install --upgrade t-tech-investments "
    "--index-url https://opensource.tbank.ru/api/v4/projects/238/packages/pypi/simple"
)


class TInvestGateway:
    """Thin adapter around T-Invest SDK.

    Imports are lazy so unit tests and dry-run logic can run without a broker connection.
    The current SDK package is installed as t-tech-investments and exposes t_tech.invest.
    """

    def __init__(self, token: str, config: dict[str, Any], logger: logging.Logger):
        self.token = token
        self.config = config
        self.logger = logger
        self._client = None
        self._stop_event = threading.Event()

    def __enter__(self):
        invest = _invest_sdk()

        target = _api_target(self.config)
        os.environ[T_INVEST_SSL_VERIFY_ENV] = "true"
        self._client_context = invest.Client(self.token, target=target)
        self._client = self._client_context.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()
        if getattr(self, "_client_context", None):
            return self._client_context.__exit__(exc_type, exc, tb)
        return None

    @property
    def client(self):
        if self._client is None:
            raise RuntimeError("TInvestGateway must be used as a context manager.")
        return self._client

    def stop(self) -> None:
        self._stop_event.set()

    def resolve_instrument(self) -> tuple[str, str]:
        figi = self.config.get("instrument", {}).get("figi")
        uid = self.config.get("instrument", {}).get("instrument_uid")
        if figi or uid:
            return figi, uid

        ticker = self.config["instrument"]["ticker"]
        class_code = self.config["instrument"].get("class_code", "SPBFUT")
        instruments = self.client.instruments.futures().instruments
        for instrument in instruments:
            if instrument.ticker == ticker and instrument.class_code == class_code:
                self.config["instrument"]["figi"] = instrument.figi
                self.config["instrument"]["instrument_uid"] = instrument.uid
                return instrument.figi, instrument.uid
        raise RuntimeError(f"Instrument {ticker}/{class_code} was not found in T-Invest futures.")

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), reraise=True)
    def get_order_book(self, figi: str, depth: int) -> OrderBookSnapshot:
        response = self.client.market_data.get_order_book(figi=figi, depth=depth)
        return OrderBookSnapshot(
            timestamp=_now_utc(),
            bids=[OrderBookLevel(_quotation_to_float(level.price), level.quantity) for level in response.bids],
            asks=[OrderBookLevel(_quotation_to_float(level.price), level.quantity) for level in response.asks],
        )

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), reraise=True)
    def get_recent_candles(self, figi: str, interval: Any, minutes_back: int) -> list[Candle]:
        to_dt = _now_utc()
        from_dt = to_dt - timedelta(minutes=minutes_back)
        response = self.client.market_data.get_candles(figi=figi, from_=from_dt, to=to_dt, interval=interval)
        timeframe = _interval_to_timeframe(interval)
        return [
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

    def post_market_order(self, account_id: str | None, side: Side, lots: int) -> str:
        if not account_id:
            raise RuntimeError("Live order requires account_id.")
        invest = _invest_sdk()

        figi, _ = self.resolve_instrument()
        direction = invest.OrderDirection.ORDER_DIRECTION_BUY
        if side == Side.SHORT:
            direction = invest.OrderDirection.ORDER_DIRECTION_SELL

        response = self.client.orders.post_order(
            figi=figi,
            quantity=lots,
            direction=direction,
            account_id=account_id,
            order_type=invest.OrderType.ORDER_TYPE_MARKET,
        )
        return response.order_id

    def start_stream(
        self,
        figi: str,
        on_orderbook: Callable[[OrderBookSnapshot], None],
        on_candle: Callable[[Candle], None],
        on_trade: Callable[[TradeTick], None],
    ) -> threading.Thread:
        thread = threading.Thread(
            target=self._stream_loop,
            args=(figi, on_orderbook, on_candle, on_trade),
            name="t-invest-stream",
            daemon=True,
        )
        thread.start()
        return thread

    def _stream_loop(self, figi, on_orderbook, on_candle, on_trade) -> None:
        while not self._stop_event.is_set():
            try:
                self._run_market_data_stream(figi, on_orderbook, on_candle, on_trade)
            except Exception as exc:
                if self._stop_event.is_set():
                    self.logger.info(
                        "market_data_stream_stopped",
                        extra={
                            "event": "market_data_stream_stopped",
                            "details": {"reason": str(exc)},
                        },
                    )
                    break
                self.logger.exception(
                    "market_data_stream_failed",
                    extra={"event": "market_data_stream_failed", "details": {"error": str(exc)}},
                )
                time.sleep(float(self.config.get("streaming", {}).get("reconnect_delay_seconds", 5)))

    def _run_market_data_stream(self, figi, on_orderbook, on_candle, on_trade) -> None:
        # The SDK stream API is generator-based. The request builder is kept isolated here because
        # SDK enum names occasionally change between invest-python and t-tech-investments releases.
        invest = _invest_sdk()

        def request_iterator():
            yield invest.MarketDataRequest(
                subscribe_order_book_request=invest.SubscribeOrderBookRequest(
                    subscription_action=invest.SubscriptionAction.SUBSCRIPTION_ACTION_SUBSCRIBE,
                    instruments=[
                        invest.OrderBookInstrument(
                            figi=figi,
                            depth=int(self.config["orderbook"]["depth"]),
                        )
                    ],
                )
            )
            yield invest.MarketDataRequest(
                subscribe_trades_request=invest.SubscribeTradesRequest(
                    subscription_action=invest.SubscriptionAction.SUBSCRIPTION_ACTION_SUBSCRIBE,
                    instruments=[invest.TradeInstrument(figi=figi)],
                )
            )
            yield invest.MarketDataRequest(
                subscribe_candles_request=invest.SubscribeCandlesRequest(
                    subscription_action=invest.SubscriptionAction.SUBSCRIPTION_ACTION_SUBSCRIBE,
                    instruments=[invest.CandleInstrument(figi=figi, interval=_candle_interval("1min"))],
                    waiting_close=True,
                )
            )
            yield invest.MarketDataRequest(
                subscribe_info_request=invest.SubscribeInfoRequest(
                    subscription_action=invest.SubscriptionAction.SUBSCRIPTION_ACTION_SUBSCRIBE,
                    instruments=[invest.InfoInstrument(figi=figi)],
                )
            )
            while not self._stop_event.is_set():
                time.sleep(1)

        for item in self.client.market_data_stream.market_data_stream(request_iterator()):
            if self._stop_event.is_set():
                break
            if getattr(item, "orderbook", None):
                book = item.orderbook
                on_orderbook(
                    OrderBookSnapshot(
                        timestamp=getattr(book, "time", None) or _now_utc(),
                        bids=[
                            OrderBookLevel(_quotation_to_float(level.price), level.quantity)
                            for level in book.bids
                        ],
                        asks=[
                            OrderBookLevel(_quotation_to_float(level.price), level.quantity)
                            for level in book.asks
                        ],
                    )
                )
            if getattr(item, "candle", None):
                candle = item.candle
                on_candle(
                    Candle(
                        timestamp=candle.time,
                        open=_quotation_to_float(candle.open),
                        high=_quotation_to_float(candle.high),
                        low=_quotation_to_float(candle.low),
                        close=_quotation_to_float(candle.close),
                        volume=float(candle.volume),
                        timeframe="1min",
                    )
                )
            if getattr(item, "trade", None):
                trade = item.trade
                on_trade(
                    TradeTick(
                        timestamp=trade.time,
                        price=_quotation_to_float(trade.price),
                        quantity=float(trade.quantity),
                        side=_trade_direction(trade),
                    )
                )


def _quotation_to_float(value) -> float:
    return float(value.units) + float(value.nano) / 1_000_000_000


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _trade_direction(trade) -> str:
    direction = str(getattr(trade, "direction", "")).lower()
    if "buy" in direction:
        return "buy"
    if "sell" in direction:
        return "sell"
    return "unknown"


def _candle_interval(timeframe: str):
    invest = _invest_sdk()

    mapping = {
        "1min": invest.SubscriptionInterval.SUBSCRIPTION_INTERVAL_ONE_MINUTE,
        "5min": invest.SubscriptionInterval.SUBSCRIPTION_INTERVAL_FIVE_MINUTES,
    }
    return mapping.get(timeframe, invest.SubscriptionInterval.SUBSCRIPTION_INTERVAL_ONE_MINUTE)


def candle_interval_for_polling(timeframe: str):
    invest = _invest_sdk()

    mapping = {
        "1min": invest.CandleInterval.CANDLE_INTERVAL_1_MIN,
        "5min": invest.CandleInterval.CANDLE_INTERVAL_5_MIN,
        "15min": invest.CandleInterval.CANDLE_INTERVAL_15_MIN,
    }
    return mapping[timeframe]


def _interval_to_timeframe(interval) -> str:
    name = str(interval).lower()
    if "15" in name:
        return "15min"
    if "5" in name:
        return "5min"
    return "1min"


def _invest_sdk():
    try:
        import t_tech.invest as invest

        return invest
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "T-Invest SDK is missing or outdated. Install the current SDK with: "
            f"{T_TECH_INSTALL_COMMAND}"
        ) from exc


def _api_target(config: dict[str, Any]) -> str:
    configured_target = (
        config.get("api", {}).get("target")
        or config.get("tbank", {}).get("target")
        or os.getenv("T_INVEST_API_TARGET")
    )
    target = str(configured_target or _sdk_default_target()).strip()
    _validate_api_target(target)
    return target


def _sdk_default_target() -> str:
    try:
        from t_tech.invest.constants import INVEST_GRPC_API

        return INVEST_GRPC_API
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "T-Invest SDK is missing or outdated. Install the current SDK with: "
            f"{T_TECH_INSTALL_COMMAND}"
        ) from exc


def _validate_api_target(target: str) -> None:
    target_lower = target.lower()
    if not target:
        raise RuntimeError(f"T-Invest API target is empty. Use {T_INVEST_API_TARGET}.")
    if "://" in target_lower:
        raise RuntimeError(
            "T-Invest API target must be a gRPC host without scheme. "
            f"Use {T_INVEST_API_TARGET} or {T_INVEST_API_TARGET}:443."
        )
    if "tinkoff.ru" in target_lower:
        raise RuntimeError(
            "T-Invest API target uses obsolete tinkoff.ru domain. "
            f"Use {T_INVEST_API_TARGET} or {T_INVEST_API_TARGET}:443."
        )
    host = target_lower.split("/", 1)[0].split(":", 1)[0]
    if not host.endswith("tbank.ru"):
        raise RuntimeError(
            "T-Invest API target must use the tbank.ru domain. "
            f"Use {T_INVEST_API_TARGET} or {T_INVEST_API_TARGET}:443."
        )
