from __future__ import annotations

import logging
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from ngn6_bot.config import RuntimeConfig
from ngn6_bot.costs import ExecutionCostConfig, trade_covers_costs
from ngn6_bot.execution import BrokerExecutor
from ngn6_bot.indicators import add_indicators, candles_to_frame
from ngn6_bot.learning.daily_oracle import DailyOracleScheduler
from ngn6_bot.learning.feedback_model import FEATURE_KEYS, FeedbackModel, build_feature_snapshot
from ngn6_bot.learning.paper_feedback import sync_paper_trade_feedback
from ngn6_bot.models import MarketState, Side
from ngn6_bot.orderbook import analyze_order_book, spread_is_acceptable
from ngn6_bot.paper import PaperPortfolio
from ngn6_bot.recorder import StrategyRecorder
from ngn6_bot.review import ReviewScheduler
from ngn6_bot.risk import (
    RiskConfig,
    calculate_position_lots,
    drawdown_limit_hit,
    liquidity_covers_lots,
    move_stop_to_breakeven,
    must_flatten_before_clearing,
    should_take_partial,
    stop_with_buffer,
    trading_session_block_reason,
    trailing_stop_hit,
    update_trailing_stop,
)
from ngn6_bot.signals import SignalConfig, generate_signal, microstructure_allows_entry
from ngn6_bot.tbank import TInvestGateway, candle_interval_for_polling
from ngn6_bot.tradeflow import analyze_trade_flow


class TradingBot:
    def __init__(self, config: RuntimeConfig, logger: logging.Logger, runtime_services: bool = True):
        self.config = config
        self.logger = logger
        self.state = MarketState()
        self.paper_portfolio = (
            PaperPortfolio.from_runtime_config(config)
            if runtime_services and bool(config.get("paper", "enabled", default=config.dry_run))
            else None
        )
        if self.paper_portfolio is not None:
            self.state.position = self.paper_portfolio.restore_position()
        self.recorder = (
            StrategyRecorder.from_runtime_config(config)
            if runtime_services
            else StrategyRecorder.disabled()
        )
        self.review_scheduler = ReviewScheduler(config, logger) if runtime_services else None
        self.daily_oracle_scheduler = (
            DailyOracleScheduler(config, logger, self._reload_feedback_model)
            if runtime_services
            else None
        )
        self.feedback_model = FeedbackModel.from_runtime_config(config)
        self._last_full_exit: tuple[Side, datetime, str, float] | None = None
        self._pending_exit_signal: dict[str, object] | None = None
        self._restore_last_full_exit(datetime.now(timezone.utc))

    def run_forever(self) -> None:
        token = self.config.token
        with TInvestGateway(token, self.config.raw, self.logger) as gateway:
            figi, uid = gateway.resolve_instrument()
            self.logger.info(
                "instrument_resolved",
                extra={"event": "instrument_resolved", "details": {"figi": figi, "uid": uid}},
            )
            self._bootstrap_history(gateway, figi)

            if self.config.get("streaming", "enabled", default=True):
                gateway.start_stream(
                    figi,
                    on_orderbook=self.state.update_order_book,
                    on_candle=self.state.update_candle,
                    on_trade=self.state.update_trade,
                )

            executor = BrokerExecutor(
                gateway=gateway,
                account_id=self.config.account_id,
                dry_run=self.config.dry_run,
                logger=self.logger,
                paper_portfolio=self.paper_portfolio,
            )
            self._main_loop(gateway, executor, figi)

    def run_once_dry(self) -> None:
        token = self.config.token
        with TInvestGateway(token, self.config.raw, self.logger) as gateway:
            figi, uid = gateway.resolve_instrument()
            self.logger.info(
                "instrument_resolved",
                extra={"event": "instrument_resolved", "details": {"figi": figi, "uid": uid}},
            )
            self._bootstrap_history(gateway, figi)
            executor = BrokerExecutor(
                gateway=gateway,
                account_id=self.config.account_id,
                dry_run=True,
                logger=self.logger,
                paper_portfolio=self.paper_portfolio,
            )
            self._evaluate_once(executor, datetime.now(timezone.utc))
            self.logger.info(
                "strategy_check_ok",
                extra={
                    "event": "strategy_check_ok",
                    "details": {
                        "candles_1m": len(self.state.candles_1m),
                        "candles_5m": len(self.state.candles_5m),
                        "candles_15m": len(self.state.candles_15m),
                        "has_orderbook": self.state.order_book is not None,
                    },
                },
            )

    def check_stream(self, seconds: float) -> None:
        token = self.config.token
        with TInvestGateway(token, self.config.raw, self.logger) as gateway:
            figi, uid = gateway.resolve_instrument()
            self.logger.info(
                "instrument_resolved",
                extra={"event": "instrument_resolved", "details": {"figi": figi, "uid": uid}},
            )
            gateway.start_stream(
                figi,
                on_orderbook=self.state.update_order_book,
                on_candle=self.state.update_candle,
                on_trade=self.state.update_trade,
            )
            deadline = time.time() + seconds
            while time.time() < deadline:
                if self.state.order_book is not None or self.state.candles_1m or self.state.trades:
                    break
                time.sleep(0.25)
            gateway.stop()
            if self.state.order_book is None and not self.state.candles_1m and not self.state.trades:
                raise RuntimeError(f"No stream events received during {seconds} seconds.")
            self.logger.info(
                "stream_check_ok",
                extra={
                    "event": "stream_check_ok",
                    "details": {
                        "seconds": seconds,
                        "has_orderbook": self.state.order_book is not None,
                        "candles_1m": len(self.state.candles_1m),
                        "trades": len(self.state.trades),
                    },
                },
            )

    def _bootstrap_history(self, gateway: TInvestGateway, figi: str) -> None:
        for timeframe, minutes in [("1min", 300), ("5min", 1500), ("15min", 4500)]:
            candles = gateway.get_recent_candles(
                figi,
                candle_interval_for_polling(timeframe),
                minutes_back=minutes,
            )
            for candle in candles:
                if not self._use_candle(candle.timestamp):
                    continue
                self.state.update_candle(candle)
        self.state.update_order_book(gateway.get_order_book(figi, int(self.config.get("orderbook", "depth"))))

    def _use_candle(self, timestamp: datetime) -> bool:
        if not bool(self.config.get("market_data", "today_only", default=True)):
            return True
        tz = ZoneInfo(self.config.timezone)
        now_local = datetime.now(timezone.utc).astimezone(tz)
        candle_time = timestamp
        if candle_time.tzinfo is None:
            candle_time = candle_time.replace(tzinfo=timezone.utc)
        return candle_time.astimezone(tz).date() == now_local.date()

    def _main_loop(self, gateway: TInvestGateway, executor: BrokerExecutor, figi: str) -> None:
        interval = float(self.config.get("polling", "interval_seconds", default=5))
        while True:
            now = datetime.now(timezone.utc)
            self._poll_if_stale(gateway, figi, now)
            self._evaluate_once(executor, now)
            if self.review_scheduler is not None:
                self.review_scheduler.maybe_run(now, self.state)
            if self.daily_oracle_scheduler is not None:
                self.daily_oracle_scheduler.maybe_run(now)
            time.sleep(interval)

    def _poll_if_stale(self, gateway: TInvestGateway, figi: str, now: datetime) -> None:
        if not self.config.get("polling", "enabled", default=True):
            return
        stale_after = float(self.config.get("polling", "stale_after_seconds", default=15))
        last = self.state.last_stream_update
        if last is not None and (now - last).total_seconds() <= stale_after:
            return

        self.logger.warning(
            "stream_stale_polling_market_data",
            extra={"event": "stream_stale_polling_market_data", "details": {"stale_after": stale_after}},
        )
        try:
            self.state.update_order_book(
                gateway.get_order_book(figi, int(self.config.get("orderbook", "depth")))
            )
            for timeframe, minutes in [("1min", 30), ("5min", 120), ("15min", 240)]:
                candles = gateway.get_recent_candles(
                    figi,
                    candle_interval_for_polling(timeframe),
                    minutes_back=minutes,
                )
                for candle in candles:
                    if not self._use_candle(candle.timestamp):
                        continue
                    self.state.update_candle(candle)
        except Exception as exc:
            self.logger.exception(
                "polling_failed",
                extra={"event": "polling_failed", "details": {"error": str(exc)}},
            )

    def _evaluate_once(self, executor: BrokerExecutor, now: datetime) -> None:
        orderbook_features = self._orderbook_features(now)
        trade_flow = analyze_trade_flow(
            list(self.state.trades),
            now,
            int(self.config.get("signals", "trade_flow_lookback_seconds", default=30)),
        )
        last_price = orderbook_features.mid_price or self._last_close()
        if last_price is None:
            self._record_decision(now, "skip", "no_price", details={})
            return
        self._record_market(now, last_price, orderbook_features, trade_flow)
        paper_state = None
        if self.paper_portfolio is not None:
            paper_state = self.paper_portfolio.mark_to_market(self.state.position, last_price, now)

        risk_config = self._risk_config()
        if self.state.position.is_open:
            execution_df = self._indicator_frame("1min", now)
            confirmation_df = self._indicator_frame("5min", now)
            context_df = self._indicator_frame("15min", now)
            flip_signal = None
            ml_exit_reason = None
            ml_exit_prediction = None
            if self._ml_control_enabled():
                ml_exit_reason, ml_exit_prediction = self.feedback_model.exit_reason_from_prediction(
                    position_side=self.state.position.side,
                    execution_df=execution_df,
                    confirmation_df=confirmation_df,
                    context_df=context_df,
                    orderbook=orderbook_features,
                    trade_flow=trade_flow,
                    now=now,
                    allow_candidate=self._paper_candidate_execution_enabled(),
                )
            else:
                flip_signal = generate_signal(
                    execution_df=execution_df,
                    confirmation_df=confirmation_df,
                    context_df=context_df,
                    orderbook=orderbook_features,
                    trade_flow=trade_flow,
                    config=self._signal_config(),
                    now=now,
                )
            self._manage_open_position(
                executor,
                last_price,
                now,
                risk_config,
                execution_df=execution_df,
                confirmation_df=confirmation_df,
                context_df=context_df,
                flip_signal=flip_signal,
                ml_exit_reason=ml_exit_reason,
                ml_exit_prediction=ml_exit_prediction,
            )
            return

        session_reason = self._session_block_reason(now)
        if session_reason:
            self._record_decision(now, "skip", session_reason, details={"price": last_price})
            return

        paper_risk_reason, paper_risk_details = self._paper_risk_entry_block(now, risk_config)
        if paper_risk_reason:
            self._record_decision(now, "skip", paper_risk_reason, details=paper_risk_details)
            return

        if paper_state and drawdown_limit_hit(
            float(paper_state.get("equity", 0.0)),
            float(paper_state.get("initial_cash", 0.0)),
            risk_config,
        ):
            self._record_decision(
                now,
                "skip",
                "max_drawdown_reached",
                details={
                    "equity": paper_state.get("equity"),
                    "initial_cash": paper_state.get("initial_cash"),
                    "max_drawdown_pct": risk_config.max_drawdown_pct,
                },
            )
            return

        if must_flatten_before_clearing(now, risk_config):
            self._record_decision(now, "skip", "pre_clearing_window", details={"price": last_price})
            return
        if orderbook_features.spread_bps is not None and not spread_is_acceptable(
            orderbook_features,
            float(self.config.get("orderbook", "max_spread_bps")),
        ):
            self._log_skip("spread_too_wide", {"spread_bps": orderbook_features.spread_bps}, now)
            return
        stale_reason, stale_details = self._entry_market_data_block(now, orderbook_features)
        if stale_reason:
            self._log_skip(stale_reason, stale_details, now)
            return

        execution_df = self._indicator_frame("1min", now)
        confirmation_df = self._indicator_frame("5min", now)
        context_df = self._indicator_frame("15min", now)
        ml_control = self._ml_control_enabled()
        if ml_control:
            signal = self.feedback_model.signal_from_prediction(
                execution_df=execution_df,
                confirmation_df=confirmation_df,
                context_df=context_df,
                orderbook=orderbook_features,
                trade_flow=trade_flow,
                now=now,
                price=last_price,
                allow_candidate=self._paper_candidate_execution_enabled(),
            )
        else:
            signal = generate_signal(
                execution_df=execution_df,
                confirmation_df=confirmation_df,
                context_df=context_df,
                orderbook=orderbook_features,
                trade_flow=trade_flow,
                config=self._signal_config(),
                now=now,
            )
            signal = self._apply_feedback(
                signal,
                execution_df=execution_df,
                confirmation_df=confirmation_df,
                context_df=context_df,
                orderbook_features=orderbook_features,
                trade_flow=trade_flow,
                now=now,
            )
        if signal.side == Side.FLAT:
            self._log_skip(
                signal.reason,
                {"confidence": signal.confidence, "metadata": signal.metadata},
                now,
            )
            return
        blocked_reason = self._entry_signal_block_reason(signal)
        if blocked_reason:
            self._log_skip(
                blocked_reason,
                {
                    "side": signal.side.value,
                    "confidence": signal.confidence,
                    "signal_reason": signal.reason,
                    "metadata": signal.metadata,
                },
                now,
            )
            return
        signal, allowed, confirm_reason = self._apply_5m_entry_confirmation(
            signal,
            confirmation_df,
        )
        if not allowed:
            self._log_skip(
                confirm_reason,
                {
                    "side": signal.side.value,
                    "confidence": signal.confidence,
                    "signal_reason": signal.reason,
                    "metadata": signal.metadata,
                },
                now,
            )
            return
        signal, allowed, exhaustion_reason = self._apply_short_exhaustion_guard(
            signal,
            context_df,
            orderbook_features,
        )
        if not allowed:
            self._log_skip(
                exhaustion_reason,
                {
                    "side": signal.side.value,
                    "confidence": signal.confidence,
                    "signal_reason": signal.reason,
                    "metadata": signal.metadata,
                },
                now,
            )
            return
        if ml_control:
            micro_ok, micro_reason = microstructure_allows_entry(
                signal.side,
                orderbook_features,
                trade_flow,
                self._signal_config(),
            )
            if not micro_ok:
                self._log_skip(
                    micro_reason,
                    {
                        "side": signal.side.value,
                        "confidence": signal.confidence,
                        "signal_reason": signal.reason,
                        "metadata": signal.metadata,
                    },
                    now,
                )
                return
        cooldown_reason, cooldown_details = self._reentry_cooldown_block(signal.side, now)
        if cooldown_reason:
            self._log_skip(
                cooldown_reason,
                {"side": signal.side.value, "signal_reason": signal.reason, **cooldown_details},
                now,
            )
            return

        stop_price = stop_with_buffer(signal, risk_config)
        signal = replace(signal, stop_price=stop_price)
        cost_check = trade_covers_costs(
            price=signal.price,
            expected_move_pct=self._expected_move_pct(signal, risk_config),
            spread_bps=orderbook_features.spread_bps,
            config=self._execution_cost_config(),
        )
        if not cost_check.accepted:
            self._log_skip(
                cost_check.reason,
                {
                    "expected_move_ticks": cost_check.expected_move_ticks,
                    "round_trip_cost_ticks": cost_check.round_trip_cost_ticks,
                    "min_required_ticks": cost_check.min_required_ticks,
                    "spread_bps": orderbook_features.spread_bps,
                    "signal_reason": signal.reason,
                },
                now,
            )
            return
        lots = calculate_position_lots(signal, risk_config)
        if signal.metadata.get("exploration"):
            lots = min(
                lots,
                max(1, int(self.config.get("learning", "exploration_max_lots", default=1))),
            )
        if not liquidity_covers_lots(
            signal.side,
            lots,
            orderbook_features.bid_depth,
            orderbook_features.ask_depth,
            float(self.config.get("orderbook", "min_liquidity_cover", default=0.0)),
        ):
            self._log_skip(
                "liquidity_cover_below_min",
                {
                    "side": signal.side.value,
                    "lots": lots,
                    "bid_depth": orderbook_features.bid_depth,
                    "ask_depth": orderbook_features.ask_depth,
                    "min_liquidity_cover": float(
                        self.config.get("orderbook", "min_liquidity_cover", default=0.0)
                    ),
                    "signal_reason": signal.reason,
                },
                now,
            )
            return
        risk_block_reason, risk_block_details = self._final_entry_risk_check(
            signal=signal,
            lots=lots,
            now=now,
            risk_config=risk_config,
            orderbook_features=orderbook_features,
            cost_check=cost_check,
        )
        if risk_block_reason:
            self._log_skip(risk_block_reason, risk_block_details, now)
            return
        result = executor.open_position(signal, lots, stop_price)
        self._record_decision(
            now,
            "open_accepted" if result.accepted else "open_rejected",
            result.reason,
            signal=signal,
            details={"requested_lots": lots, "lots": result.lots, "stop_price": stop_price},
        )
        executor.apply_open(self.state.position, result, stop_price, signal=signal)

    def _apply_5m_entry_confirmation(self, signal, confirmation_df):
        if not bool(self.config.get("signals", "confirm_5m_for_entry", default=True)):
            return signal, True, "5m_confirmation_disabled"
        state, details = self._confirmation_5m_state(confirmation_df)
        metadata = {**signal.metadata, "confirmation_5m": {"state": state, **details}}
        signal = replace(signal, metadata=metadata)
        if signal.side == Side.LONG and state == "confirm_short":
            return signal, False, "blocked_by_5m_confirmation:short"
        if signal.side == Side.SHORT and state == "confirm_long":
            return signal, False, "blocked_by_5m_confirmation:long"
        return signal, True, f"5m_confirmation:{state}"

    @staticmethod
    def _confirmation_5m_state(confirmation_df):
        if confirmation_df is None or confirmation_df.empty or len(confirmation_df) < 3:
            return "neutral", {"reason": "not_enough_5m_candles", "candles": 0 if confirmation_df is None else len(confirmation_df)}
        row = confirmation_df.iloc[-1]
        required = ("close", "ema_fast", "ema_slow", "macd_hist", "rsi")
        if any(column not in row or row[column] != row[column] for column in required):
            return "neutral", {"reason": "missing_5m_indicators", "candles": len(confirmation_df)}
        close = float(row["close"])
        ema_fast = float(row["ema_fast"])
        ema_slow = float(row["ema_slow"])
        macd_hist = float(row["macd_hist"])
        rsi = float(row["rsi"])
        long_score = sum((ema_fast > ema_slow, macd_hist > 0, close >= ema_fast, rsi >= 50))
        short_score = sum((ema_fast < ema_slow, macd_hist < 0, close <= ema_fast, rsi <= 50))
        details = {
            "candles": len(confirmation_df),
            "close": close,
            "ema_fast": ema_fast,
            "ema_slow": ema_slow,
            "macd_hist": macd_hist,
            "rsi": rsi,
            "long_score": long_score,
            "short_score": short_score,
        }
        if long_score >= 3 and long_score > short_score:
            return "confirm_long", details
        if short_score >= 3 and short_score > long_score:
            return "confirm_short", details
        return "neutral", details

    def _apply_short_exhaustion_guard(self, signal, context_df, orderbook_features):
        if not bool(self.config.get("signals", "block_short_exhaustion_entries", default=True)):
            return signal, True, "short_exhaustion_guard_disabled"
        if signal.side != Side.SHORT:
            return signal, True, "short_exhaustion_guard_not_short"

        state, details = self._short_exhaustion_state(context_df, signal.price, orderbook_features)
        metadata = {**signal.metadata, "short_exhaustion": details}
        signal = replace(signal, metadata=metadata)
        if state == "microstructure_bid_support":
            return signal, False, "blocked_by_short_exhaustion:bid_support"
        if state == "capitulation_low":
            return signal, False, "blocked_by_short_exhaustion:capitulation_low"
        return signal, True, f"short_exhaustion_guard:{state}"

    def _short_exhaustion_state(self, context_df, price: float, orderbook_features):
        details = self._short_exhaustion_candle_details(context_df, price)
        if details.get("available"):
            near_low_pct = float(details["near_low_pct"])
            if self._orderbook_bid_support_near_low(orderbook_features, near_low_pct):
                details.update(
                    {
                        "state": "microstructure_bid_support",
                        "bid_ask_imbalance": float(orderbook_features.bid_ask_imbalance),
                        "depth_pressure": float(orderbook_features.depth_pressure),
                    }
                )
                return "microstructure_bid_support", details
            if self._candle_capitulation_low(details):
                details["state"] = "capitulation_low"
                return "capitulation_low", details
        details["state"] = "clear"
        return "clear", details

    def _short_exhaustion_candle_details(self, context_df, price: float):
        if context_df is None or context_df.empty or len(context_df) < 7:
            return {"available": False, "reason": "not_enough_context_candles", "candles": 0 if context_df is None else len(context_df)}
        required = ("close", "low", "rsi", "macd_hist")
        if any(column not in context_df.columns for column in required):
            return {"available": False, "reason": "missing_context_columns", "candles": len(context_df)}

        lookback = int(self.config.get("signals", "short_exhaustion_lookback_bars", default=20))
        impulse_bars = int(self.config.get("signals", "short_exhaustion_impulse_bars", default=6))
        recent = context_df.tail(max(impulse_bars + 1, 2))
        prior = context_df.tail(max(lookback, 1))
        row = context_df.iloc[-1]
        rolling_low = float(prior["low"].min())
        reference_price = float(price or row["close"])
        impulse_start = float(recent.iloc[0]["close"])
        near_low_pct = (reference_price - rolling_low) / reference_price * 100
        impulse_down_pct = (impulse_start - reference_price) / reference_price * 100
        rsi = float(row["rsi"])
        macd_hist = float(row["macd_hist"])
        return {
            "available": True,
            "candles": len(context_df),
            "reference_price": reference_price,
            "rolling_low": rolling_low,
            "near_low_pct": near_low_pct,
            "impulse_down_pct": impulse_down_pct,
            "rsi": rsi,
            "macd_hist": macd_hist,
        }

    def _orderbook_bid_support_near_low(self, orderbook_features, near_low_pct: float) -> bool:
        if not bool(self.config.get("signals", "short_exhaustion_use_orderbook", default=True)):
            return False
        max_near_low_pct = float(self.config.get("signals", "short_exhaustion_orderbook_near_low_pct", default=0.25))
        min_imbalance = float(self.config.get("signals", "short_exhaustion_min_bid_ask_imbalance", default=0.56))
        min_depth_pressure = float(self.config.get("signals", "short_exhaustion_min_depth_pressure", default=0.12))
        return (
            near_low_pct <= max_near_low_pct
            and float(orderbook_features.bid_ask_imbalance) >= min_imbalance
            and float(orderbook_features.depth_pressure) >= min_depth_pressure
        )

    def _candle_capitulation_low(self, details: dict) -> bool:
        max_near_low_pct = float(self.config.get("signals", "short_exhaustion_candle_near_low_pct", default=0.10))
        min_impulse_pct = float(self.config.get("signals", "short_exhaustion_min_impulse_down_pct", default=1.0))
        max_rsi = float(self.config.get("signals", "short_exhaustion_max_rsi", default=40.0))
        max_macd_hist = float(self.config.get("signals", "short_exhaustion_max_macd_hist", default=-0.004))
        return (
            float(details["near_low_pct"]) <= max_near_low_pct
            and float(details["impulse_down_pct"]) >= min_impulse_pct
            and (
                float(details["rsi"]) <= max_rsi
                or float(details["macd_hist"]) <= max_macd_hist
            )
        )

    def _manage_open_position(
        self,
        executor: BrokerExecutor,
        last_price: float,
        now: datetime,
        risk_config: RiskConfig,
        execution_df=None,
        confirmation_df=None,
        context_df=None,
        flip_signal=None,
        ml_exit_reason=None,
        ml_exit_prediction=None,
    ) -> None:
        position = self.state.position
        exit_candle = self._latest_exit_candle(execution_df)
        management_price = last_price
        session_reason = self._session_block_reason(now)
        if session_reason:
            self._close_open_position(executor, position, last_price, session_reason, now)
            return
        if must_flatten_before_clearing(now, risk_config):
            self._close_open_position(executor, position, last_price, "pre_clearing_flatten", now)
            return

        exit_reason, exit_price = self._exit_from_candle(position, exit_candle, last_price)
        if exit_reason:
            self._close_open_position(
                executor,
                position,
                exit_price,
                exit_reason,
                now,
                details=self._exit_candle_details(exit_candle, "1m"),
            )
            return

        if ml_exit_reason:
            confirmed, pending_details = self._exit_signal_confirmed(
                kind="ml_exit",
                position=position,
                reason=ml_exit_reason,
                now=now,
                price=management_price,
                confirmation_df=confirmation_df,
            )
            if not confirmed:
                self._record_decision(
                    now,
                    "hold",
                    "ml_exit_confirmation_pending",
                    details={
                        "price": management_price,
                        "exit_reason": ml_exit_reason,
                        "feedback": (
                            ml_exit_prediction.as_metadata()
                            if ml_exit_prediction is not None
                            else None
                        ),
                        **pending_details,
                        **self._exit_candle_details(exit_candle, "1m"),
                    },
                    position=position,
                )
                return
            self._close_open_position(
                executor,
                position,
                management_price,
                ml_exit_reason,
                now,
                details={
                    "price": management_price,
                    "feedback": (
                        ml_exit_prediction.as_metadata() if ml_exit_prediction is not None else None
                    ),
                    **pending_details,
                    **self._exit_candle_details(exit_candle, "1m"),
                },
            )
            return

        if (
            flip_signal is not None
            and flip_signal.side != Side.FLAT
            and flip_signal.side != position.side
        ):
            confirmed, pending_details = self._exit_signal_confirmed(
                kind="signal_flip",
                position=position,
                reason="signal-flip",
                now=now,
                price=management_price,
                confirmation_df=confirmation_df,
                target_side=flip_signal.side,
                signal_confidence=flip_signal.confidence,
            )
            if not confirmed:
                self._record_decision(
                    now,
                    "hold",
                    "signal_flip_confirmation_pending",
                    details={
                        "price": management_price,
                        "signal_side": flip_signal.side.value,
                        "signal_reason": flip_signal.reason,
                        "signal_confidence": flip_signal.confidence,
                        **pending_details,
                        **self._exit_candle_details(exit_candle, "1m"),
                    },
                    position=position,
                )
                return
            self._close_open_position(
                executor,
                position,
                management_price,
                "signal-flip",
                now,
                details={
                    "price": management_price,
                    "signal_side": flip_signal.side.value,
                    "signal_reason": flip_signal.reason,
                    "signal_confidence": flip_signal.confidence,
                    **pending_details,
                    **self._exit_candle_details(exit_candle, "1m"),
                },
            )
            return

        self._pending_exit_signal = None
        breakeven_moved = move_stop_to_breakeven(position, management_price, risk_config)
        update_trailing_stop(position, management_price, risk_config)
        if exit_candle is None and trailing_stop_hit(position, management_price):
            self._close_open_position(executor, position, management_price, "trailing_stop_hit", now)
            return

        if self._take_profit_reached(position, management_price, "take_profit1"):
            result = executor.take_partial(position, management_price, risk_config.partial_take_fraction)
            self._record_decision(
                now,
                "partial_accepted" if result.accepted else "partial_rejected",
                result.reason,
                details={
                    "price": management_price,
                    "lots": result.lots,
                    "take_profit1": position.take_profit1,
                },
                position=position,
            )
            executor.apply_partial(position, result)
            self._move_stop_after_legacy_tp1(position)
            return

        if should_take_partial(position, management_price, risk_config):
            result = executor.take_partial(position, management_price, risk_config.partial_take_fraction)
            self._record_decision(
                now,
                "partial_accepted" if result.accepted else "partial_rejected",
                result.reason,
                details={"price": management_price, "lots": result.lots},
                position=position,
            )
            executor.apply_partial(position, result)
            return
        self._record_decision(
            now,
            "hold",
            "position_open",
            details={
                "price": management_price,
                "breakeven_moved": breakeven_moved,
                "stop_price": position.stop_price,
                "trailing_stop": position.trailing_stop,
                **self._exit_candle_details(exit_candle, "1m"),
            },
            position=position,
        )

    def _close_open_position(
        self,
        executor: BrokerExecutor,
        position,
        price: float,
        reason: str,
        now: datetime,
        details: dict | None = None,
    ) -> None:
        result = executor.close_position(position, price, reason)
        close_details = {"price": price, "lots": result.lots}
        if details:
            close_details.update(details)
        self._record_decision(
            now,
            "close_accepted" if result.accepted else "close_rejected",
            result.reason,
            details=close_details,
            position=position,
        )
        self._remember_full_exit(position, now, result.reason, result.accepted, price)
        if result.accepted:
            self._pending_exit_signal = None
        executor.apply_close(position, result)
        if result.accepted and self.paper_portfolio is not None:
            self._sync_paper_trade_feedback()

    def _sync_paper_trade_feedback(self) -> None:
        if not bool(
            self.config.get("learning", "paper_trade_feedback_enabled", default=False)
        ):
            return
        try:
            report = sync_paper_trade_feedback(self.config)
            self.logger.info(
                "paper_trade_feedback_synced",
                extra={
                    "event": "paper_trade_feedback_synced",
                    "details": {
                        "completed_trades": report.completed_trades,
                        "matched_entries": report.matched_entries,
                        "labels_added": report.labels_added,
                        "labels_total": report.labels_total,
                    },
                },
            )
        except Exception as exc:
            self.logger.exception(
                "paper_trade_feedback_sync_failed",
                extra={
                    "event": "paper_trade_feedback_sync_failed",
                    "details": {"error": str(exc)},
                },
            )

    @staticmethod
    def _latest_exit_candle(context_df):
        if context_df is None or getattr(context_df, "empty", True):
            return None
        return context_df.iloc[-1]

    @staticmethod
    def _candle_close(candle, fallback_price: float) -> float:
        if candle is None:
            return fallback_price
        try:
            return float(candle["close"])
        except (KeyError, TypeError, ValueError):
            return fallback_price

    @staticmethod
    def _exit_from_candle(position, candle, fallback_price: float) -> tuple[str | None, float]:
        if not position.is_open:
            return None, fallback_price
        if candle is None or not TradingBot._candle_started_after_entry(position, candle):
            return TradingBot._exit_from_price(position, fallback_price)

        high = float(candle["high"])
        low = float(candle["low"])
        if position.side == Side.LONG:
            if position.stop_price is not None and low <= position.stop_price:
                return "hard_stop_hit", position.stop_price
            if position.take_profit2 is not None and high >= position.take_profit2:
                return "take_profit_2_5r", position.take_profit2
        if position.side == Side.SHORT:
            if position.stop_price is not None and high >= position.stop_price:
                return "hard_stop_hit", position.stop_price
            if position.take_profit2 is not None and low <= position.take_profit2:
                return "take_profit_2_5r", position.take_profit2
        return None, fallback_price

    _exit_from_15m_candle = _exit_from_candle

    @staticmethod
    def _exit_from_price(position, price: float) -> tuple[str | None, float]:
        if position.side == Side.LONG:
            if position.stop_price is not None and price <= position.stop_price:
                return "hard_stop_hit", position.stop_price
            if position.take_profit2 is not None and price >= position.take_profit2:
                return "take_profit_2_5r", position.take_profit2
        if position.side == Side.SHORT:
            if position.stop_price is not None and price >= position.stop_price:
                return "hard_stop_hit", position.stop_price
            if position.take_profit2 is not None and price <= position.take_profit2:
                return "take_profit_2_5r", position.take_profit2
        return None, price

    @staticmethod
    def _candle_started_after_entry(position, candle) -> bool:
        opened_at = getattr(position, "opened_at", None)
        timestamp = getattr(candle, "name", None)
        if opened_at is None or timestamp is None:
            return False
        if hasattr(timestamp, "to_pydatetime"):
            timestamp = timestamp.to_pydatetime()
        if not isinstance(timestamp, datetime):
            return False
        opened = opened_at if opened_at.tzinfo else opened_at.replace(tzinfo=timezone.utc)
        started = timestamp if timestamp.tzinfo else timestamp.replace(tzinfo=timezone.utc)
        return started >= opened

    @staticmethod
    def _exit_candle_details(candle, timeframe: str = "15m") -> dict:
        if candle is None:
            return {}
        details = {}
        for column in ("open", "high", "low", "close"):
            try:
                details[f"{timeframe}_{column}"] = float(candle[column])
            except (KeyError, TypeError, ValueError):
                pass
        timestamp = getattr(candle, "name", None)
        if timestamp is not None:
            details[f"{timeframe}_timestamp"] = (
                timestamp.isoformat() if hasattr(timestamp, "isoformat") else str(timestamp)
            )
        return details

    def _exit_signal_confirmed(
        self,
        *,
        kind: str,
        position,
        reason: str,
        now: datetime,
        price: float,
        confirmation_df=None,
        target_side: Side | None = None,
        signal_confidence: float | None = None,
    ) -> tuple[bool, dict]:
        if kind == "ml_exit" and reason.startswith("ml_exit_opposite"):
            hold_allowed, hold_details = self._opposite_exit_min_hold_allows(position, now)
            if not hold_allowed:
                return False, hold_details
        if kind == "signal_flip":
            flip_allowed, flip_details = self._signal_flip_hysteresis_allows(
                position=position,
                now=now,
                signal_confidence=signal_confidence,
            )
            if not flip_allowed:
                return False, flip_details

        target_side = target_side or self._exit_target_side_from_reason(reason)
        required = self._exit_confirmation_count(kind)
        key = f"{kind}:{position.side.value}:{reason}:{target_side.value if target_side else 'none'}"
        pending = self._pending_exit_signal if self._pending_exit_signal else {}
        if pending.get("key") != key:
            pending = {
                "key": key,
                "kind": kind,
                "count": 0,
                "first_seen_at": now,
                "reason": reason,
            }
        pending["count"] = int(pending.get("count", 0)) + 1
        pending["last_seen_at"] = now
        self._pending_exit_signal = pending

        confirmation_state, confirmation_details = self._confirmation_5m_state(confirmation_df)
        confirmed_by_5m = (
            target_side == Side.LONG
            and confirmation_state == "confirm_long"
            or target_side == Side.SHORT
            and confirmation_state == "confirm_short"
        )
        adverse_r = self._adverse_move_r(position, price)
        adverse_threshold = self._exit_adverse_r_threshold(kind)
        confirmed_by_adverse = adverse_r >= adverse_threshold if adverse_threshold > 0 else False
        confirmed_by_count = int(pending["count"]) >= required
        details = {
            "exit_confirmation_kind": kind,
            "exit_confirmation_count": int(pending["count"]),
            "exit_confirmation_required": required,
            "exit_confirmation_target_side": target_side.value if target_side else None,
            "exit_confirmation_5m_state": confirmation_state,
            "exit_confirmation_5m": confirmation_details,
            "exit_confirmation_adverse_r": round(adverse_r, 4),
            "exit_confirmation_adverse_threshold": adverse_threshold,
            "exit_confirmed_by_count": confirmed_by_count,
            "exit_confirmed_by_5m": confirmed_by_5m,
            "exit_confirmed_by_adverse": confirmed_by_adverse,
        }
        confirmed = confirmed_by_count or confirmed_by_5m or confirmed_by_adverse
        if confirmed:
            self._pending_exit_signal = None
        return confirmed, details

    def _exit_confirmation_count(self, kind: str) -> int:
        if kind == "ml_exit":
            return max(
                1,
                int(self.config.get("learning", "control_exit_confirmations", default=3)),
            )
        return max(
            1,
            int(
                self.config.get(
                    "signal_flip",
                    "exit_confirmations",
                    default=self.config.get("signals", "flip_exit_confirmations", default=3),
                )
            ),
        )

    def _exit_adverse_r_threshold(self, kind: str) -> float:
        if kind == "ml_exit":
            return float(
                self.config.get("learning", "control_exit_adverse_r_threshold", default=0.35)
            )
        return float(
            self.config.get("signals", "flip_exit_adverse_r_threshold", default=0.35)
        )

    def _signal_flip_hysteresis_allows(
        self,
        *,
        position,
        now: datetime,
        signal_confidence: float | None,
    ) -> tuple[bool, dict]:
        if not bool(self.config.get("signal_flip", "close_on_flip", default=True)):
            return False, {"signal_flip_close_disabled": True}

        min_hold_bars = int(
            self.config.get("signal_flip", "min_hold_bars_before_flip_exit", default=0)
        )
        hold_bars = self._position_hold_bars(position, now)
        details = {
            "signal_flip_hold_bars": round(hold_bars, 3),
            "signal_flip_min_hold_bars": min_hold_bars,
        }
        if min_hold_bars > 0 and hold_bars < min_hold_bars:
            return False, {**details, "signal_flip_blocked": "min_hold_bars"}

        min_strength = float(
            self.config.get("signal_flip", "require_opposite_signal_strength", default=0.0)
        )
        if signal_confidence is not None:
            details["signal_flip_confidence"] = signal_confidence
        if signal_confidence is not None and min_strength > 0 and signal_confidence < min_strength:
            return False, {
                **details,
                "signal_flip_min_confidence": min_strength,
                "signal_flip_blocked": "opposite_signal_strength",
            }

        if bool(
            self.config.get("signal_flip", "require_opposite_ml_confirmation", default=False)
        ):
            ready, reason, model_details = self.feedback_model.control_model_validation()
            details["signal_flip_ml_confirmation_ready"] = ready
            details["signal_flip_ml_confirmation_reason"] = reason
            if not ready:
                return False, {
                    **details,
                    "signal_flip_blocked": "ml_confirmation_not_ready",
                    "model_validation": model_details,
                }

        return True, details

    def _opposite_exit_min_hold_allows(self, position, now: datetime) -> tuple[bool, dict]:
        min_hold_bars = int(
            self.config.get("signal_flip", "min_hold_bars_before_flip_exit", default=0)
        )
        hold_bars = self._position_hold_bars(position, now)
        details = {
            "opposite_exit_hold_bars": round(hold_bars, 3),
            "opposite_exit_min_hold_bars": min_hold_bars,
        }
        if min_hold_bars > 0 and hold_bars < min_hold_bars:
            return False, {**details, "opposite_exit_blocked": "min_hold_bars"}
        return True, details

    def _position_hold_bars(self, position, now: datetime) -> float:
        bar_seconds = float(self.config.get("signal_flip", "hold_bar_seconds", default=60.0))
        opened_at = getattr(position, "opened_at", None)
        if opened_at is None or bar_seconds <= 0:
            return 0.0
        if opened_at.tzinfo is None:
            opened_at = opened_at.replace(tzinfo=timezone.utc)
        current = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
        return max(0.0, (current - opened_at).total_seconds() / bar_seconds)

    @staticmethod
    def _exit_target_side_from_reason(reason: str) -> Side | None:
        if reason.endswith(":long") or ":long:" in reason:
            return Side.LONG
        if reason.endswith(":short") or ":short:" in reason:
            return Side.SHORT
        return None

    @staticmethod
    def _adverse_move_r(position, price: float) -> float:
        if not position.is_open or position.avg_price <= 0:
            return 0.0
        if position.stop_price is not None:
            risk = abs(position.avg_price - position.stop_price)
        else:
            risk = position.avg_price * 0.01
        if risk <= 0:
            return 0.0
        if position.side == Side.LONG:
            adverse = max(0.0, position.avg_price - price)
        elif position.side == Side.SHORT:
            adverse = max(0.0, price - position.avg_price)
        else:
            adverse = 0.0
        return adverse / risk

    @staticmethod
    def _take_profit_reached(position, last_price: float, field: str) -> bool:
        if not position.is_open or position.partial_taken and field == "take_profit1":
            return False
        target = getattr(position, field, None)
        if target is None:
            return False
        if position.side == Side.LONG:
            return last_price >= target
        if position.side == Side.SHORT:
            return last_price <= target
        return False

    def _blocked_by_reentry_cooldown(self, side: Side, now: datetime) -> bool:
        reason, _ = self._reentry_cooldown_block(side, now)
        return reason is not None

    def _reentry_cooldown_block(self, side: Side, now: datetime) -> tuple[str | None, dict]:
        self._restore_last_full_exit(now)
        if self._last_full_exit is None:
            return None, {}
        exit_side, exit_time, exit_reason, exit_pnl_ticks = self._last_full_exit
        elapsed_seconds = (now - exit_time).total_seconds()
        details = {
            "last_exit_side": exit_side.value,
            "last_exit_reason": exit_reason,
            "last_exit_pnl_ticks": exit_pnl_ticks,
            "elapsed_seconds": elapsed_seconds,
        }

        loss_minutes = float(
            self.config.get("signals", "reentry_cooldown_after_loss_minutes", default=0)
        )
        if loss_minutes > 0 and exit_pnl_ticks < 0 and elapsed_seconds <= loss_minutes * 60:
            return "reentry_cooldown_after_loss", {
                **details,
                "cooldown_minutes": loss_minutes,
            }

        exit_minutes = float(
            self.config.get("signals", "reentry_cooldown_after_exit_minutes", default=0)
        )
        if exit_minutes > 0 and elapsed_seconds <= exit_minutes * 60:
            return "reentry_cooldown_after_exit", {
                **details,
                "cooldown_minutes": exit_minutes,
            }

        take_profit_minutes = float(
            self.config.get(
                "signals",
                "reentry_cooldown_after_take_profit_minutes",
                default=0,
            )
        )
        if (
            take_profit_minutes > 0
            and exit_side == side
            and _is_take_profit_exit(exit_reason)
            and elapsed_seconds <= take_profit_minutes * 60
        ):
            return "reentry_cooldown_after_take_profit", {
                **details,
                "cooldown_minutes": take_profit_minutes,
            }
        return None, {}

    def _restore_last_full_exit(self, now: datetime) -> None:
        if self.paper_portfolio is None:
            return
        snapshot = self.paper_portfolio.risk_snapshot(now, self.config.timezone)
        if snapshot.last_exit_side is None or snapshot.last_exit_time is None:
            return
        self._last_full_exit = (
            snapshot.last_exit_side,
            snapshot.last_exit_time,
            snapshot.last_exit_reason or "unknown",
            snapshot.last_exit_pnl_ticks,
        )

    def _paper_risk_entry_block(
        self, now: datetime, risk_config: RiskConfig
    ) -> tuple[str | None, dict]:
        if self.paper_portfolio is None:
            return None, {}
        snapshot = self.paper_portfolio.risk_snapshot(now, self.config.timezone)
        details = {
            "daily_net_pnl": round(snapshot.daily_net_pnl, 2),
            "completed_trades_today": snapshot.completed_trades_today,
            "consecutive_losses": snapshot.consecutive_losses,
            "consecutive_hard_stops": snapshot.consecutive_hard_stops,
        }
        daily_limit = risk_config.deposit_value * _fraction_from_percent_or_fraction(
            risk_config.daily_max_loss_pct
        )
        details["daily_loss_limit"] = round(daily_limit, 2)
        if daily_limit > 0 and snapshot.daily_net_pnl <= -daily_limit:
            return "daily_loss_limit_reached", details
        if (
            risk_config.stop_after_consecutive_hard_stops > 0
            and snapshot.consecutive_hard_stops
            >= risk_config.stop_after_consecutive_hard_stops
        ):
            return "consecutive_hard_stop_limit_reached", details
        if (
            risk_config.stop_after_consecutive_losses > 0
            and snapshot.consecutive_losses >= risk_config.stop_after_consecutive_losses
        ):
            return "consecutive_loss_limit_reached", details
        return None, details

    def _session_block_reason(self, now: datetime) -> str | None:
        return trading_session_block_reason(
            now,
            timezone=self.config.timezone,
            trading_start=self.config.get("session", "trading_start", default=None),
            trading_end=self.config.get("session", "trading_end", default=None),
            forced_flat_hours=list(
                self.config.get("session", "forced_flat_hours", default=[])
            ),
            forced_flat_weekdays=list(
                self.config.get("session", "forced_flat_weekdays", default=[])
            ),
        )

    @staticmethod
    def _position_pnl_ticks(position, price: float) -> float:
        if not position.is_open or position.avg_price <= 0:
            return 0.0
        if position.side == Side.LONG:
            return price - position.avg_price
        if position.side == Side.SHORT:
            return position.avg_price - price
        return 0.0

    def _remember_full_exit(
        self,
        position,
        now: datetime,
        reason: str,
        accepted: bool,
        price: float,
    ) -> None:
        if accepted and position.is_open:
            self._last_full_exit = (
                position.side,
                now,
                reason,
                self._position_pnl_ticks(position, price),
            )

    @staticmethod
    def _move_stop_after_legacy_tp1(position) -> None:
        if not position.is_open or position.stop_price is None:
            return
        if position.side == Side.LONG:
            risk = max(position.avg_price - position.stop_price, 0.0)
            new_stop = position.avg_price + risk * 0.25
            position.stop_price = max(position.stop_price, new_stop)
            position.trailing_stop = max(position.trailing_stop or new_stop, new_stop)
        elif position.side == Side.SHORT:
            risk = max(position.stop_price - position.avg_price, 0.0)
            new_stop = position.avg_price - risk * 0.25
            position.stop_price = min(position.stop_price, new_stop)
            position.trailing_stop = min(position.trailing_stop or new_stop, new_stop)

    def _apply_feedback(
        self,
        signal,
        *,
        execution_df,
        confirmation_df,
        context_df,
        orderbook_features,
        trade_flow,
        now: datetime,
    ):
        if not bool(self.config.get("learning", "enabled", default=False)):
            return signal
        if str(self.config.get("learning", "mode", default="off")) != "filter":
            return signal
        return self.feedback_model.apply(
            signal,
            execution_df=execution_df,
            confirmation_df=confirmation_df,
            context_df=context_df,
            orderbook=orderbook_features,
            trade_flow=trade_flow,
            now=now,
        )

    def _entry_signal_block_reason(self, signal) -> str | None:
        reason = str(getattr(signal, "reason", "")).lower()
        allow_fallback = bool(
            self.config.get(
                "execution",
                "allow_fallback_entries",
                default=self.config.get("signals", "allow_fallback_entries", default=False),
            )
        )
        if "fallback" in reason and not allow_fallback:
            return "fallback_entry_disabled"
        if signal.metadata.get("exploration") and not bool(
            self.config.get("learning", "exploration_enabled", default=False)
        ):
            return "exploration_entry_disabled"
        if not bool(
            self.config.get("execution", "allow_trade_without_promoted_model", default=False)
        ):
            ready, model_reason, _ = self.feedback_model.control_model_validation()
            candidate_execution = bool(
                signal.metadata.get("candidate_execution")
                and self._paper_candidate_execution_enabled()
            )
            if not ready and not candidate_execution:
                return "ml_not_ready_no_fallback"
            if not candidate_execution and not reason.startswith("ml_entry"):
                return "non_ml_entry_without_promoted_consensus_disabled"
        return None

    def _paper_candidate_execution_enabled(self) -> bool:
        return bool(
            self.config.dry_run
            and not self.config.live_enabled
            and self.config.get("learning", "candidate_can_trade", default=False)
        )

    def _ml_control_enabled(self) -> bool:
        return bool(self.config.get("learning", "enabled", default=False)) and str(
            self.config.get("learning", "mode", default="off")
        ) in {"control", "shadow_then_control"}

    def _reload_feedback_model(self) -> None:
        self.feedback_model = FeedbackModel.from_runtime_config(self.config)
        self.logger.info(
            "feedback_model_reloaded",
            extra={"event": "feedback_model_reloaded", "details": {}},
        )

    def _indicator_frame(self, timeframe: str, now: datetime | None = None):
        candles = {
            "1min": list(self.state.candles_1m),
            "5min": list(self.state.candles_5m),
            "15min": list(self.state.candles_15m),
        }[timeframe]
        if now is not None:
            minutes = {"1min": 1, "5min": 5, "15min": 15}[timeframe]
            current = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
            candles = [
                candle
                for candle in candles
                if _aware_utc(candle.timestamp) + timedelta(minutes=minutes) <= current
            ]
        return add_indicators(
            candles_to_frame(candles),
            ema_fast=int(self.config.get("indicators", "ema_fast")),
            ema_slow=int(self.config.get("indicators", "ema_slow")),
            rsi_period=int(self.config.get("indicators", "rsi_period")),
            atr_period=int(self.config.get("indicators", "atr_period", default=14)),
            adx_period=int(self.config.get("indicators", "adx_period", default=14)),
            macd_fast=int(self.config.get("indicators", "macd_fast", default=12)),
            macd_slow=int(self.config.get("indicators", "macd_slow", default=26)),
            macd_signal=int(self.config.get("indicators", "macd_signal", default=9)),
            bollinger_period=int(self.config.get("indicators", "bollinger_period")),
            bollinger_std=float(self.config.get("indicators", "bollinger_std")),
            volume_ma_period=int(self.config.get("indicators", "volume_ma_period")),
        )

    def _orderbook_features(self, now: datetime | None = None):
        return analyze_order_book(
            self.state.order_book,
            self.state.previous_order_book,
            levels=int(self.config.get("orderbook", "imbalance_levels")),
            wall_multiplier=float(self.config.get("orderbook", "wall_multiplier")),
            absorption_drop_pct=float(self.config.get("orderbook", "absorption_drop_pct")),
            min_wall_notional=float(self.config.get("orderbook", "min_wall_notional")),
            now=now,
        )

    def _entry_market_data_block(self, now: datetime, orderbook_features) -> tuple[str | None, dict]:
        if not bool(
            self.config.get(
                "execution",
                "block_stale_entries",
                default=self.config.get("market_data", "block_entries_when_stale", default=True),
            )
        ):
            return None, {}

        required_source = str(
            self.config.get(
                "market_data",
                "required_entry_orderbook_source",
                default=self.config.get("microstructure", "required_orderbook_source", default="live"),
            )
        )
        if bool(self.config.get("execution", "require_live_orderbook", default=True)):
            required_source = "live"
        source = str(getattr(orderbook_features, "source", "missing") or "missing")
        if required_source and source != required_source:
            return (
                "entry_market_data_untrusted",
                {
                    "orderbook_source": source,
                    "required_source": required_source,
                },
            )

        age_limits = [
            float(self.config.get("market_data", "max_entry_staleness_seconds", default=15.0)),
            float(self.config.get("microstructure", "max_orderbook_age_seconds", default=15.0)),
            float(self.config.get("execution", "max_entry_staleness_sec", default=15.0)),
        ]
        max_age_seconds = min(limit for limit in age_limits if limit > 0)
        age_seconds = getattr(orderbook_features, "age_seconds", None)
        if age_seconds is None and self.state.last_stream_update is not None:
            age_seconds = (now - self.state.last_stream_update).total_seconds()
        if age_seconds is None:
            return (
                "entry_market_data_age_unknown",
                {
                    "orderbook_source": source,
                    "max_age_seconds": max_age_seconds,
                },
            )
        if max_age_seconds > 0 and float(age_seconds) > max_age_seconds:
            return (
                "entry_market_data_stale",
                {
                    "orderbook_source": source,
                    "age_seconds": float(age_seconds),
                    "max_age_seconds": max_age_seconds,
                },
            )
        return None, {}

    def _final_entry_risk_check(
        self,
        *,
        signal,
        lots: int,
        now: datetime,
        risk_config: RiskConfig,
        orderbook_features,
        cost_check,
    ) -> tuple[str | None, dict]:
        details = {
            "side": signal.side.value,
            "lots": lots,
            "signal_reason": signal.reason,
            "dry_run": self.config.dry_run,
            "live_enabled": self.config.live_enabled,
            "max_position_lots": risk_config.max_position_lots,
            "expected_move_ticks": getattr(cost_check, "expected_move_ticks", None),
            "min_required_ticks": getattr(cost_check, "min_required_ticks", None),
            "spread_bps": getattr(orderbook_features, "spread_bps", None),
            "orderbook_age_sec": getattr(orderbook_features, "age_seconds", None),
        }
        if not self.config.dry_run or self.config.live_enabled:
            return "live_execution_disabled", details
        if lots <= 0:
            return "size_below_min", details
        if lots > risk_config.max_position_lots:
            return "position_lots_above_max", details

        market_reason, market_details = self._entry_market_data_block(now, orderbook_features)
        if market_reason:
            return market_reason, {**details, **market_details}
        if orderbook_features.spread_bps is not None and not spread_is_acceptable(
            orderbook_features,
            float(self.config.get("orderbook", "max_spread_bps")),
        ):
            return "spread_too_wide", details

        if not bool(
            self.config.get("execution", "allow_trade_without_promoted_model", default=False)
        ):
            ready, model_reason, model_details = self.feedback_model.control_model_validation()
            details["model_validation"] = {"ready": ready, "reason": model_reason, **model_details}
            candidate_execution = bool(
                signal.metadata.get("candidate_execution")
                and self._paper_candidate_execution_enabled()
            )
            if not ready and not candidate_execution:
                return "ml_not_ready_no_fallback", details
            if not candidate_execution and not str(signal.reason).lower().startswith("ml_entry"):
                return "non_ml_entry_without_promoted_consensus_disabled", details

        return None, details

    def _signal_config(self) -> SignalConfig:
        return SignalConfig(
            rsi_overbought=float(self.config.get("indicators", "rsi_overbought")),
            rsi_oversold=float(self.config.get("indicators", "rsi_oversold")),
            min_bollinger_width_pct=float(self.config.get("indicators", "min_bollinger_width_pct")),
            volume_multiplier=float(self.config.get("indicators", "volume_multiplier")),
            hold_above_ema_bars=int(self.config.get("signals", "hold_above_ema_bars")),
            support_resistance_lookback=int(self.config.get("signals", "support_resistance_lookback")),
            max_level_distance_pct=float(self.config.get("signals", "max_level_distance_pct")),
            min_confidence=float(self.config.get("signals", "min_confidence")),
            min_imbalance=float(self.config.get("orderbook", "min_imbalance")),
            trade_flow_min_buy_ratio=float(self.config.get("signals", "trade_flow_min_buy_ratio")),
            trade_flow_min_sell_ratio=float(self.config.get("signals", "trade_flow_min_sell_ratio")),
            news_halt=bool(self.config.get("signals", "news_halt")),
            allow_rsi_extreme_with_strong_trend=bool(
                self.config.get("signals", "allow_rsi_extreme_with_strong_trend")
            ),
            require_15m_direction=bool(self.config.get("signals", "require_15m_direction", default=True)),
            engine=str(self.config.get("signals", "engine", default="python")),
            legacy_bridge_path=str(
                self._project_path(
                    self.config.get(
                        "signals",
                        "legacy_bridge_path",
                        default="reference/ngn6_signal_source/bridge/compute_signal.js",
                    )
                )
            ),
            legacy_node_command=str(self.config.get("signals", "legacy_node_command", default="node")),
            legacy_timeout_seconds=float(
                self.config.get("signals", "legacy_timeout_seconds", default=3.0)
            ),
            legacy_timeframe=str(self.config.get("signals", "legacy_timeframe", default="intraday")),
            legacy_min_candles=int(self.config.get("signals", "legacy_min_candles", default=80)),
            legacy_max_candles=int(self.config.get("signals", "legacy_max_candles", default=320)),
            legacy_min_probability=float(
                self.config.get("signals", "legacy_min_probability", default=52.0)
            ),
            legacy_news_bias=str(self.config.get("signals", "legacy_news_bias", default="auto")),
            legacy_retest=str(self.config.get("signals", "legacy_retest", default="auto")),
            legacy_structure=str(self.config.get("signals", "legacy_structure", default="auto")),
            legacy_event_risk=str(self.config.get("signals", "legacy_event_risk", default="none")),
            legacy_impulse_enabled=bool(
                self.config.get("signals", "legacy_impulse_enabled", default=True)
            ),
            legacy_impulse_move_pct=float(
                self.config.get("signals", "legacy_impulse_move_pct", default=1.2)
            ),
            legacy_impulse_breakout_buffer_pct=float(
                self.config.get("signals", "legacy_impulse_breakout_buffer_pct", default=0.03)
            ),
            legacy_impulse_min_trend=float(
                self.config.get("signals", "legacy_impulse_min_trend", default=0.28)
            ),
            legacy_impulse_min_momentum=float(
                self.config.get("signals", "legacy_impulse_min_momentum", default=0.24)
            ),
            legacy_impulse_min_candles=int(
                self.config.get("signals", "legacy_impulse_min_candles", default=2)
            ),
            legacy_impulse_min_probability=float(
                self.config.get("signals", "legacy_impulse_min_probability", default=55.5)
            ),
            legacy_impulse_max_probability=float(
                self.config.get("signals", "legacy_impulse_max_probability", default=63.5)
            ),
            microstructure_enabled=bool(self.config.get("microstructure", "enabled", default=True)),
            require_microstructure=bool(
                self.config.get("microstructure", "require_for_entry", default=False)
            ),
            max_orderbook_age_seconds=float(
                self.config.get("microstructure", "max_orderbook_age_seconds", default=8.0)
            ),
            min_book_pressure=float(
                self.config.get("microstructure", "min_book_pressure", default=0.12)
            ),
            min_trade_pressure=float(
                self.config.get("microstructure", "min_trade_pressure", default=0.12)
            ),
            min_trade_flow_volume=float(
                self.config.get("microstructure", "min_trade_flow_volume", default=0.0)
            ),
            require_trade_flow=bool(
                self.config.get("microstructure", "require_trade_flow", default=False)
            ),
            max_adverse_mid_move_bps=float(
                self.config.get("microstructure", "max_adverse_mid_move_bps", default=8.0)
            ),
            max_spread_widening_bps=float(
                self.config.get("microstructure", "max_spread_widening_bps", default=8.0)
            ),
            ema_adx_macd_warmup_candles=int(
                self.config.get("signals", "ema_adx_macd_warmup_candles", default=80)
            ),
            ema_adx_macd_min_adx=float(
                self.config.get("signals", "ema_adx_macd_min_adx", default=20.0)
            ),
            ema_adx_macd_require_adx_rising=bool(
                self.config.get("signals", "ema_adx_macd_require_adx_rising", default=False)
            ),
            ema_adx_macd_long_rsi_min=float(
                self.config.get("signals", "ema_adx_macd_long_rsi_min", default=50.0)
            ),
            ema_adx_macd_long_rsi_max=float(
                self.config.get("signals", "ema_adx_macd_long_rsi_max", default=75.0)
            ),
            ema_adx_macd_short_rsi_min=float(
                self.config.get("signals", "ema_adx_macd_short_rsi_min", default=30.0)
            ),
            ema_adx_macd_short_rsi_max=float(
                self.config.get("signals", "ema_adx_macd_short_rsi_max", default=50.0)
            ),
            ema_adx_macd_min_trend_strength=float(
                self.config.get("signals", "ema_adx_macd_min_trend_strength", default=0.002)
            ),
            ema_adx_macd_min_signal_strength=float(
                self.config.get("signals", "ema_adx_macd_min_signal_strength", default=0.30)
            ),
            ema_adx_macd_signal_strength_trend_scale=float(
                self.config.get("signals", "ema_adx_macd_signal_strength_trend_scale", default=0.0125)
            ),
            ema_adx_macd_stop_atr_multiple=float(
                self.config.get("signals", "ema_adx_macd_stop_atr_multiple", default=1.5)
            ),
            ema_adx_macd_take_profit_r_multiple=float(
                self.config.get("signals", "ema_adx_macd_take_profit_r_multiple", default=2.5)
            ),
            ema_adx_macd_orderbook_required=bool(
                self.config.get("signals", "ema_adx_macd_orderbook_required", default=True)
            ),
            ema_adx_macd_always_trade=bool(
                self.config.get("signals", "ema_adx_macd_always_trade", default=False)
            ),
        )

    def _risk_config(self) -> RiskConfig:
        return RiskConfig(
            deposit_value=float(self.config.get("account", "deposit_value")),
            risk_per_trade_pct=float(self.config.get("risk", "risk_per_trade_pct")),
            max_risk_per_trade_pct=float(self.config.get("risk", "max_risk_per_trade_pct")),
            max_position_lots=int(self.config.get("risk", "max_position_lots")),
            min_position_lots=int(self.config.get("risk", "min_position_lots")),
            stop_buffer_ticks=int(self.config.get("risk", "stop_buffer_ticks")),
            min_price_increment=float(self.config.get("instrument", "min_price_increment")),
            money_value_per_price_step=float(self.config.get("instrument", "money_value_per_price_step")),
            partial_take_profit_pct=float(self.config.get("risk", "partial_take_profit_pct")),
            partial_take_fraction=float(self.config.get("risk", "partial_take_fraction")),
            trailing_stop_pct=float(self.config.get("risk", "trailing_stop_pct")),
            close_before_clearing_minutes=int(
                self.config.get("session", "close_positions_before_clearing_minutes")
            ),
            clearings=list(self.config.get("session", "clearings", default=[])),
            timezone=self.config.timezone,
            take_profit_r_multiple=float(self.config.get("risk", "take_profit_r_multiple", default=2.5)),
            breakeven_trigger_pct=float(
                self.config.get("risk", "breakeven_trigger_pct", default=0.75)
            ),
            trailing_profit_trigger_rub=float(
                self.config.get("risk", "trailing_profit_trigger_rub", default=0.0)
            ),
            trailing_profit_lock_ratio=float(
                self.config.get("risk", "trailing_profit_lock_ratio", default=0.35)
            ),
            notional_multiplier=float(self.config.get("paper", "notional_multiplier", default=0.0)),
            max_gross_exposure_multiplier=float(
                self.config.get("risk", "max_gross_exposure_multiplier", default=0.0)
            ),
            max_position_exposure_pct=float(
                self.config.get("risk", "max_position_exposure_pct", default=0.0)
            ),
            cash_reserve_pct=float(self.config.get("risk", "cash_reserve_pct", default=0.0)),
            max_drawdown_pct=float(self.config.get("risk", "max_drawdown_pct", default=0.0)),
            daily_max_loss_pct=float(
                self.config.get(
                    "risk",
                    "daily_max_loss_pct",
                    default=self.config.get("risk", "max_daily_loss_pct", default=0.0),
                )
            ),
            stop_after_consecutive_losses=int(
                self.config.get("risk", "stop_after_consecutive_losses", default=0)
            ),
            stop_after_consecutive_hard_stops=int(
                self.config.get("risk", "stop_after_consecutive_hard_stops", default=0)
            ),
        )

    def _execution_cost_config(self) -> ExecutionCostConfig:
        return ExecutionCostConfig(
            slippage_bps_assumption=float(
                self.config.get(
                    "execution",
                    "slippage_bps_assumption",
                    default=self.config.get("costs", "slippage_bps_per_side", default=0.0),
                )
            ),
            commission_per_lot_per_side=float(
                self.config.get("execution", "commission_per_lot_per_side", default=0.0)
            ),
            commission_round_trip_bps=float(
                self.config.get(
                    "execution",
                    "commission_round_trip_bps",
                    default=self.config.get("costs", "commission_roundtrip_bps", default=0.0),
                )
            ),
            min_expected_net_ticks=float(
                self.config.get(
                    "execution",
                    "min_expected_net_ticks",
                    default=self.config.get("costs", "min_expected_net_ticks", default=0.0),
                )
            ),
            min_price_increment=float(self.config.get("instrument", "min_price_increment")),
            money_value_per_price_step=float(self.config.get("instrument", "money_value_per_price_step")),
        )

    @staticmethod
    def _expected_move_pct(signal, risk_config: RiskConfig) -> float:
        if signal.take_profit1 and signal.price:
            return abs(signal.take_profit1 - signal.price) / abs(signal.price) * 100
        if signal.take_profit2 and signal.price:
            return abs(signal.take_profit2 - signal.price) / abs(signal.price) * 100
        return risk_config.partial_take_profit_pct

    def _project_path(self, value) -> Path:
        path = Path(str(value))
        if path.is_absolute():
            return path
        config_path = self.config.path.resolve()
        project_root = config_path.parent.parent if config_path.parent.name == "config" else config_path.parent
        return (project_root / path).resolve()

    def _last_close(self) -> float | None:
        if not self.state.candles_1m:
            return None
        return self.state.candles_1m[-1].close

    def _recent_trades(self, now: datetime):
        limit = int(self.config.get("data_collection", "recent_trades_limit", default=80))
        lookback_seconds = float(
            self.config.get(
                "data_collection",
                "raw_trade_lookback_seconds",
                default=max(
                    120,
                    int(self.config.get("signals", "trade_flow_lookback_seconds", default=30)),
                ),
            )
        )
        cutoff = now - timedelta(seconds=lookback_seconds)
        recent = [trade for trade in self.state.trades if trade.timestamp >= cutoff]
        return recent[-limit:] if limit > 0 else []

    def _market_context(self, now: datetime) -> dict | None:
        if not self.recorder.record_decision_context:
            return None
        trade_flow = analyze_trade_flow(
            list(self.state.trades),
            now,
            int(self.config.get("signals", "trade_flow_lookback_seconds", default=30)),
        )
        context = {
            "orderbook": self._orderbook_features(now),
            "trade_flow": trade_flow,
            "orderbook_snapshot": self.state.order_book,
            "previous_orderbook_snapshot": self.state.previous_order_book,
            "recent_trades": self._recent_trades(now),
        }
        trust_reason, trust_details = self._entry_market_data_block(now, context["orderbook"])
        feature_report = self._feature_snapshot_for_recording(now, context["orderbook"], trade_flow)
        context.update(feature_report)
        context.update(
            {
                "market_data_trusted": trust_reason is None,
                "market_data_trust_reason": trust_reason,
                "market_data_trust_details": trust_details,
            }
        )
        return context

    def _feature_snapshot_for_recording(self, now: datetime, orderbook_features, trade_flow):
        try:
            execution_df = self._indicator_frame("1min", now)
            confirmation_df = self._indicator_frame("5min", now)
            context_df = self._indicator_frame("15min", now)
            if execution_df.empty:
                return {
                    "features": {},
                    "feature_complete": False,
                    "feature_timestamp": None,
                    "missing_feature_fields": list(FEATURE_KEYS),
                    "feature_reject_reason": "no_closed_candles",
                }
            feature_timestamp = self._last_closed_feature_timestamp(execution_df, now)
            features = build_feature_snapshot(
                execution_df=execution_df,
                confirmation_df=confirmation_df,
                context_df=context_df,
                orderbook=orderbook_features,
                trade_flow=trade_flow,
                now=now,
            )
            missing = [key for key in FEATURE_KEYS if key not in features]
            return {
                "features": features,
                "feature_complete": not missing,
                "feature_timestamp": feature_timestamp.isoformat() if feature_timestamp else None,
                "missing_feature_fields": missing,
                "feature_reject_reason": "missing_feature_fields" if missing else None,
            }
        except Exception as exc:
            return {
                "features": {},
                "feature_complete": False,
                "feature_timestamp": None,
                "missing_feature_fields": list(FEATURE_KEYS),
                "feature_reject_reason": f"feature_builder_error:{type(exc).__name__}",
            }

    @staticmethod
    def _last_closed_feature_timestamp(frame, now: datetime) -> datetime | None:
        if frame.empty:
            return None
        if not hasattr(frame.index, "tz"):
            return None
        timestamp = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
        index = frame.index
        if index.tz is None:
            cutoff = timestamp.astimezone(timezone.utc).replace(tzinfo=None)
        else:
            cutoff = timestamp.astimezone(index.tz)
        closed = frame.loc[index <= cutoff]
        if closed.empty:
            return None
        value = closed.index[-1]
        if getattr(value, "tzinfo", None) is None:
            return value.to_pydatetime().replace(tzinfo=timezone.utc)
        return value.to_pydatetime()

    def _record_market(self, now: datetime, last_price: float, orderbook_features, trade_flow) -> None:
        trust_reason, trust_details = self._entry_market_data_block(now, orderbook_features)
        self.recorder.record_market(
            timestamp=now,
            ticker=str(self.config.get("instrument", "ticker")),
            last_price=last_price,
            orderbook=orderbook_features,
            trade_flow=trade_flow,
            candle_counts={
                "1min": len(self.state.candles_1m),
                "5min": len(self.state.candles_5m),
                "15min": len(self.state.candles_15m),
                "trades": len(self.state.trades),
            },
            position=self.state.position,
            orderbook_snapshot=self.state.order_book,
            recent_trades=self._recent_trades(now),
            market_data_trusted=trust_reason is None,
            market_data_trust_reason=trust_reason,
            market_data_trust_details=trust_details,
        )

    def _record_decision(
        self,
        now: datetime,
        action: str,
        reason: str,
        *,
        signal=None,
        details: dict | None = None,
        position=None,
    ) -> None:
        self.recorder.record_decision(
            timestamp=now,
            action=action,
            reason=reason,
            signal=signal,
            details=details or {},
            position=position,
            market_context=self._market_context(now),
        )

    def _log_skip(self, reason: str, details: dict, now: datetime | None = None) -> None:
        self.logger.debug("signal_skipped", extra={"event": "signal_skipped", "details": {"reason": reason, **details}})
        if now is not None:
            self._record_decision(now, "skip", reason, details=details)


def _is_take_profit_exit(reason: str) -> bool:
    return "take_profit" in reason


def _fraction_from_percent_or_fraction(value: float) -> float:
    parsed = max(0.0, float(value))
    return parsed / 100.0 if parsed > 1.0 else parsed


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
