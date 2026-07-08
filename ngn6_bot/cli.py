from __future__ import annotations

import argparse
import copy
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from ngn6_bot.backtest import (
    fetch_1m_history,
    run_replay_backtest,
    run_walk_forward,
    save_report,
    save_walk_forward_report,
)
from ngn6_bot.bot import TradingBot
from ngn6_bot.charting import fetch_day_candles, plot_indicator_chart
from ngn6_bot.config import load_config
from ngn6_bot.config import RuntimeConfig
from ngn6_bot.dashboard import run_dashboard
from ngn6_bot.learning.daily_oracle import generate_daily_oracle_from_api
from ngn6_bot.learning.feature_audit import audit_feature_completeness, save_feature_completeness_report
from ngn6_bot.learning.labeling import generate_labeling_charts
from ngn6_bot.learning.model_diagnostics import (
    DEFAULT_THRESHOLDS,
    generate_model_diagnostics,
    save_model_diagnostics,
)
from ngn6_bot.learning.promotion import check_model_eligibility, save_promotion_check
from ngn6_bot.learning.regime_report import generate_regime_report, save_regime_report
from ngn6_bot.learning.shadow import evaluate_shadow_predictions, save_shadow_report
from ngn6_bot.learning.training import train_feedback_from_api
from ngn6_bot.logging_json import setup_logging
from ngn6_bot.review import generate_review_from_api
from ngn6_bot.tbank import TInvestGateway


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ngn6-bot")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run live market-data loop.")
    run_parser.add_argument("--config", default="config/ngn6.yaml")

    smoke_parser = subparsers.add_parser("smoke", help="Validate config and local strategy code.")
    smoke_parser.add_argument("--config", default="config/ngn6.yaml")

    check_parser = subparsers.add_parser("check-api", help="Read-only T-Invest connectivity check.")
    check_parser.add_argument("--config", default="config/ngn6.yaml")

    strategy_parser = subparsers.add_parser(
        "check-strategy",
        help="Read-only API + one dry-run strategy evaluation.",
    )
    strategy_parser.add_argument("--config", default="config/ngn6.yaml")

    stream_parser = subparsers.add_parser(
        "check-stream",
        help="Read-only market-data stream check.",
    )
    stream_parser.add_argument("--config", default="config/ngn6.yaml")
    stream_parser.add_argument("--seconds", type=float, default=15)

    backtest_parser = subparsers.add_parser(
        "backtest",
        help="Fetch recent 1m candles and run a candle replay backtest.",
    )
    backtest_parser.add_argument("--config", default="config/ngn6.yaml")
    backtest_parser.add_argument("--minutes", type=int, default=4500)
    backtest_parser.add_argument("--report", default="reports/backtest.json")
    backtest_parser.add_argument("--promoted-only", action="store_true")

    wf_parser = subparsers.add_parser(
        "walk-forward",
        help="Run chronological folds over recent 1m candle history.",
    )
    wf_parser.add_argument("--config", default="config/ngn6.yaml")
    wf_parser.add_argument("--minutes", type=int, default=4500)
    wf_parser.add_argument("--folds", type=int, default=3)
    wf_parser.add_argument("--report", default="reports/walk_forward.json")
    wf_parser.add_argument("--promoted-only", action="store_true")

    chart_parser = subparsers.add_parser(
        "chart",
        help="Build a PNG chart with strategy indicators for a trading date.",
    )
    chart_parser.add_argument("--config", default="config/ngn6.yaml")
    chart_parser.add_argument("--date", default=None, help="YYYY-MM-DD, defaults to yesterday.")
    chart_parser.add_argument("--timeframe", choices=["1min", "5min", "15min"], default="15min")
    chart_parser.add_argument("--output", default=None)

    review_parser = subparsers.add_parser(
        "review",
        help="Build scheduled-review PNG charts for a trading date.",
    )
    review_parser.add_argument("--config", default="config/ngn6.yaml")
    review_parser.add_argument("--date", default=None, help="YYYY-MM-DD, defaults to today.")
    review_parser.add_argument("--label", default=None, help="Output label, defaults to local HHMM.")
    review_parser.add_argument(
        "--timeframes",
        nargs="+",
        choices=["1min", "5min", "15min"],
        default=None,
    )

    labeling_parser = subparsers.add_parser(
        "labeling",
        help="Build feedback-labeling charts for recent trading days.",
    )
    labeling_parser.add_argument("--config", default="config/ngn6.yaml")
    labeling_parser.add_argument("--days", type=int, default=5)
    labeling_parser.add_argument("--minutes", type=int, default=12_000)
    labeling_parser.add_argument("--timeframe", choices=["1min", "5min", "15min"], default="15min")
    labeling_parser.add_argument("--output-dir", default="reports/labeling")
    labeling_parser.add_argument(
        "--backtest-report",
        default=None,
        help="Optional backtest JSON report whose trades are overlaid on labeling charts.",
    )

    daily_oracle_parser = subparsers.add_parser(
        "daily-oracle",
        help="Build post-market oracle review and ML labels for one trading date.",
    )
    daily_oracle_parser.add_argument("--config", default="config/ngn6.yaml")
    daily_oracle_parser.add_argument("--date", default=None, help="YYYY-MM-DD, defaults to today.")
    daily_oracle_parser.add_argument("--minutes", type=int, default=12_000)
    daily_oracle_parser.add_argument("--output-dir", default="reports/daily_oracle")

    train_feedback_parser = subparsers.add_parser(
        "train-feedback",
        help="Train the feedback ensemble from API history and human labels.",
    )
    train_feedback_parser.add_argument("--config", default="config/ngn6.yaml")
    train_feedback_parser.add_argument("--minutes", type=int, default=90_000)
    train_feedback_parser.add_argument("--output", default=None)
    train_feedback_parser.add_argument("--min-examples", type=int, default=None)

    train_candidate_parser = subparsers.add_parser(
        "train-candidate",
        help="Train feedback ensemble to candidate path only.",
    )
    train_candidate_parser.add_argument("--config", default="config/ngn6.yaml")
    train_candidate_parser.add_argument("--minutes", type=int, default=90_000)
    train_candidate_parser.add_argument("--output", default=None)
    train_candidate_parser.add_argument("--min-examples", type=int, default=None)

    promotion_parser = subparsers.add_parser(
        "promotion-check",
        help="Check active or candidate model eligibility without promoting it.",
    )
    promotion_parser.add_argument("--config", default="config/ngn6.yaml")
    promotion_parser.add_argument(
        "--model",
        default="active",
        help="'active', 'candidate', or explicit model path.",
    )
    promotion_parser.add_argument("--report", default="reports/promotion_check.json")

    shadow_parser = subparsers.add_parser(
        "shadow-evaluate",
        help="Evaluate candidate shadow predictions against matured labels.",
    )
    shadow_parser.add_argument("--config", default="config/ngn6.yaml")
    shadow_parser.add_argument("--decisions", default=None)
    shadow_parser.add_argument("--labels", default=None)
    shadow_parser.add_argument("--report", default=None)

    feature_audit_parser = subparsers.add_parser(
        "feature-audit",
        help="Audit feature completeness and market-data trust in runtime JSONL.",
    )
    feature_audit_parser.add_argument("--config", default="config/ngn6.yaml")
    feature_audit_parser.add_argument("--decisions", default=None)
    feature_audit_parser.add_argument("--market", default=None)
    feature_audit_parser.add_argument("--report", default="reports/feature_completeness.json")

    regime_parser = subparsers.add_parser(
        "regime-report",
        help="Evaluate oracle-derived regime candidates without promoting or trading.",
    )
    regime_parser.add_argument("--config", default="config/ngn6.yaml")
    regime_parser.add_argument("--decisions", default=None)
    regime_parser.add_argument("--labels-dir", default=None)
    regime_parser.add_argument("--folds", type=int, default=8)
    regime_parser.add_argument("--report", default="reports/regime_report.json")

    diagnostics_parser = subparsers.add_parser(
        "model-diagnostics",
        help="Explain current candidate examples, thresholds, flat bias, and regimes.",
    )
    diagnostics_parser.add_argument("--config", default="config/ngn6.yaml")
    diagnostics_parser.add_argument(
        "--model",
        default="candidate",
        help="'active', 'candidate', or explicit model path.",
    )
    diagnostics_parser.add_argument("--decisions", default=None)
    diagnostics_parser.add_argument("--labels-dir", default=None)
    diagnostics_parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=list(DEFAULT_THRESHOLDS),
    )
    diagnostics_parser.add_argument("--top-features", type=int, default=20)
    diagnostics_parser.add_argument("--report", default="reports/model_diagnostics.json")

    dashboard_parser = subparsers.add_parser(
        "dashboard",
        help="Run the local paper-trading dashboard.",
    )
    dashboard_parser.add_argument("--config", default="config/ngn6.yaml")
    dashboard_parser.add_argument("--host", default=None)
    dashboard_parser.add_argument("--port", type=int, default=None)

    args = parser.parse_args(argv)
    config = load_config(args.config)
    logger = setup_logging(config.get("bot", "log_level", default="INFO"), config.get("bot", "log_file"))

    if args.command == "smoke":
        logger.info(
            "smoke_ok",
            extra={
                "event": "smoke_ok",
                "details": {
                    "dry_run": config.dry_run,
                    "ticker": config.get("instrument", "ticker"),
                    "news_halt": config.get("signals", "news_halt"),
                },
            },
        )
        return 0

    if args.command == "dashboard":
        host = args.host or str(config.get("dashboard", "host", default="127.0.0.1"))
        port = int(args.port or config.get("dashboard", "port", default=8080))
        logger.info(
            "dashboard_starting",
            extra={"event": "dashboard_starting", "details": {"host": host, "port": port}},
        )
        run_dashboard(config, host, port)
        return 0

    if args.command == "check-api":
        with TInvestGateway(config.token, config.raw, logger) as gateway:
            figi, uid = gateway.resolve_instrument()
            book = gateway.get_order_book(figi, int(config.get("orderbook", "depth")))
            logger.info(
                "api_check_ok",
                extra={
                    "event": "api_check_ok",
                    "details": {
                        "ticker": config.get("instrument", "ticker"),
                        "figi": figi,
                        "uid": uid,
                        "bids": len(book.bids),
                        "asks": len(book.asks),
                        "best_bid": book.bids[0].price if book.bids else None,
                        "best_ask": book.asks[0].price if book.asks else None,
                    },
                },
            )
        return 0

    if args.command == "check-strategy":
        bot = TradingBot(config, logger)
        bot.run_once_dry()
        return 0

    if args.command == "check-stream":
        bot = TradingBot(config, logger)
        bot.check_stream(args.seconds)
        return 0

    if args.command == "backtest":
        figi, candles = fetch_1m_history(config, logger, args.minutes)
        report = run_replay_backtest(config, candles, figi, promoted_only=args.promoted_only)
        save_report(report, args.report)
        logger.info(
            "backtest_ok",
            extra={
                "event": "backtest_ok",
                "details": {
                    "report": args.report,
                    "candles": report.candles,
                    "trades": report.metrics.trades,
                    "win_rate_pct": report.metrics.win_rate_pct,
                    "avg_trade_pct": report.metrics.avg_trade_pct,
                    "max_drawdown_pct": report.metrics.max_drawdown_pct,
                    "final_equity_pct": report.metrics.final_equity_pct,
                    "profit_factor": report.metrics.profit_factor,
                    "promoted_only": args.promoted_only,
                },
            },
        )
        return 0

    if args.command == "walk-forward":
        figi, candles = fetch_1m_history(config, logger, args.minutes)
        report = run_walk_forward(
            config,
            candles,
            figi,
            args.folds,
            promoted_only=args.promoted_only,
        )
        save_walk_forward_report(report, args.report)
        logger.info(
            "walk_forward_ok",
            extra={
                "event": "walk_forward_ok",
                "details": {
                    "report": args.report,
                    "folds": len(report.folds),
                    "trades_by_fold": [fold.metrics.trades for fold in report.folds],
                    "final_equity_pct_by_fold": [
                        fold.metrics.final_equity_pct for fold in report.folds
                    ],
                    "max_drawdown_pct_by_fold": [
                        fold.metrics.max_drawdown_pct for fold in report.folds
                    ],
                    "promoted_only": args.promoted_only,
                },
            },
        )
        return 0

    if args.command == "chart":
        tz = ZoneInfo(config.timezone)
        trading_date = (
            date.fromisoformat(args.date)
            if args.date
            else datetime.now(tz).date() - timedelta(days=1)
        )
        output = args.output or (
            f"reports/{config.get('instrument', 'ticker')}_{trading_date.isoformat()}_{args.timeframe}_indicators.png"
        )
        figi, candles = fetch_day_candles(config, logger, trading_date, args.timeframe)
        output_path = plot_indicator_chart(config, candles, trading_date, args.timeframe, output)
        logger.info(
            "chart_ok",
            extra={
                "event": "chart_ok",
                "details": {
                    "figi": figi,
                    "date": trading_date.isoformat(),
                    "timeframe": args.timeframe,
                    "candles": len(candles),
                    "output": str(output_path),
                },
            },
        )
        return 0

    if args.command == "review":
        tz = ZoneInfo(config.timezone)
        trading_date = date.fromisoformat(args.date) if args.date else datetime.now(tz).date()
        paths = generate_review_from_api(
            config,
            logger,
            trading_date=trading_date,
            label=args.label,
            timeframes=args.timeframes,
        )
        logger.info(
            "review_ok",
            extra={
                "event": "review_ok",
                "details": {
                    "date": trading_date.isoformat(),
                    "paths": [str(path) for path in paths],
                },
            },
        )
        return 0

    if args.command == "labeling":
        result = generate_labeling_charts(
            config,
            logger,
            days=args.days,
            minutes=args.minutes,
            timeframe=args.timeframe,
            output_dir=args.output_dir,
            backtest_report=args.backtest_report,
        )
        logger.info(
            "labeling_ok",
            extra={
                "event": "labeling_ok",
                "details": {
                    "figi": result.figi,
                    "paths": [str(path) for path in result.paths],
                    "decisions": len(result.decisions),
                    "regimes": len(result.regimes),
                    "backtest_trades": result.backtest_trades,
                },
            },
        )
        return 0

    if args.command == "daily-oracle":
        tz = ZoneInfo(config.timezone)
        trading_date = date.fromisoformat(args.date) if args.date else datetime.now(tz).date()
        result = generate_daily_oracle_from_api(
            config,
            logger,
            trading_date=trading_date,
            minutes=args.minutes,
            output_dir=args.output_dir,
        )
        logger.info(
            "daily_oracle_ok",
            extra={
                "event": "daily_oracle_ok",
                "details": {
                    "figi": result.figi,
                    "date": result.trading_date,
                    "candles_1m": result.candles_1m,
                    "candles_15m": result.candles_15m,
                    "oracle_trades": len(result.oracle_trades),
                    "sideways_labels": result.sideways_labels,
                    "json": str(result.json_path),
                    "labels_csv": str(result.labels_csv_path),
                },
            },
        )
        return 0

    if args.command == "train-feedback":
        report = train_feedback_from_api(
            config,
            logger,
            minutes=args.minutes,
            output_path=args.output,
            min_examples=args.min_examples,
        )
        logger.info(
            "train_feedback_ok",
            extra={
                "event": "train_feedback_ok",
                "details": {
                    "figi": report.figi,
                    "candles": report.candles,
                    "output_path": str(report.report.path),
                    "examples": report.total_examples,
                    "usable_examples": report.report.examples,
                    "models": report.report.models,
                    "classes": report.report.classes,
                    "holdout_accuracy": report.report.holdout_accuracy,
                    "promotion_score": report.report.promotion_score,
                    "promoted": report.report.promoted,
                    "generated_examples": report.generated_examples,
                },
            },
        )
        return 0

    if args.command == "train-candidate":
        output = args.output or config.get("learning", "candidate_model_path")
        report = train_feedback_from_api(
            _candidate_training_config(config),
            logger,
            minutes=args.minutes,
            output_path=output,
            min_examples=args.min_examples,
        )
        logger.info(
            "train_candidate_ok",
            extra={
                "event": "train_candidate_ok",
                "details": {
                    "figi": report.figi,
                    "candles": report.candles,
                    "output_path": str(report.report.path),
                    "examples": report.total_examples,
                    "usable_examples": report.report.examples,
                    "models": report.report.models,
                    "classes": report.report.classes,
                    "promotion_status": report.report.promotion_status,
                    "promotion_score": report.report.promotion_score,
                    "promoted": report.report.promoted,
                    "generated_examples": report.generated_examples,
                },
            },
        )
        return 0

    if args.command == "promotion-check":
        model_path = _model_path_from_selector(config, args.model)
        report = check_model_eligibility(config, model_path=model_path)
        save_promotion_check(report, args.report)
        logger.info(
            "promotion_check_ok",
            extra={
                "event": "promotion_check_ok",
                "details": {
                    "report": args.report,
                    "model_path": report.model_path,
                    "ready": report.ready,
                    "reason": report.reason,
                    "details": report.details,
                },
            },
        )
        return 0

    if args.command == "shadow-evaluate":
        report = evaluate_shadow_predictions(
            config,
            decisions_path=args.decisions,
            labels_path=args.labels,
        )
        report_path = args.report or str(config.get("shadow", "report_file", default="reports/shadow/shadow_evaluation.json"))
        save_shadow_report(report, report_path)
        logger.info(
            "shadow_evaluate_ok",
            extra={
                "event": "shadow_evaluate_ok",
                "details": {
                    "report": report_path,
                    "predictions": report.predictions,
                    "shadow_trade_signals": report.shadow_trade_signals,
                    "matured_labels": report.matured_labels,
                    "passed": report.passed,
                    "reason": report.reason,
                },
            },
        )
        return 0

    if args.command == "feature-audit":
        report = audit_feature_completeness(
            config,
            decisions_path=args.decisions,
            market_path=args.market,
        )
        save_feature_completeness_report(report, args.report)
        logger.info(
            "feature_audit_ok",
            extra={
                "event": "feature_audit_ok",
                "details": {
                    "report": args.report,
                    "decisions": report["decisions"],
                    "feature_complete_records": report["feature_complete_records"],
                    "trainable_decision_records": report["trainable_decision_records"],
                    "missing_feature_records": report["missing_feature_records"],
                },
            },
        )
        return 0

    if args.command == "regime-report":
        report = generate_regime_report(
            config,
            decisions_path=args.decisions,
            labels_dir=args.labels_dir,
            folds=args.folds,
        )
        save_regime_report(report, args.report)
        logger.info(
            "regime_report_ok",
            extra={
                "event": "regime_report_ok",
                "details": {
                    "report": args.report,
                    "matured_feature_rows": report["matured_feature_rows"],
                    "regimes": {
                        key: value["trades"]
                        for key, value in report["by_regime"].items()
                    },
                },
            },
        )
        return 0

    if args.command == "model-diagnostics":
        model_path = _model_path_from_selector(config, args.model)
        report = generate_model_diagnostics(
            config,
            model_path=model_path,
            decisions_path=args.decisions,
            labels_dir=args.labels_dir,
            thresholds=tuple(args.thresholds),
            top_features=args.top_features,
        )
        save_model_diagnostics(report, args.report)
        logger.info(
            "model_diagnostics_ok",
            extra={
                "event": "model_diagnostics_ok",
                "details": {
                    "report": args.report,
                    "model_path": report["model_path"],
                    "examples": report["model"].get("examples_total_heads"),
                    "label_distribution": report["label_distribution"],
                    "current_target": report["current_candidate"].get("target"),
                    "thresholds": report["threshold_replay"],
                },
            },
        )
        return 0

    if args.command == "run":
        bot = TradingBot(config, logger)
        try:
            bot.run_forever()
        except KeyboardInterrupt:
            logger.info("shutdown_requested", extra={"event": "shutdown_requested", "details": {}})
            return 0
    return 0


def _model_path_from_selector(config, selector: str) -> Path:
    if selector == "active":
        return Path(config.get("learning", "ensemble_model_path"))
    if selector == "candidate":
        return Path(config.get("learning", "candidate_model_path"))
    return Path(selector)


def _candidate_training_config(config: RuntimeConfig) -> RuntimeConfig:
    raw = copy.deepcopy(config.raw)
    raw.setdefault("learning", {})["promote_enabled"] = False
    raw["learning"]["ensemble_model_path"] = str(
        raw["learning"].get("candidate_model_path", "data/models/feedback_ensemble.candidate.joblib")
    )
    return RuntimeConfig(raw=raw, path=config.path)


if __name__ == "__main__":
    sys.exit(main())
