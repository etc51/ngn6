# tests

## Purpose
Automated tests for bot safety, trading logic, ML gates, reporting, and integrations.

## Contents
- `test_self_learning_safety.py` - strict safety gates, promotion, shadow, fail-closed behavior.
- `test_training_promotion.py`, `test_walk_forward.py`, `test_feedback_model.py` - ML training and validation behavior.
- `test_backtest.py`, `test_tradeflow.py`, `test_signals.py`, `test_risk.py`, `test_costs.py` - strategy and risk behavior.
- `test_paper.py`, `test_runtime_metadata.py` - paper portfolio and commit-hash report attribution.
- `test_strategy_audit.py` - paper-event pairing, P&L reconciliation, and audit report output.
- `test_config*.py`, `test_token_normalization.py`, `test_tbank_migration.py` - config and API safety.
- Other files test indicators, orderbook, microstructure replay, daily oracle, and legacy bridge behavior.

## Rules
- Any change to entry/exit/risk/model-gating logic needs tests.
- Any new runtime report field should have a format test.
- Do not use live API calls in unit tests.
- Keep tests deterministic and independent of current market state.

## Quick Checks
- Full suite: `python -m pytest -q`
- Safety subset: `python -m pytest tests/test_self_learning_safety.py -q`
