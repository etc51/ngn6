# ngn6_bot.monitoring

## Purpose
Monitoring helpers for model and feature drift.

## Contents
- `drift.py` - drift metrics and actions for warning/block states.
- `__init__.py` - package marker.

## Rules
- Drift blocks must disable ML entries or require re-promotion, not silently continue.
- Monitoring code should produce machine-readable details suitable for reports.
- Keep thresholds configurable from `config/ngn6.yaml`.

## Quick Checks
- Relevant tests: `python -m pytest tests/test_self_learning_safety.py -q`
