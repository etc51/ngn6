# ngn6_bot.learning

## Purpose
Machine-learning, labeling, validation, training, shadow evaluation, and promotion logic.

## Contents
- `feedback_model.py` - runtime ML controller and model eligibility checks.
- `ensemble.py` - model training artifact format and prediction ensemble.
- `training.py` - candidate training, promotion decision, walk-forward evaluation.
- `daily_oracle.py` - post-session oracle label generation and scheduled retraining.
- `labeling.py` - chart-assisted labeling utilities.
- `triple_barrier.py` - label maturity and entry outcome logic.
- `purged_walk_forward.py` - leakage-aware validation splits.
- `shadow.py` - candidate shadow predictions versus matured labels.
- `promotion.py` - explicit model promotion eligibility reports.

## Rules
- Candidate models are not allowed to trade.
- Active models must pass schema, head, class, sample, shadow, and OOS promotion gates.
- Do not train on future information. Labels may use future candles; features must be rebuilt at label time only.
- Keep promotion reports explainable: include counts, gates, reasons, and `commit_hash`.
- Never silently copy a candidate over the active model without promotion approval.

## Quick Checks
- Candidate check: `python -m ngn6_bot.cli promotion-check --config config/ngn6.yaml --model candidate`
- Shadow check: `python -m ngn6_bot.cli shadow-evaluate --config config/ngn6.yaml`
- Candidate training: `python -m ngn6_bot.cli train-candidate --config config/ngn6.yaml`
