# reference/ngn6_signal_source/bridge

## Purpose
Bridge code for calling the legacy JavaScript signal engine from the Python bot.

## Contents
- `compute_signal.js` - command-line bridge that accepts JSON input and returns a signal payload.

## Rules
- Keep bridge input/output JSON stable.
- Do not add secrets or network side effects here.
- Changes must be covered by Python legacy bridge tests and, when relevant, Node tests.

## Quick Checks
- Python integration tests: `python -m pytest tests/test_legacy_signal.py -q`
