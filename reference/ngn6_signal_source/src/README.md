# reference/ngn6_signal_source/src

## Purpose
Legacy Node.js runtime source used as a reference for signal generation and old watcher behavior.

## Contents
- `server.js` - legacy service entry point.
- `signalWatcher.js` - old watcher loop and signal workflow.
- `paperTrader.js` - legacy paper trading state.
- `lib/` - reusable legacy modules.

## Rules
- This is not the active Python runtime.
- Preserve behavior when using it as a regression reference.
- Prefer porting concepts into tested Python modules instead of adding new production logic here.

## Quick Checks
- Active Python runtime starts from `ngn6_bot/cli.py`.
