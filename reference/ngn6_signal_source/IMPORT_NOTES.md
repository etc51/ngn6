# NGN6 Signal Source Import

Copied from: `C:\Users\HONOR\Documents\NGN6`
Copied on: `2026-06-30`

This directory is a source snapshot for porting the NGN6 signal logic into the Python bot.

Included:
- `src/` - signal engine, watcher, market context, news/context helpers, T-Bank REST client, and paper-trading logic.
- `test/` - Node.js tests that document expected signal, plan, watcher, and paper-trader behavior.
- `package.json` and `package-lock.json` - runtime/test metadata for the copied Node.js logic.
- `README.md` and `.env.example` - source project documentation and non-secret environment template.

Excluded:
- `.env` - real local secrets/config.
- `.git` - source repository metadata.
- `node_modules/` - generated dependencies.
- `data/` and `logs/` - runtime state and market/event logs.
- `public/`, `scripts/`, `Dockerfile`, and `docker-compose.yml` - dashboard/deployment infrastructure, not trading logic.

Core trading files:
- `src/lib/signalEngine.js` - feature calculation, direction scoring, probability, trade levels, trade plan, backtest.
- `src/signalWatcher.js` - executable entry/stop/take-profit plan, impulse overlay, daily/intraday signal gating, emergency close and TP1 stop management.
- `src/paperTrader.js` - paper fills from order book, stop/target handling, partial TP1 close, stop movement, PnL/event accounting.
- `src/lib/autoContext.js` - automatic market structure, retest, event/news context.
- `src/lib/marketData.js`, `src/lib/tbankClient.js`, `src/lib/newsFeed.js`, `src/lib/timeframes.js` - data inputs used by the signal stack.

