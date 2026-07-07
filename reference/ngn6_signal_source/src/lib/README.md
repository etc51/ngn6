# reference/ngn6_signal_source/src/lib

## Purpose
Legacy JavaScript support modules for the old signal source.

## Contents
- `signalEngine.js` - legacy signal calculations.
- `marketData.js`, `tbankClient.js` - legacy market-data access.
- `config.js` - legacy configuration loader.
- `newsFeed.js`, `officialCalendar.js`, `autoContext.js` - legacy context helpers.
- `paperTrader.js` lives one level up and consumes these helpers.

## Rules
- Do not treat these modules as authoritative for current safety gates.
- If a legacy rule is ported, add Python tests in `tests/`.
- Keep Node dependency changes isolated to `reference/ngn6_signal_source/package*.json`.

## Quick Checks
- Legacy Node tests are under `reference/ngn6_signal_source/test/`.
